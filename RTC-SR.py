#! /usr/bin/python
import os
import sys
import pickle
import datetime

import numpy as np

# Import keras + tensorflow without the "Using XX Backend" message
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Add, Concatenate, MaxPooling2D, Cropping2D
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense, SeparableConv2D
from keras.layers import UpSampling2D, Lambda, Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
# from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
# from keras.applications import densenet
# from keras.applications.densenet import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

sys.stderr = stderr
from attention_keras import SelfAttention
from vgg19_noAct import VGG19
from util import DataLoader, plot_test_images, plot_bigger_images, plot_test_only


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.99):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))

    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))


def spectral_normalization(w):
    return w / spectral_norm(w)


class SRGAN():
    """
    Implementation of SRGAN as described in the paper:
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    https://arxiv.org/abs/1609.04802
    """

    def __init__(self,
                 height_lr=24, width_lr=24, channels=3,
                 upscaling_factor=4,
                 gen_lr=1e-4, dis_lr=1e-4,
                 # VGG scaled with 1/12.75 as in paper
                 loss_weights=[0.005, 0.005],  # gan, percept, [pixel(2e-3)]
                 training_mode=True,
                 refer_model=None,
                 use_EMA=False,
                 ):
        """
        :param int height_lr: Height of low-resolution images
        :param int width_lr: Width of low-resolution images
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        """

        self.training_mode = training_mode

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor not in [2, 4, 8]:
            raise ValueError('Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        # Scaling of losses
        self.loss_weights = loss_weights

        # Gan setup settings
        self.gan_loss = 'mae'
        self.dis_loss = 'binary_crossentropy'

        # Build & compile the generator network
        # self.generator = self.build_generator()
        self.use_EMA = use_EMA
        self.EMAer = None
        self.refer_model = refer_model
        # if refer_model is None:
        #     self.generator = self.build_old_RRDB()
        # else:
        self.generator = self.build_RRDB(sn=True)
        self.compile_generator(self.generator)

        # If training, build rest of GAN network
        if training_mode:
            self.vgg = self.build_vgg()
            self.compile_vgg(self.vgg)
            self.discriminator = self.build_discriminator()
            self.compile_discriminator(self.discriminator)
            self.srgan = self.build_srgan()
            self.compile_srgan(self.srgan)


    def save_weights(self, filepath, e=None):
        """Save the generator and discriminator networks"""
        if self.use_EMA:self.EMAer.apply_ema_weights()  # 将EMA的权重应用到模型中

        self.generator.save_weights("{}_generator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))
        self.discriminator.save_weights("{}_discriminator_{}X_epoch{}.h5".format(filepath, self.upscaling_factor, e))

        if self.use_EMA:self.EMAer.reset_old_weights()  # 继续训练之前，要恢复模型旧权重。还是那句话，EMA不影响模型的优化轨迹。

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)

    def SubpixelConv2D(self, name, scale=2):
        """
        Keras layer to do subpixel convolution.
        NOTE: Tensorflow backend only. Uses tf.depth_to_space

        :param scale: upsampling scale compared to input_shape. Default=2
        :return:
        """

        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape

        def subpixel(x):
            return tf.depth_to_space(x, scale)

        return Lambda(subpixel, output_shape=subpixel_shape, name=name)

    def build_vgg(self):
        """
        Load pre-trained VGG weights from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        """

        # Input image to extract features from
        img = Input(shape=self.shape_hr)

        # Get the vgg network. Extract features from last conv layer
        vgg = VGG19(weights="imagenet", include_top=False)
        # vgg.summary()
        vgg.outputs = [vgg.layers[20].output]
        print('output layer is :', vgg.layers[20].name)
        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        return model

    def preprocess_vgg(self, x):
        """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
        if isinstance(x, np.ndarray):
            return preprocess_input((x + 1) * 127.5)
        else:
            return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)

    def build_old_RRDB(self, sn=False):
        """
        Build the generator network according to description in the paper.
        :return: the compiled model
        """
        sn_layer = spectral_normalization if sn else None

        def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  # 这里跟论文原图有冲突，论文没x3???

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same',
                       name='upSampleConv2d_' + str(number), kernel_constraint=sn_layer)(x)
            x = self.SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
            x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(None, None, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])

        # Upsampling depending on factor
        x = upsample(x, 1)
        if self.upscaling_factor > 2:
            x = upsample(x, 2)
        if self.upscaling_factor > 4:
            x = upsample(x, 3)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        # model.summary()
        return model

    def build_RRDB(self, sn=False):
        """
        Build the generator network according to description in the paper.
        :return: the compiled model
        """
        sn_layer = spectral_normalization if sn else None

        def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  # 这里跟论文原图有冲突，论文没x3???

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same',
                       name='upSampleConv2d_' + str(number), kernel_constraint=sn_layer)(x)
            x = self.SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
            x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
            return x

        def get_crop_shape(target):
            # width, the 3rd dimension
            cw = tf.cast(tf.shape(target)[2], tf.int32)
            # height, the 2nd dimension
            ch = tf.cast(tf.shape(target)[1], tf.int32)
            return ch, cw

        # Input low resolution image
        lr_input = Input(shape=(None, None, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x_RRDB = Add()([x, x_start])

        # x = MaxPooling2D(padding="same")(x_RRDB)
        # x = SelfAttention(64)(x)
        # x = UpSampling2D()(x)
        # x = upsample(x, 0)
        # if self.training_mode is not True:
        #     x = Lambda(lambda x:tf.image.resize_images(x, get_crop_shape(x_RRDB), method=1), name="Resize")(x)

        # Upsampling depending on factor
        x = upsample(x, 1)
        if self.upscaling_factor > 2:
            x = upsample(x, 2)
        if self.upscaling_factor > 4:
            x = upsample(x, 3)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_constraint=sn_layer)(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        model.summary()
        return model

    def build_generator(self, residual_blocks=16):
        """
        Build the generator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int residual_blocks: How many residual blocks to use
        :return: the compiled model
        """

        def residual_block(input):
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x = BatchNormalization(momentum=0.8)(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Add()([x, input])
            return x

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
            x = self.SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
            x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(None, None, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)
        x_start = PReLU(shared_axes=[1, 2])(x_start)

        # Residual blocks
        r = residual_block(x_start)
        for _ in range(residual_blocks - 1):
            r = residual_block(r)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, x_start])

        # Upsampling depending on factor
        x = upsample(x, 1)
        if self.upscaling_factor > 2:
            x = upsample(x, 2)
        if self.upscaling_factor > 4:
            x = upsample(x, 3)

        # Generate high resolution output
        # tanh activation, see:
        # https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
        hr_output = Conv2D(
            self.channels,
            kernel_size=9,
            strides=1,
            padding='same',
            activation='tanh'
        )(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        # model.summary()
        return model

    def build_discriminator(self, filters=64):
        """
        Build the discriminator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv2d_block(input, filters, strides=1, bn=True, sn=True):
            sn_layer = spectral_normalization if sn else None
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same',
                       kernel_constraint=sn_layer)(input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # Input high resolution image
        img = Input(shape=self.shape_hr)
        x = conv2d_block(img, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters * 2)
        x = conv2d_block(x, filters * 2, strides=2)
        x = conv2d_block(x, filters * 4)
        x = conv2d_block(x, filters * 4, strides=2)
        x = conv2d_block(x, filters * 8)
        x = conv2d_block(x, filters * 8, strides=2)
        # x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        x = Dense(filters * 16)(x)
        # x = Dropout(0.2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x)
        # model.summary()
        return model

    def build_srgan(self):
        """Create the combined SRGAN network"""

        # Input LR images
        img_lr = Input(self.shape_lr)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        generated_features = self.vgg(
            self.preprocess_vgg(generated_hr)
        )

        # In the combined model we only train the generator
        self.discriminator.trainable = False

        # Determine whether the generator HR images are OK
        generated_check = self.discriminator(generated_hr)

        # Create sensible names for outputs in logs
        generated_features = Lambda(lambda x: x, name='Content')(generated_features)
        generated_check = Lambda(lambda x: x, name='Adversarial')(generated_check)
        # generated_hr = Lambda(lambda x: x, name='pixel')(generated_hr)

        # Create model and compile
        # Using binary_crossentropy with reversed label, to get proper loss, see:
        # https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
        model = Model(inputs=img_lr, outputs=[generated_check, generated_features])
        return model

    def compile_vgg(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(self.gen_lr, 0.9),
            metrics=['accuracy']
        )

    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(self.gen_lr, 0.9),
            metrics=[self.gan_loss, self.PSNR]
        )

    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.dis_loss,
            optimizer=Adam(self.dis_lr, 0.9),
            metrics=['accuracy']
        )

    def compile_srgan(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=[self.dis_loss, self.gan_loss],
            loss_weights=self.loss_weights,
            optimizer=Adam(self.gen_lr, 0.9)
        )

    def PSNR(self, y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    def train_generator(self,
                        epochs, batch_size,
                        workers=1,
                        dataname='train_gen',
                        datapath_train='./train_dir',
                        datapath_validation='./val_dir',
                        datapath_test='./val_dir',
                        steps_per_epoch=1000,
                        steps_per_validation=1000,
                        crops_per_image=4,
                        log_weight_path='./data/weights/',
                        log_tensorboard_path='./data/logs/',
                        log_tensorboard_name='SA-RRDB',
                        log_tensorboard_update_freq=1,
                        log_test_path="./images/samples/"
                        ):
        """Trains the generator part of the network with MSE loss"""

        # Create data loaders
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image
        )
        test_loader = None
        if datapath_validation is not None:
            test_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image
            )

        self.gen_lr = 4e-4
        for step in range(epochs // 10):
            self.compile_generator(self.generator)
            # Callback: tensorboard
            callbacks = []
            if log_tensorboard_path:
                tensorboard = TensorBoard(
                    log_dir=os.path.join(log_tensorboard_path, log_tensorboard_name),
                    histogram_freq=0,
                    batch_size=batch_size,
                    write_graph=False,
                    write_grads=False,
                    update_freq=log_tensorboard_update_freq
                )
                callbacks.append(tensorboard)
            else:
                print(">> Not logging to tensorboard since no log_tensorboard_path is set")

            # Callback: save weights after each epoch
            modelcheckpoint = ModelCheckpoint(
                os.path.join(log_weight_path, dataname + '_{}X.h5'.format(self.upscaling_factor)),
                monitor='PSNR',
                save_best_only=True,
                save_weights_only=True
            )
            callbacks.append(modelcheckpoint)

            # Callback: test images plotting
            if datapath_test is not None:
                testplotting = LambdaCallback(
                    on_epoch_end=lambda epoch, logs: plot_test_images(
                        self,
                        test_loader,
                        datapath_test,
                        log_test_path,
                        epoch + step * 10,
                        name='SA-RRDB'
                    )
                )
                callbacks.append(testplotting)

            # Fit the model
            self.generator.fit_generator(
                train_loader,
                steps_per_epoch=steps_per_epoch,
                epochs=10,
                validation_data=test_loader,
                validation_steps=steps_per_validation,
                callbacks=callbacks,
                use_multiprocessing=workers > 1,
                workers=workers
            )
            self.generator.save_weights('./data/weights/SA-RRDB(Step %dK).h5' % (step * 10 + 10))
            self.gen_lr /= 2
            print(step, self.gen_lr)

    def train_srgan(self,
                    epochs, batch_size,
                    dataname,
                    datapath_train,
                    datapath_validation=None,
                    steps_per_validation=10,
                    datapath_test=None,
                    workers=40, max_queue_size=100,
                    first_epoch=0,
                    print_frequency=50,
                    crops_per_image=3,
                    log_weight_frequency=1000,
                    log_weight_path='./data/weights/doctor_gan_ct_sn/',
                    log_tensorboard_path='./data/logs/',
                    log_tensorboard_name='RTC-SR',
                    log_tensorboard_update_freq=1000,
                    log_test_frequency=1000,
                    log_test_path="./images/samples-ct-sn/",
                    ):
        """Train the SRGAN network

        :param int epochs: how many epochs to train the network for
        :param str dataname: name to use for storing model weights etc.
        :param str datapath_train: path for the image files to use for training
        :param str datapath_test: path for the image files to use for testing / plotting
        :param int print_frequency: how often (in epochs) to print progress to terminal. Warning: will run validation inference!
        :param int log_weight_frequency: how often (in epochs) should network weights be saved. None for never
        :param int log_weight_path: where should network weights be saved
        :param int log_test_frequency: how often (in epochs) should testing & validation be performed
        :param str log_test_path: where should test results be saved
        :param str log_tensorboard_path: where should tensorflow logs be sent
        :param str log_tensorboard_name: what folder should tf logs be saved under
        """

        # Create train data loader
        loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image
        )

        # Validation data loader
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image
            )
        print("Picture Loaders has been ready.")
        # Use several workers on CPU for preparing batches
        enqueuer = OrderedEnqueuer(
            loader,
            use_multiprocessing=False,
            shuffle=True
        )
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
        print("Data Enqueuer has been ready.")
        # Callback: tensorboard
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, log_tensorboard_name),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=False,
                write_grads=False,
                update_freq=log_tensorboard_update_freq
            )
            tensorboard.set_model(self.srgan)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Callback: format input value
        def named_logs(model, logs):
            """Transform train_on_batch return value to dict expected by on_batch_end callback"""
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        # Shape of output from discriminator
        disciminator_output_shape = list(self.discriminator.output_shape)
        disciminator_output_shape[0] = batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)

        # VALID / FAKE targets for discriminator
        real = np.ones(disciminator_output_shape)
        fake = np.zeros(disciminator_output_shape)

        # Each epoch == "update iteration" as defined in the paper
        print_losses = {"G": [], "D": []}
        start_epoch = datetime.datetime.now()

        # Random images to go through
        idxs = np.random.randint(0, len(loader), epochs)
        if self.use_EMA:self.EMAer = ExponentialMovingAverage(self.srgan)  # 在模型compile之后执行
        if self.use_EMA:self.EMAer.inject()  # 在模型compile之后执行

        # Loop through epochs / iterations
        for epoch in range(first_epoch, int(epochs) + first_epoch):
            # print(epoch)
            # Start epoch time
            if epoch % print_frequency == 1:
                start_epoch = datetime.datetime.now()

            # Train discriminator
            imgs_lr, imgs_hr = next(output_generator)

            if self.use_EMA:self.EMAer.apply_ema_weights()  # 将EMA的权重应用到模型中
            generated_hr = self.generator.predict(imgs_lr)# 进行预测、验证、保存等操作
            if self.use_EMA:self.EMAer.reset_old_weights()  # 继续训练之前，要恢复模型旧权重。还是那句话，EMA不影响模型的优化轨迹。

            real_loss = self.discriminator.train_on_batch(imgs_hr, real)
            fake_loss = self.discriminator.train_on_batch(generated_hr, fake)
            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)

            # Train generator
            features_hr = self.vgg.predict(self.preprocess_vgg(imgs_hr))
            generator_loss = self.srgan.train_on_batch(imgs_lr, [real, features_hr])

            # Callbacks
            logs = named_logs(self.srgan, generator_loss)
            tensorboard.on_epoch_end(epoch, logs)
            # print(generator_loss, discriminator_loss)
            # Save losses
            print_losses['G'].append(generator_loss)
            print_losses['D'].append(discriminator_loss)

            # Show the progress
            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['G']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                print("\nEpoch {}/{} | Time: {}s\n>> Generator/GAN: {}\n>> Discriminator: {}".format(
                    epoch, epochs + first_epoch,
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.srgan.metrics_names, g_avg_loss)]),
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.discriminator.metrics_names, d_avg_loss)])
                ))
                print_losses = {"G": [], "D": []}
                # Run validation inference if specified
                # if datapath_validation:
                #     print(">> Running validation inference")
                #     validation_losses = self.generator.evaluate_generator(
                #         validation_loader,
                #         steps=steps_per_validation,
                #         use_multiprocessing=workers>1,
                #         workers=workers
                #     )
                #     print(">> Validation Losses: {}".format(
                #         ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.generator.metrics_names, validation_losses)])
                #     ))

            # If test images are supplied, run model on them and save to log_test_path
            if datapath_test and epoch % log_test_frequency == 0:
                print(">> Ploting test images")
                if self.use_EMA:self.EMAer.apply_ema_weights()  # 将EMA的权重应用到模型中
                plot_test_images(self, loader, datapath_test, log_test_path, epoch, refer_model=self.refer_model)
                if self.use_EMA:self.EMAer.reset_old_weights()

                # Check if we should save the network weights
            if log_weight_frequency and epoch % log_weight_frequency == 0:
                # Save the network weights
                print(">> Saving the network weights")
                if self.use_EMA:self.EMAer.apply_ema_weights()  # 将EMA的权重应用到模型中
                self.save_weights(os.path.join(log_weight_path, dataname), epoch)
                if self.use_EMA:self.EMAer.reset_old_weights()

    def test(self,
             refer_model=None,
             batch_size=4,
             datapath_test='./images/val_dir',
             crops_per_image=1,
             log_test_path="./images/test/"
             ):
        """Trains the generator part of the network with MSE loss"""

        # Create data loaders
        loader = DataLoader(
            datapath_test, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image
        )
        print(">> Ploting test images")
        plot_test_images(self, loader, datapath_test, log_test_path, 0, refer_model=refer_model)

    def print_test_only(self,
             datapath_test='./images/val_dir',
             log_test_path="./images/test"
             ):
        print(">> Ploting test images")
        plot_test_only(self, datapath_test, log_test_path)


# Run the SRGAN network
if __name__ == '__main__':
    mode = 3
    # Instantiate the SRGAN object
    print(">> Creating the SRGAN network")
    if mode == 1:
        gan = SRGAN(training_mode=False)
        gan.generator.load_weights('./data/weights/SelfAttention/SA-RRDB_4X.h5')
        gan.print_test_only(
            # datapath_test='./small',
            # log_test_path='./big',
            datapath_test='./test-images_x4',
            log_test_path='./images/SRGAN'
        )
    RRDB = SRGAN(training_mode=True)
    RRDB.generator.load_weights('./data/weights/SA-RRDB.h5', by_name=True)
    # sr.load_weights('./DIV2K_generator_4X_epoch71500.h5')
    gan = SRGAN(gen_lr=1e-4, dis_lr=4e-4,
                height_lr=48, width_lr=48,
                refer_model=RRDB.generator,
                use_EMA=False,
                )

    if mode == 2:
        gan.generator.load_weights('./data/weights/SA-RRDB(Step 50K).h5', by_name=True)
        gan.train_generator(
            epochs=40,
            batch_size=40,
            steps_per_epoch=500,
            crops_per_image=3,
            dataname='SA-RRDB',
            datapath_train='./val-images/val-images_original',
            datapath_validation='./images/val_dir_crop',
            datapath_test='./images/val_dir_crop',
            log_test_path='./images/SelfAttention',
            log_weight_path='./data/weights/SelfAttention',
        )
        gan.generator.save_weights('./data/weights/SA-RRDB.h5')

    if mode == 3:
        # Load previous imagenet weights
        print(">> Loading old weights")
        gan.load_weights('./data/weights/woEMA/RTC-SR_generator_4X_epoch18000.h5',
                         # None,)
                         './data/weights/woEMA/RTC-SR_discriminator_4X_epoch18000.h5')
        # gan.generator.load_weights('./data/weights/DIV2K_generator_4X_epoch65000.h5')
        # Train the SRGAN
        gan.train_srgan(
            epochs=40000,
            first_epoch=0,
            batch_size=24,
            crops_per_image=3,
            dataname='RTC-SR',
            datapath_train=r'./val-images/val-images_original',
            datapath_validation='./images/val_dir',
            datapath_test='./images/val_dir/',
            log_test_path='./images/Flickr',
            log_weight_path='./data/weights/Flickr',
            log_weight_frequency=500,
            log_test_frequency=500,
            print_frequency=25,
        )
        gan.save_weights('./data/weights/')

