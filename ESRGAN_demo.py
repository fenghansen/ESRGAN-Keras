#! /usr/bin/python
import os
import sys
import pickle
import datetime
import numpy as np
# Import keras + tensorflow without the "Using XX Backend" message
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Add, Concatenate
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense, SeparableConv2D
from keras.layers import UpSampling2D, Lambda, Dropout, Flatten
from keras.optimizers import Adam
# from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
# from keras.applications import densenet
# from keras.applications.densenet import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from vgg19_noAct import VGG19
from util import DataLoader, plot_test_images, plot_bigger_images, compute_metric
from ESRGAN import SRGAN


class ESRGAN_Demo(SRGAN):
    """
    Implementation of ESRGAN as described in the paper:
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    https://arxiv.org/abs/1809.00219
    """
    def test(self,
        refer_model=None,
        batch_size=1,
        datapath_test='./images/inputs',
        crops_per_image=1,
        log_test_path="./images/outputs",
        model_name='',
    ):
        # Create data loaders
        loader = DataLoader(
            datapath_test, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image
        )
        e = -1
        print(">> Ploting test images")
        if self.refer_model is not None:
            refer_model = self.refer_model
        plot_bigger_images(self, loader, datapath_test, log_test_path, e, refer_model=refer_model)

    def psnr_and_ssim(self,
                      num,
                      batch_size=1,
                      crops_per_image=1,
                      datapath_test='./images/inputs',
                      log_test_path="./images/outputs",
                      ):
        # Create data loaders
        loader = DataLoader(
            datapath_test, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image
        )
        print(">> Computing PSNR and SSIM")
        return compute_metric(self, loader, datapath_test, log_test_path, num)


# Run the SRGAN network
if __name__ == '__main__':
    nums = []
    psnrs = [[], [], [], []]
    ssims = [[], [], [], []]
    # Instantiate the SRGAN object
    mode = 'ESRGAN'
    if mode == 'ESRGAN':
        RDDB = SRGAN(training_mode=False)
        RDDB.generator.load_weights(r'./data/weights/DIV2K_gan.h5')

        print(">> Creating the ESRGAN network")
        gan = SRGAN(training_mode=False,
                    refer_model=RDDB.generator,
                    )
        gan.generator.load_weights(r'./data/weights/DIV2K_generator_4X_epoch65000.h5')
        gan.test(datapath_test='./images/inputs',
                 log_test_path="./images/outputs",
                )

    elif mode == 'SR-RRDB':
        RDDB = SRGAN(training_mode=False)
        print(">> Creating the ESRGAN network")
        gan = SRGAN(training_mode=False,
                    )
        gan.generator.load_weights(r'./data/weights/SRGAN-D_generator_MRI.h5')
        gan.test(datapath_test='./images/inputs',
                 log_test_path="./images/outputs",
                 )