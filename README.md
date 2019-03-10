# ESRGAN-Keras
### Recurring the ESRGAN(https://arxiv.org/abs/1809.00219) with Keras

### It was programed based on https://github.com/MathiasGruber/SRGAN-Keras, and refered from https://github.com/SavaStevanovic/ESRGAN

***It is not finished totally, there are several bugs, so please don't directly use ESRGAN.py.   
You'd butter only regard it as a reference!!! Especially the weights of losses!!  
I really don't know which number is able to use.  
=_= -> QAQ -> orz***

#### Environment: Python 3.6 + Keras 2.2.4 + Tensorflow 1.12 + PyCharm 2018

I have upload the weights of my generator model(RRDB). You can use it after copying my generator model code. If you don't copy the model code, it may report some errors beacause I used 'tf.xxx' in my model.

To be honest, I recommand you to train your own discriminator due to the discriminator has several types and visions, and I'm not sure which one is better. You can add a Dropout(0.4) layer at the last of the discriminator to keep the train process stable.

My recurrence doesn't use RaGAN due to bugs. I think my code maybe have some bugs I couldn't understand.

What's more, I use DIV2K datasets only. After doing experiments, I'm sure that **'the more high quality data, the better model performance' is TRUE**.

**the examples of my ESRGAN(without RaGAN)**  
#### Baboon in Set14
![Baboon in Set14](https://github.com/fenghansen/ESRGAN-Keras/blob/master/img_001_SRF_4_HR-Epoch99000.png)  
#### Zebra in Set14
![Zebra in Set14](https://github.com/fenghansen/ESRGAN-Keras/blob/master/img_014_SRF_4_HR-Epoch99000.png)  
**Other pictures' PSNR and SSIM is higher than these, but I think it is more clear. Don't mind it too much if you don't want to use it on security field and medical field.**  
