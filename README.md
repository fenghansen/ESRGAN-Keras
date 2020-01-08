# ESRGAN-Keras
## Implement of ESRGAN(https://arxiv.org/abs/1809.00219) with Keras  
#### It was programed based on https://github.com/MathiasGruber/SRGAN-Keras, and refered from https://github.com/SavaStevanovic/ESRGAN 
#### ESRGAN_demo.py has released. You can try your images by changing images under ./images/inputs.

***It is not finished totally, there are several bugs, but you can try to run ESRGAN_demo.py to have a test.   
If you want to initially train a pleasant result by yourself, you'd better use 'RTC-SR.py' for a moment.***  
  

## Environment: Python 3.6 + Keras 2.2.4 + Tensorflow 1.12 + PyCharm 2018
### I have uploaded the conda_list.txt, which can make sure your environment works.(TF 1.14)

I have uploaded the weights of my generator model(RRDB). You can use it after copying my generator model code. If you don't copy the model code, it may report some errors beacause I used 'tf.xxx' in my model.

To be honest, I recommand you to train your own discriminator due to the discriminator has several versions, and I'm not sure which one is better. You can add a Dropout(0.4) layer at the last of the discriminator to keep the training process stable (you'd better gradually decrease it to zero at last).

My Implement now **doesn't use RaGAN due to bugs**.  
*Thanks **@Zeyu_Lian** for telling me that I forgot add **K.sigmoid() in ESRGAN.py(line 322~323)**. What's more, he pointed out that **"The last layer of the generator show have used filter with kernal size 9, not 3."** However, I'm suffering with my final exam(都是鸽的借口）, so I will try to fix bugs and test them later.*  

What's more, I use DIV2K datasets only. After doing experiments, I'm sure that **"the more high quality data, the better model performance" is TRUE**.

**the examples of my ESRGAN (without RaGAN)**  
Don't mind the name of the third subtitle 'SRGAN'. It should be 'ESRGAN' (I forgot to change at that time).
#### Baboon in Set14
![Baboon in Set14](https://github.com/fenghansen/ESRGAN-Keras/blob/master/images/show/img_001_SRF_4_HR-Epoch99000.png)  
#### Zebra in Set14
![Zebra in Set14](https://github.com/fenghansen/ESRGAN-Keras/blob/master/images/show/img_014_SRF_4_HR-Epoch99000.png)  
***The next two figures show how ESRGAN directly super-resolute the actual natural image. These two images are cropped from the original image of the DIV2K dataset. There is no "super-resolution original image" that can be compared, so their effects can truly reflect the actual application effect of super-resolution, rather than the reconstruction effect.***
#### 002-(4,5) in DIV2K
![002-(4,5) in DIV2K](https://github.com/fenghansen/ESRGAN-Keras/blob/master/images/outputs/2-(4,5).png)  
#### 050-(2,2) in DIV2K
![050-(2,2) in DIV2K](https://github.com/fenghansen/ESRGAN-Keras/blob/master/images/outputs/50-(2,2).png)  


**Other pictures' PSNR and SSIM are higher than these, but I think it is more clear. Don't mind it too much if you don't want to use it on security field and medical field.**  
  
### Hint
1. I deleted the **Flatten()** to achieve a FCN structure, and it is equal to use **GlobalAveragePooling()**. It is ok for the discriminator in order to use a progressive training method, which is more stable and efficient.
2. You'd better add **SN layer** in D(actually G is ok, too) to optimize the stability while training(I use it in **RTC_SR.py**). For the details, you can read https://kexue.fm/archives/6051#Keras%E5%AE%9E%E7%8E%B0.  
