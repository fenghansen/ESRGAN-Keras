# ESRGAN-Keras
### Recurring the ESRGAN(https://arxiv.org/abs/1809.00219) with Keras

### It was programed based on https://github.com/MathiasGruber/SRGAN-Keras, and refered from https://github.com/SavaStevanovic/ESRGAN

*Now it is not finished, there are several bugs, so please don't directly use ESRGAN.py. You'd butter only regard it as a reference!!*

I have upload the weights of my generator model(RRDB). You can use it after copying my generator model code. If you don't copy the model code, it may report some errors beacause I used 'tf.xxx' in my model.
To be honest, I recommand you to train your own discriminator due to the discriminator has several types and visions, and I'm not sure which one is better. You can add a Dropout(0.4) layer at the last of the discriminator to keep the train process stable.

My recurrence doesn't use RaGAN due to bugs. I think my code maybe have some bugs I couldn't understand.
What's more, I use DIV2K datasets only. After doing experiments, I'm sure that **'the more high quality data, the better model performance' is TRUE**.
**the examples of my ESRGAN(without RaGAN)**
![Baboon in Set14](https://github.com/fenghansen/ESRGAN-Keras/edit/master/img_001_SRF_4_HR-Epoch99000.png) 
![Zebra in Set14](https://github.com/fenghansen/ESRGAN-Keras/edit/master/img_014_SRF_4_HR-Epoch99000.png) 
