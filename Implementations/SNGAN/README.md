## Spectral Normalization GAN

Goal of SNGAN is to train robust discriminator.
This is done by regularizing discriminator with  singular value.



#### [Implementation detail]
Dataset: cifar10
Implemented [Table. 3] on the paper.



#### [Environment]
TITAN RTX
python                    3.6.13 
tensorboardx              2.2 
pytorch                   1.8.1           py3.6_cuda11.1_cudnn8.0.5_0



#### [Generated Output]
![Generated outputs](https://user-images.githubusercontent.com/41980618/125380461-2e542780-e3cd-11eb-817f-3230db87e7b1.gif)



#### [Generator loss & Discriminator loss]

![loss](https://user-images.githubusercontent.com/41980618/125381048-319be300-e3ce-11eb-8424-3949b7955a5b.png)

