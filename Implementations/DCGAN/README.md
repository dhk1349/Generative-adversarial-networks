## Deep Convolution GAN
[[paper](https://arxiv.org/abs/1511.06434)]

**Goal: to make stable GAN**



**[Changes: 2016년 당시에는 그랬다고 한다.]**

Deterministic pooling function → up/downscaling with learnable parameters

Trend for not using FC layer + GAP is stable but converges slowly →First layer of G as projection layer + last layer of D with sigmoid

BN at every layer: oscillation, mode instability → Not in G's output layer, D's input layer

BN: soothes poor initialization, enable gradient to flow deeper, prevent mode collapse

Discriminator worked better with Leaky ReLU especially on high resolution imgs



**[Training Detail]**

Image pixels are pre-processed(mapped) [-1, 1]

Mini-batch:128

SGD

Weight init: (0, 0.02)

Leaky ReLU: 0.2

Adam: lr→0.0002, $\beta_1$→0.5



#### Discriminator and Generator Loss





#### Generated output

