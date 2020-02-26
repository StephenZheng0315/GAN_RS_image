# 遥感影像生成

这是一个基于GAN利用tensorflow模块实现的RS图像生成项目。



##导言

遥感是指在不与物体发生接触的情况下获取有关物体或现象的信息，因而与现场观测（特别是地球）形成对照。

遥感图像的场景分类在许多遥感图像应用中的作用。训练一个好的分类器需要大量的训练样本。而分类好的样品文件通常很难获得。因此，我利用生成性对抗网络来生成(128*128*3)遥感图像。



## 主要工作：

将大小为100的随机噪声矢量提供给生成器。生成器有4个转置卷积层，每一层之后是批处理正规化和泄漏无关性。再次卷积第4层的输出以获得形状为128x128x3的图像向量。生成的图像随后被传递到由五个卷积层组成的鉴别器网络中，每个卷积层还伴随着批处理归一化和泄漏非线性。第五层的输出被压平并压缩成一个单一的概率值。基于此值，对模型进行了优化。

### Sample Images
![alt text](https://github.com/StephenZheng0315/GAN_RS_image/blob/master/sample_images/01.png)
![alt text](https://github.com/StephenZheng0315/GAN_RS_image/blob/master/sample_images/02.png)
![alt text](https://github.com/StephenZheng0315/GAN_RS_image/blob/master/sample_images/03.png)
![alt text](https://github.com/StephenZheng0315/GAN_RS_image/blob/master/sample_images/04.png)
![alt text](https://github.com/StephenZheng0315/GAN_RS_image/blob/master/sample_images/05.png)

### Losses

![alt text](https://github.com/StephenZheng0315/GAN_RS_image/blob/master/sample_images/losses_198.png)



### 平台: Python 3.7

## 所需资源库:
* __tensorflow__ 1.14
* __matplotlib__ 3.0.3
* ______PIL_____ 4.3.0
* _____numpy____ 1.16
* _____glob2____ 0.7
* ______os______
* _____time_____
* _____tqdm_____
* _____urllib___


#### 我在特斯拉T4 GPU上训练了这个模型，花了大约50分钟完成了100个epoch。
#### 在CPU上訓練，1 epoch大约需要25-30分钟。
#### 注意：如果在下载RSI-CB数据集时遇到HTTP错误，请与联系

## 参考:
* Model architecture: 'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks' by Alec   Radford, Luke Metz, Soumith Chintala
* Dataset: RSI-CB (A Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data)
* www.tensorflow.org/api_docs/python/tf/layers
* www.towardsdatascience.com
