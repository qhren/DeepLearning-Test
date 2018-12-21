# DeepLearning_Test
-------------------------------------------------------------------------------------------------------------------------------------------
## Fully Connected Network in Mnist
-------------------------------------------------------------------------------------------------------------------------------------------
### Pure Python with Numpy
+ 实现了全连接网络中的BP算法，并结合**Mnist**数据集，进行了仿真
+ 修改了网络结构的超参数，比较不同的**epochs**，**mini_batch_size**，**learning_rate**对训练的影响
+ 重点比较了交叉熵和平方误差作为**Cost Function**时，对训练的影响，对双变量时的二者的误差Surf进行了仿真
 
### Build A Simple Net in TensorFlow
+ 在**TensorFlow**框架的支持下，编写了一个简单的全连接网络
+ 了解了Tensorboard可视化**Compute Graph**和对**Train**时的**Cost**的监控，以及对网络中的**Paras**进行监控的方法
+ To be Continued:**TensorFlow中的调试器**

-------------------------------------------------------------------------------------------------------------------------------------------
## Convolution NetWork in Mnist
-------------------------------------------------------------------------------------------------------------------------------------------
### Pure Python with Numpy
+ 参考网络上的相关博客以及教程，对**CNN**的基本结构特点有了基本的了解，了解了它被提出的**Motivation**
+ **对照全连接网络**，比较反向传播的不同点，推导了反向传播的**Mathmatical Form**
+ 对**CNN**中的**Backprop**算法进行实现，测试在**Mnist DataSheet**上的影响

####  To be Continued:
+ 仿真时程序串行运行的效率太低，跑完大样本需要的时间太长，需要考虑能否使用**多进程**进行优化
+ 通过对CNN学习到的**Feature Map**进行[**Deconv**](https://github.com/qhren/DeepLearning_Test/tree/master/Classification%20on%20Mnist%20Data/A_Simple_Convolutional_Network/Visualization_CNN_Network%E5%8F%82%E8%80%83%E8%AE%BA%E6%96%87)，严格意义上来说，论文中使用的是**Transpose Conv**，可类比于卷积层的误差测度的反向传播，作特征的上采样，考虑到自编的CNN模型比较简单，无法进行多层的观测，这一部分打算用框架完成

### Build A CNN Net in TensorFlow





