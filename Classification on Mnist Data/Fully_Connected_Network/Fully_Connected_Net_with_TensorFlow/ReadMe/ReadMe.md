<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Mnist Test In TensorFlow](#mnist-test-in-tensorflow)
	- [With Zero Hidden Layer](#with-zero-hidden-layer)
		- [eta和mini_batch_size对结果的影响](#eta和minibatchsize对结果的影响)
			- [eta = 0.01 mini_batch_size = 2^4=16](#eta-001-minibatchsize-2416)
			- [eta =0.1 mini_batch_size = 2^4 = 16](#eta-01-minibatchsize-24-16)
		- [分析nan值出现的原因及解决办法](#分析nan值出现的原因及解决办法)
		- [mini_batch_size对train的影响 mini_batch_size = 2^6 = 1024](#minibatchsize对train的影响-minibatchsize-26-1024)
		- [代价函数的设置需要考虑激活函数](#代价函数的设置需要考虑激活函数)
	- [With One Hidden Layer](#with-one-hidden-layer)
		- [Hidden Layer with Sigmoid, Output Layer with Softmax](#hidden-layer-with-sigmoid-output-layer-with-softmax)
			- [eta = 0.01 mini_batch_size =16](#eta-001-minibatchsize-16)
			- [eta = 0.1 with mini_batch_size = 16](#eta-01-with-minibatchsize-16)
			- [eta= 1 with mini_batch_size = 16](#eta-1-with-minibatchsize-16)
		- [Hidden Layer with relu, Output layer with Softmax](#hidden-layer-with-relu-output-layer-with-softmax)
			- [eta = 0.01 with mini_batch_size = 16](#eta-001-with-minibatchsize-16)
			- [eta = 0.1 with mini_batch_size = 16](#eta-01-with-minibatchsize-16)
		- [Hidden Layer with no activiation function, Output Layer with Softmax](#hidden-layer-with-no-activiation-function-output-layer-with-softmax)
			- [eta = 0.1 mini_batch_size = 16](#eta-01-minibatchsize-16)
			- [eta =0.01 mini_batch_size = 16](#eta-001-minibatchsize-16)
	- [Source Code](#source-code)

<!-- /TOC -->

# Mnist Test In TensorFlow

## With Zero Hidden Layer

### eta和mini_batch_size对结果的影响

#### eta = 0.01 mini_batch_size = 2^4=16
* 代价函数使用多元熵：Cost不作**平均**而体现为Minibatch的内的**样本误差之和**;输出层<br/>不使用激活函数：网络的表达能力差，Test Accuracy和Train Accuracy都很低，训练<br/>中基本保持不变，**通过减小学习率也无法改善nan值的情况**，以下不对此结构进行再<br/>讨论。
* 代价函数仍然使用多元熵：输出层用**Softmax**函数归一化到[0,1]之间，Test Accuracy<br/>收敛到92%

#### eta =0.1 mini_batch_size = 2^4 = 16
* 代价函数仍使用多元熵：输出层使用**Softmax**函数归一化，Test Accuracy维持不变<br/>，Cost,weight,biases全都为**nan**值；

### 分析nan值出现的原因及解决办法
* 注意到这里采用的代价函数为多元熵的形式为y=-y_*log(y)的形式，和之前的二元交叉熵<br/>不一样。那么更具交叉熵的形式，可以看出很容易出现log(0)形成的下溢。
解决思路有：
  * 作截断，当y_pred非常小时，截断为epsilon
  * 考虑TensorFlow中自带的优化器
* 如果学习率设置的过大，可能会导致不收敛，也易出现nan值，需要减小学习率
* 由于网络的的权重和偏置随机初始化过大，也有可能会出现nan值
* 针对于输入数据，数据的值过大也有可能会出现nan值，需要对数据作normlization
* 对于深层的神经网络中出现的**梯度爆炸**和**梯度消失**问题，需要对它出现的原因进行分析<br/>，从而给出相应的解决方案

### mini_batch_size对train的影响 mini_batch_size = 2^6 = 1024
* 当mini_batch_size设置为1024时，学习率保持不变,仍然设置为0.01时。经过一个<br/>epoch迭代后就出现了nan值。对应于上述的学习率问题，需要将学习率减小，<br/>当学习率设置为0.001时，恢复正常（**代价为mini_batch的总体代价**）

* 学习率的设置受到多因素的影响，mini_batch_size对其影响的机理在于：mini_batch_size的<br/>选择说明单个mini_batch采用的样本数，mini_batch_size设置的越大，梯度的更新所使用的<br/>样本数就越多，很容易就引起不收敛，当采用的mini_batch的代价不平均时；如果改用平均<br/>代价，不收敛的问题也能得到改善；

测试：当采用的mini_batch的平均代价，当学习率设置为0.1，mini_batch设置为16时，仍然能有效<br/>地收敛

### 代价函数的设置需要考虑激活函数
* 在这个问题中，with no hidden layer，当最后一层的激活函数选择为sigmoid型时，实际对应<br/>着一个输入可以对应多类别的情况；而当最后一层的激活函数选择为softmax型时，实际<br/>对应为单个输入只能对应于单个的类别，多元熵虽然是由二元熵推广而来，不过在实际的<br/>应用中还是需要有所区别，否则无法达到较好的效果

## With One Hidden Layer

### Hidden Layer with Sigmoid, Output Layer with Softmax

#### eta = 0.01 mini_batch_size =16
* 一个epoch后，accuracy就已经收敛到93%，几个epochs后，Test Accuracy收敛到**96%**
，这里<br/>代价函数同样采用的是mini_batch内的误差之和；需要注意的是，在迭代中，无论是cost还是<br/>paras都未出现nan值；

#### eta = 0.1 with mini_batch_size = 16
* 同样为出现nan值，Accuracy收敛到96%

#### eta= 1 with mini_batch_size = 16
* 未出现nan值，但是cost来回振荡，不收敛，学习率设置的过大

### Hidden Layer with relu, Output layer with Softmax

#### eta = 0.01 with mini_batch_size = 16
 * Accuracy收敛到96%后出现nan值，设置**epsilon = 1e-8**对log(y)截断，避免产生下溢，测试发现<br/>迭代同样的次数,nan值消失

#### eta = 0.1 with mini_batch_size = 16
* cost表现为振荡 学习率设置不当

### Hidden Layer with no activiation function, Output Layer with Softmax

#### eta = 0.1 mini_batch_size = 16
* cost振荡, Accuracy收敛到10%

#### eta =0.01 mini_batch_size = 16
* Accuarcy收敛到92% 效果没有隐藏层也加入activation function好

## Source Code
				import os
				import tensorflow as tf
				from tensorflow.python.tools import inspect_checkpoint as chkp  # 可查看checkpoints中的Tensor
				import numpy as np
				from tensorflow.python import debug as tf_debug  # 导入调试工具
				from Exp3.mnist_loader import load_data_wrapper
				from matplotlib import pyplot as plt
				import Exp3.settings as settings


				os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略CPU编译警告 只显示warning和error
				training_data, validation_data, test_data = load_data_wrapper()

				# 生成数据
				with tf.name_scope('Input_data'):
				with tf.name_scope('training_data'):
					# 生成test data
					test_images = []
					test_labels = []
					for i in range(len(test_data)):
							test_images.append(test_data[i][0])
							test_labels.append(test_data[i][1])
					test_images = np.array(test_images).reshape((len(test_data), 784))
					test_labels = np.array(test_labels).reshape((len(test_data), 10))
				with tf.name_scope('test_data'):
					# 生成training data
					train_images = []
					train_labels = []
					for i in range(len(training_data)):
							train_images.append(training_data[i][0])
							train_labels.append(training_data[i][1])
					train_images = np.array(train_images).reshape((len(training_data), 784))
					train_labels = np.array(train_labels).reshape((len(training_data), 10))

				# 输入数据
				with tf.name_scope('inputs'):
				x = tf.placeholder("float32", shape=[None, 784], name='x_input')  # 构建占位符 在session中需要被赋值
				y_ = tf.placeholder("float32", shape=[None, 10], name='y_input')


				# w_1 = initial_weights(shape=[784, 30])
				# b_1 = initial_biases(shape=[30])
				#
				#
				# w_2 = initial_weights(shape=[30, 10])
				# b_2 = initial_biases(shape=[10])

				# layer定义
				def add_layer(layer_input, input_size, output_size, n, activiation=None):
				with tf.name_scope('hidden_layer_%d' % n):
					with tf.name_scope('weights'):
							w = tf.Variable(tf.truncated_normal(shape=[input_size, output_size], dtype=tf.float32) /
															tf.sqrt(float(input_size)), name='random_weights')
					tf.summary.histogram('hidden_layer_%d' % n + '/weights', w)
					with tf.name_scope('biases'):
							b = tf.Variable(tf.truncated_normal(shape=[output_size], dtype=tf.float32), name='biases')
					tf.summary.histogram('hidden_layer_%d' % n + '/biases', b)
					with tf.name_scope('Wx_Plus_b'):
									wx_plus_b = tf.matmul(layer_input, w, name='Multiply') + b
					if activiation:
							output = activiation(wx_plus_b)
					else:
							output = wx_plus_b
					tf.summary.histogram('hidden_layer_%d' % n + '/output', output)
				return output


				#  搭建网络结构
				temp = add_layer(layer_input=x, input_size=784, output_size=30, n=1, activiation=None)
				y = add_layer(layer_input=temp, input_size=30, output_size=10, n=2, activiation=tf.nn.softmax)

				#  loss function
				epsilon = 1e-8  # log函数下溢的截断
				with tf.name_scope('loss'):
				cross_entrophy = - tf.reduce_sum(y_*tf.log(y + epsilon))  # 多元熵->softmax
				tf.summary.scalar('loss', cross_entrophy)  # scalar
				# cross_entrophy = - tf.reduce_sum(y_*tf.log(y + epsilon) + (1.0 - y_)*tf.log(1.0 - y + epsilon))
				# 二元熵->sigmoid

				#  Gradient Descent
				with tf.name_scope('train_scope') as scope:
				train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, name='Gradient_Optimizer').\
					minimize(cross_entrophy, name='minimize_loss')

				init = tf.global_variables_initializer()
				merged = tf.summary.merge_all()

				# 启动会话
				with tf.Session() as sess:
				sess.run(init)
				#  tf.summary和写入计算图需要在session内进行
				writer = tf.summary.FileWriter("log", tf.get_default_graph())
				#  二元熵->sigmoid
				# 训练Model 学习率 eta = 0.01
				# 学习率需要根据训练样本数 采用的激活函数的形式
				# 设置mini_batch_size
				mini_batch_size = 16
				epochs = 20
				training_cost = []
				test_cost = []
				plt.figure(1)
				plt.ion()
				plt.show()
				plt.xlim([0, epochs])
				plt.ylim([0, 1])
				plt.grid(b=True, axis='both')
				for i in range(epochs):
					np.random.shuffle(training_data)
					Batch = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
					for mini_batch in Batch:
							Input = []
							Label = []
							for j in range(len(mini_batch)):
									Input.append(mini_batch[j][0])
									Label.append(mini_batch[j][1])
							Input = np.array(Input).reshape((len(mini_batch), 784))
							Label = np.array(Label).reshape((len(mini_batch), 10))
							# 利用feed_dict给占位符赋值
							train_step.run(feed_dict={x: Input, y_: Label})
					# 在每个epoch内进行准确率评估
					training_cost.append(cross_entrophy.eval(feed_dict={x: train_images, y_: train_labels}))
					test_cost.append(cross_entrophy.eval(feed_dict={x: test_images, y_: test_labels}))
					correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
					# 准确率评估
					accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
					# training accuracy
					print("epoch: %d Finished" % i)
					training_accuracy = accuracy.eval(feed_dict={x: train_images, y_: train_labels})
					print("Training_Accuracy: %.2f" % training_accuracy)
					# test accuracy
					test_accuracy = accuracy.eval(feed_dict={x: test_images, y_: test_labels})
					print("Test Accuracy: %.2f" % test_accuracy)
					plt.scatter(i, training_accuracy, marker='*', c='red')
					plt.scatter(i, test_accuracy, marker='+', c='blue')
					# 监控loss在每个epoch内的变化
					result = sess.run(merged, feed_dict={x: train_images, y_: train_labels})
					writer.add_summary(result, i)
