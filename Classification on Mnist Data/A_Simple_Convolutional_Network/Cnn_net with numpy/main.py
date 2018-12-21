from multiprocessing import Pool
import os
import numpy as np
import gzip
import pickle as cPickle
import matplotlib.pyplot as plt
import datetime
from conv2D import ConvLayer
from pool import Pooling
from fc import FcLayer
from sigmoid import sigmoid
from sigmoid_prime import sigmoid_prime
import settings


class CrossEntrophy(object):
    @ staticmethod
    def result(a, y):
        return np.nan_to_num(np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @ staticmethod
    def delta(a, y):
        # 对应于激活函数为sigmoid的误差测度
        return a - y


class CnnNet(object):
    # 搭建简单的CNN网络
    def __init__(self, conv_kernel, conv_bias, weights, bias):
        self.activations = [None, sigmoid, None, sigmoid]
        self.activations_prime = [None, sigmoid_prime, None, sigmoid_prime]
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.weights = weights
        self.bias = bias
        self.cost = CrossEntrophy

    def feedward(self, a):
        # 前向传播
        # 卷积核的设置 7*7*1*30
        # 卷积核的参数初始化
        # start_time_1 = datetime.datetime.now()
        self.conv_layer_1 = ConvLayer(layer_index=1, input_z=a, input_a=a, kernel=self.conv_kernel, bias=self.conv_bias,
                                      stride=3, padding='VALID', activation=self.activations,
                                      activation_prime=self.activations_prime)
        output1_z, output1_a = self.conv_layer_1.feedward()
        # start_time_2 = datetime.datetime.now()
        self.pool_layer_2 = Pooling(layer_index=2, input_z=output1_z, input_a=output1_a, stride=2, pool_size=2,
                                    activation=self.activations, activation_prime=self.activations_prime,
                                    method='max_pooling')
        # start_time_3 = datetime.datetime.now()
        output2_z, output2_a, max_value_index = self.pool_layer_2.feedward()
        # start_time_4 = datetime.datetime.now()
        self.fc_layer_3 = FcLayer(layer_index=3, input_z=output2_z, input_a=output2_a, activation=self.activations,
                                  activation_prime=self.activations_prime, output_size=10)
        output3_z, output3_a = self.fc_layer_3.feedward(weights=self.weights, bias=self.bias)
        # start_time_5 = datetime.datetime.now()
        return max_value_index, output3_z, output3_a

    @ staticmethod
    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    @staticmethod
    def mnist_loader():
        f = gzip.open('%s' % settings.Mnist_Path, 'rb')  # 创建文件对象
        training_data, validation_data, test_data = cPickle.load(f, encoding='bytes')  # 解压后为List
        training_inputs = [np.reshape(x, (28, 28)) for x in training_data[0]]  # numpy 中的array也可以切片
        training_output = [CnnNet.vectorized_result(int(y)) for y in training_data[1]]
        training_data = list(zip(training_inputs, training_data[1]))
        training_data_new = list(zip(training_inputs, training_output))  # training_data 重排
        test_inputs = [np.reshape(x, (28, 28)) for x in test_data[0]]
        test_output = [CnnNet.vectorized_result(int(y)) for y in test_data[1]]
        test_data_new = list(zip(test_inputs, test_output))
        test_data = list(zip(test_inputs, test_data[1]))
        return training_data_new, test_data_new, training_data, test_data

    def sgd(self, training_data, test_data, training_data_, test_data_, epochs, eta, mini_batch_size):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        training_accuracy = []
        test_accuracy = []
        training_cost = []
        test_cost = []
        for j in range(epochs):
            print('正在进行第%d个epoch：\n' % j)
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            index = 0
            # start_time_1 = datetime.datetime.now()
            for mini_batch in mini_batches:
                print('正在update第%d个mini_batch\n' % index)
                self.update_mini_batch(mini_batch, eta, n)
                print('update第%d个mini_batch over\n' % index)
                index = index + 1
            # start_time_2 = datetime.datetime.now()
            #  正确率统计
            training_accuracy.append(self.evaluate(training_data_)/n)
            #  cost 计算
            training_cost.append(self.evaluate_cost(training_data)/n)
            if test_data:  # 如果有测试数据
                temp = self.evaluate(test_data_)
                print("Epoch {0}: {1} / {2}".format(
                    j, temp, n_test))
                test_accuracy.append(temp / n_test)  # list.append 无返回值
                test_cost.append(self.evaluate_cost(test_data)/n_test)
            else:
                print("Epoch {0} complete".format(j))
        plt.ion()  # 打开交互模式 可同时显示多张图片
        plt.figure(1)  # Accuracy Visualization
        plt.plot(range(0, epochs, 1), training_accuracy, 'r--', label="training_accuracy")
        plt.plot(range(0, epochs, 1), test_accuracy, 'b--', label="test_accuracy")
        plt.title("Accuracy_Compare")
        plt.xlabel("iteration epochs")
        plt.ylabel("Accuracy")
        plt.grid(b=True, axis='both')
        plt.legend()
        plt.figure(2)
        plt.plot(range(0, epochs, 1), training_cost, 'r--', label="training_cost")
        plt.plot(range(0, epochs, 1), test_cost, 'b--', label="test_cost")
        plt.legend()
        plt.title("Cost_compare")
        plt.xlabel("iteration epochs")
        plt.ylabel("Error")
        plt.grid(b=True, axis='both')
        plt.ioff()
        plt.show()

    def update_mini_batch(self, mini_batch, eta, n):  # 必选参数 默认参数 可变参数
        nabla_conv_weights = np.zeros_like(self.conv_kernel)
        nabla_conv_bias = np.zeros_like(self.conv_bias)
        nabla_weights = np.zeros_like(self.weights)
        nabla_bias = np.zeros_like(self.bias)
        for x, y in mini_batch:
            # start_time_1 = datetime.datetime.now()
            # print('正在进行样本的迭代\n')
            delta_weights, delta_bias, delta_conv_weights, delta_conv_bias = self.backprop(x, y)  # BP算法
            # Momentum BackProp算法
            nabla_conv_bias = nabla_conv_bias + delta_conv_bias
            nabla_conv_weights = nabla_conv_weights + delta_conv_weights
            nabla_weights = nabla_weights + delta_weights
            nabla_bias = nabla_bias + delta_bias
            # start_time_2 = datetime.datetime.now()
            pass
        #  paras update
        self.weights = self.weights - (eta / len(mini_batch)) * nabla_weights
        self.bias = self.bias - (eta/len(mini_batch)) * nabla_bias
        self.conv_kernel = self.conv_kernel - (eta/len(mini_batch)) * nabla_conv_weights
        self.conv_bias = self.conv_bias - (eta/len(mini_batch)) * nabla_conv_bias

    def backprop(self, x, y):
        # 先进行前向传播
        # start_time_1 =datetime.datetime.now()
        max_value_index, output3_z, output3_a = self.feedward(x)
        # start_time_2 = datetime.datetime.now()
        # 误差反向传播
        # 计算最后一层的误差
        # start_time_1 = datetime.datetime.now()
        delta_1 = self.cost.delta(output3_a, y)
        # start_time_3 = datetime.datetime.now()
        delta_2, delta_weights, delta_bias = self.fc_layer_3.backprop(weights=self.weights, bias=self.bias,
                                                                      next_layer_metric=delta_1)
        # start_time_4 = datetime.datetime.now()
        delta_3 = self.pool_layer_2.backprop(next_layer_metric=delta_2, max_value_index=max_value_index)
        # start_time_5 = datetime.datetime.now()
        delta_4, delta_conv_weights, delta_conv_bias = self.conv_layer_1.backprop(next_layer_metric=delta_3)
        # start_time_6 = datetime.datetime.now()
        return delta_weights, delta_bias, delta_conv_weights, delta_conv_bias

    def evaluate(self, test_data):
        temp = [(np.argmax(self.feedward(x)[2]), y) for x, y in test_data]
        return sum(int(x == y) for x, y in temp)

    def evaluate_cost(self, test_data):
        return sum(self.cost.result(self.feedward(x)[2], y) for x, y in test_data)


if __name__ == "__main__":
    # 运算效率太低  难以检测BP算法的推导的正确性
    Training_Data, Test_Data, Training_Data_, Test_Data_ = CnnNet.mnist_loader()
    # Paras Initializing
    Conv_Kernel = np.random.randn(7, 7, 1, 30)/np.sqrt(7 * 7)
    Conv_Bias = np.random.randn(1, 30)
    Weights = np.random.randn(4, 4, 30, 10)/np.sqrt(4 * 4)
    Bias =np.random.randn(10, 1)
    cnn_net = CnnNet(conv_kernel=Conv_Kernel, conv_bias=Conv_Bias, weights=Weights, bias=Bias)
    cnn_net.sgd(training_data=Training_Data[0:1000], test_data=Test_Data[0:1000], training_data_=Training_Data_[0:1000], test_data_=Test_Data_[0:1000],
               epochs=5, eta=0.5, mini_batch_size=32)
