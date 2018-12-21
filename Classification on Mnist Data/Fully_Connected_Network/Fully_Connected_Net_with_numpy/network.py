import numpy as np
import gzip
import _pickle as cPickle
from sigmoid import sigmoid
from sigmoid_prime import sigmoid_prime
# from Test_for_Crossentrophy.Soft_Max import soft_max
from matplotlib import pyplot
import math
import json


class SoftMax(object):
    @staticmethod
    def result(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)))

    @staticmethod
    def delta(z, a, y):
        # 针对于SoftMax Function
        result = []
        for m, n in zip(a, y):
            if n == 1:
                result.append(m-1)
            else:
                result.append(m)
        return np.array(result)


class CrossEntrophy(object):

    @staticmethod
    def result(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a-y
    # 针对于SoftMax Function
    # def delta(z, a, y):
    #     result = []
    #     for m, n in zip(a, y):
    #         if n == 1:
    #             result.append(m-1)
    #         else:
    #             result.append(m)
    #     return np.array(result)


class SquareError(object):

    @staticmethod
    def result(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod  # 误差测度
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class Network(object):

    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.size = sizes
        self.weights, self.biases = self.initializing_weights()
        self.cost = CrossEntrophy

    def initializing_weights(self, flag=None):
        if flag:
            self.biases = [np.random.randn(y, 1) for y in self.size[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(self.size[:-1], self.size[1:])]
        else:
            self.biases = [np.random.randn(x, 1) for x in self.size[1:]]
            # 改变 初始化权重的方法
            self.weights = [np.random.randn(y, x)/np.sqrt(x) for (x, y) in zip(self.size[:-1], self.size[1:])]
        return self.weights, self.biases

    @staticmethod
    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    @staticmethod
    def mnist_loader():
        f = gzip.open('E:/Pycharm_Projects/DeepLearning_Test/Reference/data/mnist.pkl.gz', 'rb')  # 创建文件对象
        training_data, validation_data, test_data = cPickle.load(f, encoding='bytes')  # 解压后为List
        training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]  # numpy 中的array也可以切片
        training_output = [Network.vectorized_result(int(y)) for y in training_data[1]]
        training_data = list(zip(training_inputs, training_data[1]))
        training_data_new = list(zip(training_inputs, training_output))  # training_data 重排
        validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
        validation_data_new = list(zip(validation_inputs, validation_data[1]))
        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data_new = list(zip(test_inputs, test_data[1]))
        return training_data_new, validation_data_new, test_data_new, training_data, test_data

    def feedward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
        # for b, w in zip(self.biases[:-1], self.weights[:-1]):  # 输入层无激活函数 隐藏层用sigmoid 输出层用soft_max
        #     a = sigmoid(np.dot(w, a) + b)
        # a = soft_max(np.dot(self.weights[-1], a) + self.biases[-1])
        # return a

    def sgd(self, training_data, epochs, eta,  mini_batch_size, training_data_, test_data_, beta,
            regularization=None, test_data=None, drop_probability=None):
        # stochastic gradient descent 随机梯度下降
        # epochs 设置为迭代次数 mini_batch_size设置为每次迭代使用的样本数 eta为学习率
        if test_data:
            n_test = len(test_data)
            test_data_ = list(zip([np.reshape(x, (784, 1)) for x in test_data_[0]],
                                  [Network.vectorized_result(int(y)) for y in test_data_[1]]))
        n = len(training_data)
        training_accuracy = []
        test_accuracy = []
        training_cost = []
        test_cost = []
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                if regularization:
                    self.update_mini_batch(mini_batch, eta, n, beta, regularization, drop_probability)
                else:
                    self.update_mini_batch(mini_batch, eta, n, beta, drop_probability)
            #  正确率统计
            training_accuracy.append(self.evaluate(training_data_)/n)
            #  cost 计算
            if regularization:
                training_cost.append(self.evaluate_cost(training_data, n, regularization))
            else:
                training_cost.append(self.evaluate_cost(training_data, n))
            if test_data:  # 如果有测试数据
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
                test_accuracy.append(self.evaluate(test_data) / n_test)  # list.append 无返回值
                test_cost.append(self.evaluate_cost(test_data_, n_test, regularization))
            else:
                print("Epoch {0} complete".format(j))
        pyplot.ion()  # 打开交互模式 可同时显示多张图片
        pyplot.figure(1)  # Accuracy Visualization
        pyplot.plot(range(0, epochs, 1), training_accuracy, 'r--', label="training_accuracy")
        pyplot.plot(range(0, epochs, 1), test_accuracy, 'b--', label="test_accuracy")
        pyplot.title("Accuracy_Compare")
        pyplot.xlabel("iteration epochs")
        pyplot.ylabel("Accuracy")
        pyplot.grid(b=True, axis='both')
        pyplot.legend()
        pyplot.figure(2)
        pyplot.plot(range(0, epochs, 1), training_cost, 'r--', label="training_cost")
        pyplot.plot(range(0, epochs, 1), test_cost, 'b--', label="test_cost")
        pyplot.legend()
        pyplot.title("Cost_compare")
        pyplot.xlabel("iteration epochs")
        pyplot.ylabel("Error")
        pyplot.grid(b=True, axis='both')
        pyplot.ioff()
        pyplot.show()

    def update_mini_batch(self, mini_batch, eta, n, beta, drop_probability=None, regularization=None):  # 必选参数 默认参数 可变参数
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Parameter Initialization
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        if drop_probability:
            drop_array = np.random.uniform(0, 1, self.size[1])
            # 数值在drop_probability以下的都被drop成0 否则为1
            drop_array = np.array([math.ceil(x - drop_probability) for x in drop_array]).reshape(30, 1)
        for x, y in mini_batch:
            if drop_probability:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y, drop_probability, drop_array)  # BP算法
            else:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # BP算法
            # Momentum BackProp算法
            nabla_b = [beta*nb + (1-beta)*dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [beta*nw + (1-beta)*dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        if regularization:  # 对w的大小进行约束
            self.weights = [(1 - regularization*eta / n) * w - (eta / len(mini_batch)) * nw  # 参数更新
                            for w, nw in zip(self.weights, nabla_w)]
        else:
            self.weights = [w - (eta / len(mini_batch)) * nw  # 参数更新
                            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]  # update parameter

    def backprop(self, x, y, drop_probability=None, drop_array=None):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed_forward
        activiation = x
        activiations = [x]
        zs = []
        #  隐藏层加入随机drop_out 以drop_out_probability
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activiation) + b
            if drop_probability:
                z = z*drop_array
                zs.append(z)
                # 训练时drop_out层 除以相应的keep_probability
                activiation = sigmoid(z)/(1-drop_probability)
            else:
                zs.append(z)
                activiation = sigmoid(z)
            activiations.append(activiation)  # 存每层网络的结果
        z = np.dot(self.weights[-1], activiations[-1]) + self.biases[-1]
        zs.append(z)
        activiations.append(sigmoid(z))

        # backward pass  delta in sigmoid 反向传播 依然需要注意drop_out层
        delta = self.cost.delta(zs[-1], activiations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activiations[-2].transpose())
        for l in range(2, self.num_layer):
            z = zs[-l]
            sp = sigmoid_prime(z)
            if drop_probability:
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp * drop_array
                nabla_b[-l] = delta/(1-drop_probability)
                nabla_w[-l] = np.dot(delta, activiations[-l-1].transpose())/(1-drop_probability)
            else:
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activiations[-l - 1].transpose())
        return nabla_b, nabla_w
        # for b, w in zip(self.biases[:-1], self.weights[:-1]):
        #     z = np.dot(w, activiation) + b
        #     zs.append(z)
        #     activiation = sigmoid(z)
        #     activiations.append(activiation)  # 存每层网络的结果
        # z = np.dot(self.weights[-1], activiations[-1]) + self.biases[-1]
        # zs.append(z)
        # activiation = soft_max(z)
        # activiations.append(activiation)
        # # backward pass
        # delta = self.cost.delta(zs[-1], activiations[-1], y)
        # nabla_b[-1] = delta
        # nabla_w[-1] = np.dot(delta, activiations[-2].transpose())
        # for l in range(2, self.num_layer):
        #     z = zs[-l]
        #     sp = sigmoid_prime(z)
        #     delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        #     nabla_b[-l] = delta
        #     nabla_w[-l] = np.dot(delta, activiations[-l-1].transpose())
        # return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def evaluate_cost(self, test_data, n, regularization=None):
        if regularization:
            return sum(self.cost.result(self.feedward(x), y) for x, y in test_data)/len(test_data) + \
                regularization / (2*n) * sum(np.linalg.norm(w)**2 for w in self.weights)
        else:
            return sum(self.cost.result(self.feedward(x), y) for x, y in test_data)/len(test_data)

    def save_net(self, filename):  # 保存Network的对象
        # 在json字符串中只有基本的python的数据类型的对应
        weights = [[list(x) for x in net.weights[page]] for page in range(len(net.weights))]
        biases = [[list(x) for x in self.biases[page]] for page in range(len(self.biases))]
        data = {"sizes": self.size,
                "weights": weights,
                "biases": biases
                }
        with open(filename, 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    net = Network([784, 30, 10])
    Training_data, Validation_data, Test_data, training_data_origin, test_data_ = net.mnist_loader()
    net.sgd(training_data=Training_data, epochs=30, eta=0.5, mini_batch_size=16, training_data_=
            training_data_origin, test_data_=test_data_, beta=0, test_data=Test_data
            , drop_probability=0)  # 参数设置
    # net.save_net("Mnist_Data")
