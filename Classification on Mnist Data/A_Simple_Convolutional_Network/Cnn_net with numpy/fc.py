import numpy as np
import datetime
from sigmoid import sigmoid
from sigmoid_prime import sigmoid_prime


class FcLayer(object):
    def __init__(self, layer_index, input_z, input_a, activation, activation_prime, output_size=10):
        self.layer_index = layer_index
        self.input_z = input_z
        self.input_a = input_a
        self.output_size = output_size
        self.activation = activation  # activation函数列表
        self.activation_prime = activation_prime
        self.checkout()  # 输入的有效性检查

    def checkout(self):
        # 输入的有效性检测
        try:
            assert isinstance(self.input_a, np.ndarray)
            assert isinstance(self.input_z, np.ndarray)
            assert isinstance(self.layer_index, int)
        except Exception as e:
            print('Exception_info: the input TypeError\n')
            raise e

    def feedward(self, weights, bias):
        # print('全连接前向传播开始\n')
        output = np.zeros(shape=[self.output_size, 1])
        try:
            assert weights.shape[0: -1] == self.input_a.shape
            assert weights.shape[3] == self.output_size
        except Exception as e:
            print('Exception info: Matrix index not matched\n')
            raise e
        for k in range(0, weights.shape[3]):
            weights_reshape = FcLayer.img_col(input_=weights[:, :, :, k], stride=weights.shape[0], kernel_size=
                                                [weights.shape[0], weights.shape[1]])
            temp_input_reshape = FcLayer.img_col(input_=self.input_a, stride=weights.shape[0], kernel_size=
                                                 [weights.shape[0], weights.shape[1]])
            output[k, 0] = np.dot(weights_reshape, temp_input_reshape.T)
        if self.activation[self.layer_index] is not None:
            output_a = self.activation[self.layer_index](output)
        else:
            output_a = output
        # print('全连接前向传播结束\n')
        return output, output_a

    def backprop(self, weights, bias, next_layer_metric):
        # print('全连接反向传播开始\n')
        weights_reshape = np.zeros(shape=[self.output_size, weights.shape[0] * weights.shape[1] * weights.shape[2]])
        # for k in range(0, weights.shape[3]):
        #     for i in range(0, weights.shape[2]):
        #         weights_reshape[i*weights.shape[0]*weights.shape[1]:(i+1)*weights.shape[0]*weights.shape[1]
        #                         , k] = np.ndarray.flatten(weights[:, :, i, k], order='C')
        for j in range(0, weights.shape[3]):
            weights_reshape[j, :] = FcLayer.img_col(input_=weights[:, :, :, j], stride=weights.shape[0],
                                                    kernel_size=[weights.shape[0], weights.shape[1]])
        metric = np.dot(weights_reshape.transpose(), next_layer_metric)

        if self.layer_index == 1:
            pass
        else:
            if self.activation[self.layer_index - 1] is not None:
                metric = np.dot(metric, self.activation_prime[self.layer_index - 1](self.input_z))

        metric_ = np.zeros(shape=self.input_z.shape)
        # for k in range(0, self.input_z.shape[2]):
        #     metric_[:, :, k] = np.reshape(metric[k*self.input_z.shape[0] *
        #                                  self.input_z.shape[1]:(k+1)*self.input_z.shape[0]*self.input_z.shape[1], 0],
        #                                  newshape=[self.input_z.shape[0], self.input_z.shape[1]], order='C')
        metric_ = np.reshape(metric, newshape=[self.input_a.shape[2], self.input_a.shape[0], self.input_a.shape[1]])
        metric_ = metric_.transpose((1, 2, 0))

        # 本层的 weights 和bias的参数更新
        # for k in range(0, self.input_z.shape[2]):
        #     input_a_reshape[k*self.input_z.shape[0]*self.input_z.shape[1]:
        #                     (k+1)*self.input_z.shape[0]*self.input_z.shape[1], 0] = \
        #                     np.ndarray.flatten(self.input_a[:, :, k], order='C')
        input_a_reshape = self.input_a.transpose((2, 0, 1)).flatten(order='C')
        metric_weights = np.reshape(np.dot(next_layer_metric, input_a_reshape.reshape(1, self.input_a.size)),
                                    newshape=(weights.shape[3], weights.shape[2], weights.shape[0], weights.shape[1]),
                                    order='C')
        metric_weights = metric_weights.transpose((2, 3, 1, 0))
        metric_bias = next_layer_metric
        # print('全连接反向传播结束\n')
        return metric_, metric_weights, metric_bias

    @ staticmethod
    def img_col(input_, kernel_size, stride):
        # 用img_col优化卷积运算的时间复杂度 (不局限于stride)
        # 参照全连接BP神经网络的前向传播和后向传播的方法
        # 默认水平的滑动窗口和垂直的滑动窗口具有相同的size和stride
        # 传入的input如果有intepolation/padding的操作 需要提前完成
        height = int((input_.shape[0] - kernel_size[0])/stride + 1)
        width = int((input_.shape[1] - kernel_size[1])/stride + 1)
        if len(input_.shape) == 2:
            channel = 1
            input_ = input_.reshape([input_.shape[0], input_.shape[1], 1])
        else:
            channel = input_.shape[2]
        output = np.zeros(shape=[height * width, channel * kernel_size[0] * kernel_size[1]])
        input_ = input_.transpose((2, 0, 1))  # 作transpose以便后续可以按reshape进行重构
        for j in range(0, height * width):
            # 输入的通道数索引
                # 重新排列后 行的索引
                output[j, :] = \
                    np.ndarray.flatten(input_[:, int(j / width)*stride:int(j / width)*stride+kernel_size[0],
                                             (j % width)*stride:(j % width)*stride+kernel_size[1]], order='C')
        return output
