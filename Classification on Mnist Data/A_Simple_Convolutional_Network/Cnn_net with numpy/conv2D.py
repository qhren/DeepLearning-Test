import numpy as np
import datetime
from sigmoid_prime import sigmoid_prime


class ConvLayer(object):
    def __init__(self, layer_index, input_z, input_a, kernel, bias, stride, padding, activation, activation_prime):
        # input_的channel必须和kernel的channel保持一致
        # 假设width和height上的stride是相同的
        self.layer_index = layer_index
        self.input_z = input_z
        self.input_a = input_a  # shape = [height, width, channels]
        self.kernel = kernel   # shape = [height, width, channels , filter_index]
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.activation_prime = activation_prime
        #  Exception check
        self.check_out()

    def check_out(self):
        try:
            assert isinstance(self.input_a, np.ndarray) and isinstance(self.kernel, np.ndarray) and \
                   isinstance(self.bias, np.ndarray)
            assert isinstance(self.stride, int)
            assert self.padding in ['SAME', 'VALID']
            assert len(self.bias.shape) == 2 and self.bias.shape[0] == 1 and self.bias.shape[1] == self.kernel.shape[3]
        except Exception as e:
            print('Exception_info: the input TypeError\n')
            raise e
        try:
            # 输入的通道数和Kernel的通道数是否相等
            assert self.input_a.shape[2] == self.kernel.shape[2]
        except Exception:
            # print('Exception_info: the channels not equal\n')
            self.input_z = np.reshape(self.input_z, newshape=[self.input_z.shape[0],
                                                              self.input_z.shape[1], 1])
            self.input_a = np.reshape(self.input_a, newshape=[self.input_a.shape[0],
                                                              self.input_a.shape[1], 1])
        try:
            # 理论上来说 卷积核的窗口宽度应该小于input_的窗口宽度
            # ConvLayer所实现的二维的卷积其实是二者的互相关 不能交换次序
            # 当Kernel的窗口大小等于input的大小时：等效于全连接网络
            assert (self.kernel.shape[0] <= self.input_a.shape[0] and self.kernel.shape[1] <= self.input_a.shape[1])
        except Exception as e:
            print('Exception_info: the shape of the kernel must be smaller than the input\n')
            raise e

        try:
            # 为了解决padding=same时的两边补0的对称性问题，同时要求锚点处于窗口的中心
            # 最好要求Kernel的长和宽都为奇数个像素点
            assert (self.kernel.shape[0] % 2 == 1 and self.kernel.shape[1] % 2 == 1)
        except Exception as e:
            print('Exception_info: the kernel size should be odd\n')
            raise e

    def feedward(self):
        # print('卷积层前向传播\n')
        # 计算输出的size
        # padding = 'VALID'的情形
        if self.padding == 'VALID':
            height = int((self.input_a.shape[0] - self.kernel.shape[0]) / self.stride + 1)
            width = int((self.input_a.shape[1] - self.kernel.shape[1]) / self.stride + 1)
            channels = self.kernel.shape[3]
            output = np.zeros(shape=[height, width, channels])
            # for i in range(0, channels):
            #     for j in range(0, height):
            #         for k in range(0, width):
            #             output[j, k, i] = np.sum(self.input_a[j * self.stride:j * self.stride + self.kernel.shape[0],
            #                                             k*self.stride:k*self.stride+self.kernel.shape[1], :] *
            #                                             self.kernel[:, :, :, k]) + self.bias[0, k]
            input_a_reshape = ConvLayer.img_col(input_=self.input_a, kernel_size=[self.kernel.shape[0]
                                                , self.kernel.shape[1]], stride=self.stride)
            for k in range(0, self.kernel.shape[3]):
                kernel_reshape = ConvLayer.img_col(input_=self.kernel[:, :, :, k], kernel_size=
                                                   [self.kernel.shape[0], self.kernel.shape[1]], stride=self.stride)
                output[:, :, k] = np.dot(kernel_reshape, input_a_reshape.T).reshape([height, width], order='C') \
                                        + self.bias[0, k]
                # broadcast with bias
        else:
            # padding = 'SAME'
            # 经过Convolution后 output的size
            height = self.input_a.shape[0]
            width = self.input_a.shape[1]
            channels = self.kernel.shape[3]
            output = np.zeros(shape=[height, width, channels])
            # 通过output计算对input补0
            zero_padding_width = (width - 1) * self.stride + self.kernel.shape[1]
            zero_padding_height = (height - 1) * self.stride + self.kernel.shape[0]
            temp = np.zeros(shape=[zero_padding_height, zero_padding_width, self.input_a.shape[2]])
            left_padding = int((zero_padding_width - width)/2)
            # right_padding = zero_padding_width - width - left_padding
            up_padding = int((zero_padding_height - height)/2)
            # down_padding = zero_padding_height - height - up_padding
            temp[up_padding:up_padding + height, left_padding:left_padding+width, :] = self.input_a
            # for i in range(0, channels):
            #     for j in range(0, zero_padding_height):
            #         for k in range(0, zero_padding_width):
            #             output[j, k, i] = np.sum(temp[j * self.stride:j * self.stride + self.kernel.shape[0],
            #                                              k*self.stride:k*self.stride+self.kernel.shape[1],
            #                                             :] * self.kernel[:, :, :, k]) + self.bias[0, k]
            input_a_reshape = ConvLayer.img_col(input_=temp, kernel_size=[self.kernel.shape[0],
                                                self.kernel.shape[1]], stride=self.stride)
            for k in range(0, self.kernel.shaep[3]):
                kernel_reshape = ConvLayer.img_col(input_=self.kernel[:, :, :, k], kernel_size=
                                                   [self.kernel.shape[0], self.kernel.shape[1]], stride=self.stride)
                output[:, :, k] = np.dot(kernel_reshape, input_a_reshape.T).reshape([height, width]) + self.bias[0, k]
            # print('卷积层前向传播结束')
        return output, self.activation[self.layer_index](output)

    def backprop(self, next_layer_metric):
        start_time_1 = datetime.datetime.now()
        # print('卷积层反向传播开始\n')
        # 返回上一层的误差测度和本层的参数的梯度更新
        # 后一层的误差测度应该和前一层具有相同的channels数
        # 后一层的误差测度补全 (加上padding)
        # 计算插值和padding之后的维度
        try:
            assert len(next_layer_metric.shape) == 3
        except Exception as e:
            print("Exception info：the next_layer_metric needed to be reshape.\n")
            raise e
        # 检测有效输入
        valid_column = (next_layer_metric.shape[1] - 1) * self.stride + self.kernel.shape[1]
        valid_row = (next_layer_metric.shape[0] - 1) * self.stride + self.kernel.shape[0]
        width = (valid_column - 1) * 1 + self.kernel.shape[1]
        height = (valid_row - 1) * 1 + self.kernel.shape[0]

        # 插值之后的维度
        intepolation_height = (next_layer_metric.shape[0] - 1) * (self.stride - 1) + next_layer_metric.shape[0]
        intepolation_width = (next_layer_metric.shape[1] - 1) * (self.stride - 1) + next_layer_metric.shape[1]
        intepolation_zero = np.zeros(shape=[intepolation_height, intepolation_width, next_layer_metric.shape[2]])
        # Intepolation插值
        for i in range(0, intepolation_width):
            for j in range(0, intepolation_height):
                if i % self.stride == 0 and j % self.stride == 0:
                    intepolation_zero[j, i, :] = next_layer_metric[int(j/self.stride), int(i/self.stride), :]
                else:
                    intepolation_zero[j, i, :] = np.zeros(shape=[next_layer_metric.shape[2]])
        # Padding
        left_padding = int((width - intepolation_width)/2)
        up_padding = int((height - intepolation_height)/2)
        right_padding = width - intepolation_width - left_padding
        down_padding = height - intepolation_height - up_padding
        temp = np.pad(array=intepolation_zero, pad_width=((up_padding, down_padding), (left_padding, right_padding),
                                                          (0, 0)), mode='constant', constant_values=0)
        channels = self.input_z.shape[2]
        # 进行卷积计算 得到前一层的metric metric是否需要添加激活函数 和上一层是否有激活函数相关
        metric = np.zeros(shape=[valid_row, valid_column, self.input_z.shape[2]])
        # 本层的梯度更新
        # 卷积核的梯度
        metric_kernel = np.zeros(shape=self.kernel.shape)
        # 偏置的梯度
        metric_bias = np.zeros(shape=[self.kernel.shape[3]])
        # for k in range(0, channels):
        #     for j in range(0, valid_row):
        #         for i in range(0, valid_column):
        #             metric[j, i, k] = 0
        #             for index in range(0, self.kernel.shape[3]):
        #                 # 取所有滤波器组的第k个channel
        #                 metric[j, i, k] = metric[j, i, k] + np.sum(temp[j:j + self.kernel.shape[0],
        #                                                         i:i + self.kernel.shape[1], k] *
        #                                                     np.rot90(self.kernel[:, :, k, index], k=2, axes=(0, 1)))
        next_layer_metric_reshape = ConvLayer.img_col(input_=temp, stride=1, kernel_size=
                                                    [self.kernel.shape[0], self.kernel.shape[1]])
        for k in range(0, channels):
            kernel_reshape = ConvLayer.img_col(input_=self.kernel[:, :, k, :], stride=1, kernel_size=
                                               [self.kernel.shape[0], self.kernel.shape[1]])
            metric[:, :, k] = np.dot(np.flipud(np.fliplr(kernel_reshape)), next_layer_metric_reshape.T).\
                                     reshape([valid_row, valid_column])
        if self.padding == 'VALID':
            metric_fixed = np.zeros(shape=self.input_z.shape)
            # 取运算时的有效值
            metric_fixed[0:valid_row, 0:valid_column, :] = metric
        else:
            # self.padding == 'SAME'
            metric_fixed = metric[up_padding:up_padding + self.input_z.shape[1], left_padding:left_padding +
                                                                                 self.input_z.shape[0], :]
        # 加上activation function
        if self.layer_index == 1:
            pass
        else:
            if self.activation[self.layer_index - 1] is not None:
                metric_fixed = metric_fixed * self.activation_prime[self.layer_index - 1](self.input_z)
        # 本层的参数更新
        if self.padding == 'VALID':
            temp_input = self.input_a[0:valid_row, 0:valid_column, :]
        else:
            # self.padding == 'SAME':
            padding_height = (self.input_z.shape[0] - 1) * self.stride + self.kernel[0]
            padding_width = (self.input_z.shape[1] - 1) * self.stride + self.kernel[1]
            left_padding = int((padding_width - self.input_z.shape[1]) / 2)
            up_padding = int((height - self.input_z.shape[0]) / 2)
            right_padding = padding_width - left_padding
            down_padding = padding_height - up_padding
            temp_input = np.pad(array=self.input_a, pad_width=((up_padding, down_padding), (left_padding, right_padding)
                                                               , (0, 0)), mode='constant', constant_values=0)
        # for k in range(0, self.kernel.shape[3]):
        #     # 对滤波器组的索引
        #     for index in range(0, self.kernel.shape[2]):
        #         # 对滤波器的通道的索引
        #         for j in range(0, self.kernel.shape[0]):
        #             for i in range(0, self.kernel.shape[1]):
        #                 metric_kernel[j, i, index, k] = np.sum(temp_input[j:j + intepolation_zero.shape[0],
        #                                                         i:i + intepolation_zero.shape[1], index] * \
        #                                                        intepolation_zero[:, :, k])
        #     metric_bias[k] = np.sum(next_layer_metric[:, :, k])
        for k in range(0, self.kernel.shape[3]):
            intepolation_zero_reshape = ConvLayer.img_col(input_=intepolation_zero[:, :, k], stride=1,
                                                          kernel_size=[intepolation_zero.shape[0],
                                                                       intepolation_zero.shape[1]])
            for index in range(0, self.kernel.shape[2]):
                input_a_reshape = ConvLayer.img_col(input_=temp_input[:, :, index], stride=1, kernel_size=
                                                    [intepolation_zero.shape[0], intepolation_zero.shape[1]])
                metric_kernel[:, :, index, k] = np.dot(intepolation_zero_reshape, input_a_reshape.T).reshape(
                                                    [self.kernel.shape[0], self.kernel.shape[1]])
            metric_bias[k] = np.sum(next_layer_metric[:, :, k])
        start_time_2 = datetime.datetime.now()
        # print('卷积层反向传播结束\n')
        return metric_fixed, metric_kernel, metric_bias

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
