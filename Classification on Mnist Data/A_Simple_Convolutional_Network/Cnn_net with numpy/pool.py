import numpy as np
from sigmoid import sigmoid
from sigmoid_prime import sigmoid_prime


class Pooling(object):
    def __init__(self, layer_index, input_z, input_a, stride, pool_size, activation, activation_prime, method):
        self.layer_index = layer_index  # layer index
        self.input_z = input_z  # 前层的加权输入
        self.input_a = input_a  # 前层的激活输出
        self.stride = stride  # pool的滑动步长
        self.pool_size = pool_size  # 默认 pool_size在高度和宽度方向是相同的
        self.activation = activation  # 输入网络层的activation列表
        self.activation_prime = activation_prime
        self.method = method
        self.checkout()

    def checkout(self):
        # 输入的有效性检测
        try:
            assert isinstance(self.input_a, np.ndarray)
            assert isinstance(self.stride, int) and isinstance(self.pool_size, int)
            assert self.method in ['max_pooling', 'average_pooling']
        except Exception as e:
            print('Exception_info: the input TypeError\n')
            raise e

        try:
            # pool_size最多只能和输入的长/宽相等，同时pool的stride应该小于窗口大小，以免特征的丢失
            # 最好还是选取合适的pool_size和stride 保证特征不丢失
            assert (self.input_a.shape[0] >= self.pool_size >= self.stride and self.pool_size <= self.input_a.shape[1])
        except Exception as e:
            print('Exception_info: the shape of the kernel must be smaller than the input\n')
            raise e

    def feedward(self):
        # print('池化层前向传播\n')
        # 池化层的前向传播
        # 池化后的索引
        # 池化层只需要进行误差的传递
        height = int((self.input_z.shape[1] - self.pool_size) / self.stride + 1)
        width = int((self.input_z.shape[0] - self.pool_size) / self.stride + 1)
        channels = self.input_z.shape[2]
        if self.method == 'max_pooling':
            # 对于最大池化而言，需要记录最大值出现的位置
            max_value_index = []
            output = np.zeros(shape=[height, width, channels])
            for i in range(0, height):
                for j in range(0, width):
                    # 找出每一个通道上的 处于当前的pool中的最大值
                    max_value = np.max(self.input_a[i*self.stride:i*self.stride + self.pool_size,
                                       j*self.stride:j*self.stride + self.pool_size, :], axis=(0, 1))
                    output[i, j, :] = max_value
                    # 不考虑单个的pool中的最大值重合的情况，如果出现了两个相同的最大值
                    index_list = []
                    for k in range(0, channels):
                        index_tuple = np.where(self.input_a[i*self.stride:i*self.stride + self.pool_size,
                                       j*self.stride:j*self.stride + self.pool_size, :] == max_value[k])
                        if len(index_tuple[0]) != 1:
                            index_tuple = (index_tuple[0][0], index_tuple[1][0], index_tuple[2][0])
                        index_list.append(index_tuple)
                    max_value_index.append(index_list)
            if self.activation[self.layer_index] is not None:
                output_a = self.activation[self.layer_index](output)
            else:
                output_a = output
            # print('池化层前向传播结束\n')
            return output, output_a, max_value_index
        else:
            # self.method == 'average_pooling'
            output = np.zeros(shape=[height, width, channels])
            for i in range(0, height):
                for j in range(0, width):
                    average_value = np.average(self.input_a[i*self.stride:i*self.stride + self.pool_size,
                                       j*self.stride:j*self.stride + self.pool_size, :], axis=(0, 1))
                    output[i, j, :] = average_value
            if self.activation[self.layer_index] is not None:
                output_a = self.activation[self.layer_index](output)
            else:
                output_a = output
            # print('池化层前向传播结束\n')
            return output, output_a

    def backprop(self, next_layer_metric, max_value_index=None):
        # print('池化层反向传播开始\n')
        # 检查next_layer_metrc的类型
        try:
            assert len(next_layer_metric.shape) == 3
        except Exception as e:
            print("Exception info：the next_layer_metric needed to be reshape.\n")
            raise e
        # next_layer_metric的维度需要在main函数中改动
        # 上层的metric
        metric = np.zeros_like(self.input_a)
        # height = (next_layer_metric.shape[0] - 1) * self.stride + self.pool_size
        # width = (next_layer_metric.shape[1] - 1) * self.stride + self.pool_size
        if self.method == 'max_pooling':
            for i in range(0, next_layer_metric.shape[0]):
                for j in range(0, next_layer_metric.shape[1]):
                    # 需要考虑相邻的pool之间最大值的index重复的情形
                    metric[i*self.stride+np.array(max_value_index[i*next_layer_metric.shape[1]+j][0][:]),
                    j*self.stride+np.array(max_value_index[i*next_layer_metric.shape[1]+j][1][:]), :] \
                        = next_layer_metric[i, j, :] + \
                          metric[i*self.stride+np.array(max_value_index[i*next_layer_metric.shape[1]+j][0][:]),
                                j*self.stride+np.array(max_value_index[i*next_layer_metric.shape[1]+j][1][:]), :]
        else:
            for i in range(0, next_layer_metric.shape[0]):
                for j in range(0, next_layer_metric.shape[1]):
                    # 同样需要考虑重叠池化的影响
                    metric[i*self.stride:i*self.stride + self.pool_size, j*self.stride:j*self.stride + self.pool_size, :] \
                        = next_layer_metric[i, j, :] + metric[i*self.stride:i*self.stride + self.pool_size,
                         j*self.stride: j*self.stride + self.pool_size, :]

        if self.layer_index == 1:
            pass
        else:
            if self.activation[self.layer_index - 1] is not None:
                metric = metric * self.activation_prime[self.layer_index - 1](self.input_z)
        # print('池化层反向传播结束\n')
        return metric
