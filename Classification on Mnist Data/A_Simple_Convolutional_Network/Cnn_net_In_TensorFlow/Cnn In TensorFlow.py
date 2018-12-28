# -- coding: utf-8 --
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip
import pickle as cPickle
import settings
import numpy as np


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def mnist_loader():
    f = gzip.open('%s' % settings.Mnist_Path, 'rb')  # 创建文件对象
    training_data, validation_data, test_data = cPickle.load(f, encoding='bytes')  # 解压后为List
    training_inputs = [np.reshape(x, (28, 28)) for x in training_data[0]]  # numpy 中的array也可以切片
    training_output = [vectorized_result(int(y)) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_data[1]))
    training_data_new = list(zip(training_inputs, training_output))  # training_data 重排
    test_inputs = [np.reshape(x, (28, 28)) for x in test_data[0]]
    test_output = [vectorized_result(int(y)) for y in test_data[1]]
    test_data_new = list(zip(test_inputs, test_output))
    test_data = list(zip(test_inputs, test_data[1]))
    return training_data_new, test_data_new


# 构建计算图的输入 形式为 [Batch, Height, Width, Channels]
with tf.name_scope('input') as scope_input:
    x_ = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='Input_Image')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10, 1], name='Input_Label')

# 构建计算图的结点
with tf.name_scope('Conv_Layer_1') as conv_scope1:
    Conv_Filter_1 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 1, 30])/tf.sqrt(5.*5.),
                                dtype=tf.float32, name='Conv_Kernel')
    tf.summary.histogram(values=Conv_Filter_1, name='Conv_Kernel')
    input_after_conv1 = tf.nn.relu(tf.nn.conv2d(input=x_, filter=Conv_Filter_1, strides=[1, 3, 3, 1], padding='SAME',
                                                name='Conv'), name='activation')
    tf.summary.histogram(values=input_after_conv1, name='Conv_Output')
    input_after_pool1 = tf.nn.max_pool(value=input_after_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='VALID', name='Pool')
    tf.summary.histogram(values=input_after_pool1, name='Pool_Output')

with tf.name_scope('Conv_Layer_2') as conv_scope2:
    Conv_Filter_2 = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 3, 30, 50])/tf.sqrt(3.*3.*30),
                                dtype=tf.float32, name='Conv_Kernel_2')
    tf.summary.histogram(values=Conv_Filter_2, name='Conv_Kernel')
    input_after_conv2 = tf.nn.relu(tf.nn.conv2d(input=input_after_pool1, filter=Conv_Filter_2, strides=[1, 2, 2, 1],
                                   padding='VALID', name='Conv'), name='activation')
    tf.summary.histogram(values=input_after_conv2, name='Conv_Output')
    input_after_pool2 = tf.nn.max_pool(value=input_after_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='Pool')
    tf.summary.histogram(values=input_after_pool2, name='Pool_Output')

with tf.name_scope('Fc_Layer') as Fc_scope1:
    # 用卷积的方式来实现全连接
    previous_out_shape = input_after_pool2.get_shape().as_list()
    Conv_Filter_3 = tf.Variable(initial_value=tf.truncated_normal(shape=[previous_out_shape[1],
                                                                         previous_out_shape[2]
                                                                         , 50, 10], dtype=tf.float32)/tf.sqrt(
                                previous_out_shape[1]*previous_out_shape[2]*50.),
                                name='FcLayer_Weights')
    tf.summary.histogram(values=Conv_Filter_3, name='Conv_Kernel')
    input_after_fc1_conv = tf.nn.conv2d(input=input_after_pool2, filter=Conv_Filter_3, strides=[1, 1, 1, 1],
                                        padding='VALID')
    # input_after_fc1_conv_shape = input_after_fc1_conv.get_shape().as_list()
    # Conv_Bias = tf.Variable(initial_value=tf.truncated_normal(shape=[input_after_fc1_conv_shape[1],
    #                                                                  input_after_fc1_conv_shape[2],
    #                                                                  10, 1]), name='FcLayer_Bias')
    input_after_fc1 = tf.nn.softmax(input_after_fc1_conv)
    output = tf.reshape(tensor=input_after_fc1, shape=[-1, 10, 1], name='Reshape_Output')
    tf.summary.histogram(values=output, name='Output')

# Loss
epsilon = 1e-8
with tf.name_scope('Loss') as scope_loss:
    cross_entrophy = -tf.reduce_sum(output*tf.log(y_ + epsilon))
    tf.summary.scalar(tensor=cross_entrophy, name='Loss')

# Gradient Descent
with tf.name_scope('Train_Step') as scope_train:
    train_step = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9, name='Gradient_Optimizer').\
        minimize(loss=cross_entrophy, name='Minimize_Loss')

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

training_data, test_data = mnist_loader()
with tf.name_scope('Input_data'):
    with tf.name_scope('training_data'):
        # 生成test data
        test_images = []
        test_labels = []
        for i in range(len(test_data)):
            test_images.append(test_data[i][0])
            test_labels.append(test_data[i][1])
        test_images = np.array(test_images).reshape((len(test_data), 28, 28, 1))
        test_labels = np.array(test_labels).reshape((len(test_data), 10, 1))
    with tf.name_scope('test_data'):
        # 生成training data
        train_images = []
        train_labels = []
        for i in range(len(training_data)):
            train_images.append(training_data[i][0])
            train_labels.append(training_data[i][1])
        train_images = np.array(train_images).reshape((len(training_data), 28, 28, 1))
        train_labels = np.array(train_labels).reshape((len(training_data), 10, 1))


init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logdir='log', graph=tf.get_default_graph())
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
        print(i)
        np.random.shuffle(training_data)
        Batch = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
        for mini_batch in Batch:
            Input = []
            Label = []
            for j in range(len(mini_batch)):
                Input.append(mini_batch[j][0])
                Label.append(mini_batch[j][1])
            Input = np.array(Input).reshape((len(mini_batch), 28, 28, 1))
            Label = np.array(Label).reshape((len(mini_batch), 10, 1))
            # 利用feed_dict给占位符赋值
            train_step.run(feed_dict={x_: Input, y_: Label})
        # 在每个epoch内进行准确率评估
        training_cost.append(cross_entrophy.eval(feed_dict={x_: train_images, y_: train_labels}))
        test_cost.append(cross_entrophy.eval(feed_dict={x_: test_images, y_: test_labels}))
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        # 准确率评估
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        # training accuracy
        print("epoch: %d Finished" % i)
        training_accuracy = accuracy.eval(feed_dict={x_: train_images, y_: train_labels})
        print("Training_Accuracy: %.2f" % training_accuracy)
        # test accuracy
        test_accuracy = accuracy.eval(feed_dict={x_: test_images, y_: test_labels})
        print("Test Accuracy: %.2f" % test_accuracy)
        plt.scatter(i, training_accuracy, marker='*', c='red')
        plt.scatter(i, test_accuracy, marker='+', c='blue')
        # 监控loss在每个epoch内的变化
        result = sess.run(merged, feed_dict={x_: train_images, y_: train_labels})
        writer.add_summary(result, i)

