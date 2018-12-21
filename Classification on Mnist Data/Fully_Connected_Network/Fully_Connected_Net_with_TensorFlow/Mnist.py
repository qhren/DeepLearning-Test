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

g1 = tf.Graph()
with g1.as_default():
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

        # Variable的保存和加载
        # 需要保存的Variable的名字可以同{'name': Variable}形式给出
        # saver = tf.train.Saver()
        # saver.save(sess, save_path='%s%s' % (settings.Project_Path + 'Exp3\\Save_Variables\\', 'temp.ckpt'),
        # global_step=10)
        # 可通过chkp查看文件中的Tensor和Content
        # chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

        # 变量的加载 恢复后的变量不需要经过initializer的初始化
        # saver = tf.train.Saver()
        # saver.restore(sess, save_path="%s")  # 需要在Session内进行

        # Model的简单保存和加载
        # 保存
        # tf.saved_model.simple_save(session=sess, export_dir="%s" % output_dir, inputs={"x": x, "y": y},
        # outputs={"output": output})
        # 加载
        # export_dir = '%s'
        # with tf.Session(graph=tf.Graph()) as sess:
        #     tf.saved_model.loader.load(sess, export_dir='%s', tags=[])
