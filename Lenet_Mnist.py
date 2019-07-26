import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg
from new_pooling import max_pool_with_argmax



class Lenet:
    def __init__(self):
        self.raw_input_image = tf.placeholder(tf.float32, [None, 784])
        self.input_images = tf.reshape(self.raw_input_image, [-1, 28, 28, 1])
        self.raw_input_label = tf.placeholder("float", [None, 10])
        self.input_labels = tf.cast(self.raw_input_label,tf.int32)
        self.dropout = cfg.KEEP_PROB

        with tf.variable_scope("Lenet") as scope:
            self.train_digits = self.construct_net(True)
            scope.reuse_variables()
            self.pred_digits = self.construct_net(False)

        self.prediction = tf.argmax(self.pred_digits, 1)
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.input_labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))


        self.loss = slim.losses.softmax_cross_entropy(self.train_digits, self.input_labels)
        self.lr = cfg.LEARNING_RATE
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.net1 = slim.conv2d(self.input_images, 6, [5, 5], 1, padding='SAME', scope='conv1')
        self.net2, self.mask1 = max_pool_with_argmax(self.net1, 2)

        self.net3 = slim.conv2d(self.net2, 16, [5,5], 1,padding='VALID', scope='conv3')
        self.net4, self.mask2 = max_pool_with_argmax(self.net3, 2)
        self.net5 = slim.conv2d(self.net4, 120, [5,5], 1, padding='VALID',scope='conv5')
        self.w1=slim.model_variable('weights',shape=[])



    def construct_net(self,is_trained = True):
        with slim.arg_scope([slim.conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            self.net1 = slim.conv2d(self.input_images,6,[5,5],1,padding='SAME',scope='conv1')
            self.net2,self.mask1=max_pool_with_argmax(self.net1,2)
            self. net3 = slim.conv2d(self.net2,16,[5,5],1,padding='VALID',scope='conv3')
            self.net4,self.mask2=max_pool_with_argmax(self.net3,2)
            self.net5 = slim.conv2d(self.net4,120,[5,5],1,padding='VALID',scope='conv5')
            self.net5 = slim.flatten(self.net5, scope='flat6')
            net6 = slim.fully_connected(self.net5, 84, scope='fc7')
            net6 = slim.dropout(net6, self.dropout,is_training=is_trained, scope='dropout8')
            digits = slim.fully_connected(net6, 10, scope='fc9')
        return digits

