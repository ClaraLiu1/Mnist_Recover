#import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
#https://blog.csdn.net/qq_41185868/article/details/88869740  将来版本中将会弃用前面的input_data，因此要改一下方法
import tensorflow as tf
import config as cfg
import os
import sys
from Lenet_Mnist import Lenet
import numpy as np
import pickle as pk


home='/home/LeNet5_MNIST_2'
MNIST_data='~/PycharmProjects/Mnist_recover/MNIST_data'

def train(mnist):
    sess = tf.Session()
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER


    saver = tf.train.Saver()
    if os.path.exists(parameter_path):
        saver.restore(parameter_path)
    else:
        sess.run(tf.initialize_all_variables())

    for i in range(max_iter):
        batch = mnist.train.next_batch(50)
        if i % 200 == 0:
            train_accuracy = sess.run(lenet.train_accuracy,feed_dict={
                lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]
            })
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(lenet.train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})
    save_path = saver.save(sess, parameter_path)


    def get(n, label):

        x = tf.placeholder(tf.float32, [None,
                                        28,  # 第一维表示一个batch中样例的个数
                                        28,  # 第二维和第三维表示图片的尺寸
                                        1],  # 第四维表示图片的深度，对于RGB格式的图片，深度为5
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        reshaped_x = np.reshape(mnist.test.images[n], (1, 28, 28, 1))
        reshaped_y = np.reshape(mnist.test.labels[n], (1, 10))
        feed_dict = {lenet.input_images: reshaped_x,lenet.input_labels: reshaped_y}
        y_pre, y_label = sess.run([lenet.pred_digits, lenet.input_labels], feed_dict=feed_dict)
        y_prediction = np.reshape(y_pre, (10))
        y_prediction_label = np.reshape(y_label, (10))

        y_prediction = y_prediction.tolist()
        y_prediction_label = y_prediction_label.tolist()

        prediction = y_prediction.index(max(y_prediction))
        prediction_label = y_prediction_label.index(max(y_prediction_label))
        if prediction == prediction_label:
            if not os.path.isdir(home + '/save_p1/%s/%d/pk' % (label, n)):
                os.makedirs(home + '/save_p1/%s/%d/pk' % (label, n))
            with sess.as_default():
                feature_map1 = lenet.net1.eval(feed_dict=feed_dict)
                f = open(home + '/save_p1/%s/%d/pk/out1.pk' % (label, n), 'wb')  #第一个卷积结果
                pk.dump(feature_map1, f)
                f.close()

                feature_map2 = lenet.net2.eval(feed_dict=feed_dict)  #第一个池化结果
                f = open(home + '/save_p1/%s/%d/pk/out2.pk' % (label, n), 'wb')
                pk.dump(feature_map2, f)
                f.close()

                mask=lenet.mask1.eval(feed_dict)  #第一个mask信息的存储
                f=open(home + '/save_p1/%s/%d/pk/mask1.pk' % (label, n), 'wb')
                pk.dump(mask,f)
                f.close()

                feature_map3=lenet.net3.eval(feed_dict)  #第二个卷积结果
                f = open(home + '/save_p1/%s/%d/pk/out3.pk' % (label, n), 'wb')
                pk.dump(feature_map3, f)
                f.close()

                feature_map4=lenet.net4.eval(feed_dict)
                f=open(home + '/save_p1/%s/%d/pk/out4.pk' % (label, n), 'wb')
                pk.dump(feature_map4,f)
                f.close()

                mask = lenet.mask2.eval(feed_dict)  # 第一个mask信息的存储
                f = open(home + '/save_p1/%s/%d/pk/mask2.pk' % (label, n), 'wb')
                pk.dump(mask, f)
                f.close()

                feature_map5=lenet.net5.eval(feed_dict)
                f = open(home + '/save_p1/%s/%d/pk/out5.pk' % (label, n), 'wb')
                pk.dump(feature_map5, f)
                f.close()

                f = open(home + '/save_p1/%s/%d/pk/prediction_%d' % (label, n, prediction), 'w')
                f.write('1 prediction file')
                f.close()

    for i in range(10):
        get(i, 'base')
        print('第%d次' %(i))

    sys.exit()



def main():
    mnist = read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()


