'''
@Author: Ding Song
@Date: 2019-10-31 16:32:29
@LastEditors: Ding Song
@LastEditTime: 2019-10-31 19:33:57
@Description: A LeNet completion with TensorFlow.
'''
import tensorflow as tf 
import numpy as np

class MultiLabelLenet(object):

    def __init__(self,num_classes,is_training=True):
        self.num_classes = num_classes
        self.is_training = is_training

    def conv2d(self,bottom,name,is_training,kernel_size,stride,padding):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(kernel_size,is_training)
            bias = self.get_conv_bias(kernel_size,is_training)
            top = tf.nn.conv2d(bottom,filt,strides=stride,padding=padding)
            top = tf.nn.bias_add(top,bias)
            top = tf.nn.relu(top)
        return top

    def get_conv_filter(self,kernel_size,is_training):
        return tf.get_variable(name='weight',shape=kernel_size,trainable=is_training,
                               initializer=tf.random_normal_initializer(stddev=0.1))

    def get_conv_bias(self,kernel_size,is_training):
        return tf.get_variable(name='bias',shape=kernel_size[-1],trainable=is_training,
                               initializer=tf.constant_initializer(0.00))

    def max_pool(self,bottom,name,kernel_size,stride,padding):
        with tf.variable_scope(name):
            top = tf.nn.max_pool(bottom,kernel_size,strides=stride,padding=padding)
        return top

    def fully_connected(self,bottom,name,is_training,input_size,output_size):
        with tf.variable_scope(name):
            weight = tf.get_variable('weight',shape=[input_size,output_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.001))
            bias = tf.get_variable('bias',shape=output_size,
                                   initializer=tf.constant_initializer(0))
            bottom = tf.reshape(bottom,[-1,input_size])
            top = tf.matmul(bottom,weight)
            top = tf.bias_add(top,bias)
        return top

    def build()