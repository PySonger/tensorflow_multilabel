'''
@Author: Ding Song
@Date: 2019-10-31 16:32:29
@LastEditors: Ding Song
@LastEditTime: 2019-10-31 20:04:55
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

    def build(self,img):
        self.conv1 = self.conv2d(img,'conv1',self.is_training,[5,5,3,20],[1,1,1,1],'VALID')
        self.pool1 = self.max_pool(self.conv1,'pool1',[2,2,20,20],[2,2,2,2],'VALID')
        self.conv2 = self.conv2d(self.pool1,'conv2',self.is_training,[5,5,20,50],[1,1,1,1],'VALID')
        self.pool2 = self.max_pool(self.conv2,'pool2',[2,2,50,50],[2,2,2,2],'VALID')
        self.fc1 = self.fully_connected(self.pool2,'fc1',self.is_training,50,500)
        self.relu1 = tf.nn.relu(self.fc1,'relu1')
        self.fc2 = self.fully_connected(self.relu1,'fc2',self.is_training,500,7)
        self.prob = tf.nn.sigmoid(self.fc2,'prob')