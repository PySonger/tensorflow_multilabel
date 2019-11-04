'''
@Author: Ding Song
@Date: 2019-10-31 20:03:45
@LastEditors: Ding Song
@LastEditTime: 2019-11-04 20:19:16
@Description: train, evalution and test part.
'''
import os
import tensorflow as tf 
from dataset import Dataset
from model import MultiLabelLenet

os.environ['CUDA_VISIABLE_DEVICES'] = '0'

class Solver(object):

    def __init__(self,device_id,save_dir,num_classes,train_data_file,img_size,crop_size,batch_size,epoch_num,learning_rate):
        self.save_dir = save_dir
        self.device_id = device_id
        self.train_data = Dataset(train_data_file,img_size,crop_size,batch_size,epoch_num)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.loss = 
        config = tf.ConfigProto()
        config.gpu_options.all_growth = True
        self.sess = tf.Session(config=config)

        with tf.name_scope('build net'):
            s

    def train(self,train_data):

            