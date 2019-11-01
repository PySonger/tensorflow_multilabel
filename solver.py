'''
@Author: Ding Song
@Date: 2019-10-31 20:03:45
@LastEditors: Ding Song
@LastEditTime: 2019-11-01 19:05:11
@Description: train, evalution and test part.
'''
import os
import tensorflow as tf 
from dataset import Dataset
from model import MultiLabelLenet

class Solver(object):

    def __init__(self,device_id,save_dir,num_classes):
        self.device_id = device_id
        self.save_dir = save_dir
        self.model = MultiLabelLenet(num_classes)
        pass

    def train(self):
        with tf.device('/gpu:'+str(self.device_id)):
            