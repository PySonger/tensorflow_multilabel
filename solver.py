'''
@Author: Ding Song
@Date: 2019-10-31 20:03:45
@LastEditors: Ding Song
@LastEditTime: 2019-11-05 00:03:22
@Description: train, evalution and test part.
'''
import os
import tensorflow as tf 
from dataset import Dataset
from model import MultiLabelLenet

os.environ['CUDA_VISIABLE_DEVICES'] = '0'

class Solver(object):

    def __init__(self,device_id,save_dir,num_classes,train_data_file,img_size,crop_size,batch_size,epoch_num,learning_rate,step_size):
        self.save_dir = save_dir
        self.device_id = device_id
        self.epoch_num = epoch_num
        self.step_size = step_size
        self.train_data = Dataset(train_data_file,img_size,crop_size,batch_size,epoch_num)
        self.iterator = self.train_data.make_data().get_next()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        config = tf.ConfigProto()
        config.gpu_options.all_growth = True
        self.sess = tf.Session(config=config)

        with tf.name_scope('build net'):
            self.model = MultiLabelLenet(num_classes,is_training=True)
        
        with tf.name_scope('train model'):
            img = tf.placeholder(dtype=tf.float32,shape=[batch_size,img_size[0],img_size[1],3],name='input')
            self.prob = self.model.build(img)

        with tf.name_scope('cal loss'):
            logit = tf.placeholder(tf.float32,shape=[7,1],name='logit')
            label = tf.placeholder(tf.float32,shape=[7,1],name='label')
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=logit,name='loss')

    def train(self):
        for epoch in self.epoch_num:
            img,label = self.sess.run(self.iterator)
            prob = self.sess.run([self.prob],feed_dict={'img':img})
            loss = self.sess.ruN([self.loss],feed_dict={'logit':prob,'label':label})
            