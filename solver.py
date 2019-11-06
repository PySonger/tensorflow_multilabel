'''
@Author: Ding Song
@Date: 2019-10-31 20:03:45
@LastEditors: Ding Song
@LastEditTime: 2019-11-06 19:39:54
@Description: train, evalution and test part.
'''
import os
import tensorflow as tf 
import numpy as np
from dataset import Dataset
from model import MultiLabelLenet

os.environ['CUDA_VISIABLE_DEVICES'] = '0'

class Solver(object):

    def __init__(self,save_dir,num_classes,train_data_file,img_size,crop_size,batch_size,epoch_num,learning_rate,step_size):
        self.save_dir = save_dir
        self.makedirs(self.save_dir)
        self.epoch_num = epoch_num
        self.step_size = step_size
        self.decay_rate = 0.2
        self.learning_rate = learning_rate
        self.train_data = Dataset(train_data_file,img_size,crop_size,batch_size,epoch_num)
        self.iterator = self.train_data.make_data().get_next()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model = MultiLabelLenet(num_classes,is_training=True)
        
        self.input_img = tf.placeholder(dtype=tf.float32,shape=[None,crop_size[0],crop_size[1],3],name='input')
        self.model.build(self.input_img)

        self.prob = self.model.fc2
        #pred equals to sigmoid(fc2)
        self.pred = self.model.prob

        self.label = tf.placeholder(tf.float32,shape=[None,7],name='label')
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prob,labels=self.label))
        self.train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(tf.trainable_variables())

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        for epoch in range(self.epoch_num):
            for batch in range(self.train_data.num_batch):
                img,label = self.sess.run(self.iterator)
                pred,loss,_ = self.sess.run([self.pred,self.loss,self.train_op],feed_dict={self.input_img:img,self.label:label})
                pred = np.where(pred > 0.5,1,0)
                if (1+batch) % 100 == 0:
                    print('[{} | {}] {}  {}  {}  {}'.format(epoch+1,self.epoch_num,self.optimizer._learning_rate,loss,pred[0],label[0]))
            if (1+epoch) % self.step_size == 0:
                self.optimizer._learning_rate *= self.decay_rate
                print(self.optimizer._learning_rate)
                self.save(epoch)

    def makedirs(self,dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def save(self,epoch):
        self.saver.save(self.sess,os.path.join(self.save_dir,'MultiLabelLenet'),global_step=epoch+1)
        print('the model of {}th epoch has been saved...'.format(epoch+1))

if __name__ == '__main__':
    save_dir = 'models'
    num_classes = 7
    train_data_file = '/media/song/Bigger_Disk/lingang-data/processed_data_by_label/balanced_cleaned_data_file.txt'
    img_size = (50,25)
    crop_size = (48,22)
    batch_size = 60
    epoch_num = 20
    learning_rate = 0.001
    step_size = 5
    solver = Solver(save_dir,num_classes,train_data_file,img_size,crop_size,batch_size,epoch_num,learning_rate,step_size)
    solver.train()