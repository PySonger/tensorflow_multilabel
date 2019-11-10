'''
@Author: Ding Song
@Date: 2019-11-07 11:32:25
@LastEditors: Ding Song
@LastEditTime: 2019-11-08 10:41:30
@Description: test part.
'''
import os
import tensorflow as tf 
import numpy as np
from dataset import TestDataset,EvalutionDataset


class AvgMeasure(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.right = 0
    
    def append(self,pred,label):
        self.total += len(label)
        for i,j in zip(pred,label):
            self.right += np.array_equal(i,j)

    def cal(self):
        return float(self.right) / self.total

class TestMultiLabel(object):

    def __init__(self,model_path,batch_size,img_size):
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.img_size = img_size
        self.saver = tf.train.import_meta_graph(model_path)
        self.graph = tf.get_default_graph()
        self.tensor_name_list = [tensor.name for tensor in self.graph.as_graph_def().node]

        self.img = self.graph.get_tensor_by_name('input:0')
        self.label = self.graph.get_tensor_by_name('label:0')
        self.pred = self.graph.get_tensor_by_name('pred:0')

    def test(self,data_dir):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.test_data = TestDataset(data_dir,self.img_size,self.batch_size)
        self.test_iterator = self.test_data.make_data().get_next()
        for batch in range(self.test_data.num_batch):
            img = self.sess.run(self.test_iterator)
            pred = self.sess.run(self.pred,feed_dict={self.img:img})
            pred = np.where(pred > 0.5,1,0)
            return pred
            
    def evalution(self,data_file):
        self.avgmea = AvgMeasure()
        self.saver.restore(self.sess,tf.train.latest_checkpoint('models'))
        self.eval_data = EvalutionDataset(data_file,self.img_size,self.batch_size)
        self.eval_iterator = self.eval_data.make_data().get_next()
        for batch in range(self.eval_data.num_batch):
            img,label = self.sess.run(self.eval_iterator)
            pred = self.sess.run(self.pred,feed_dict={self.img:img,self.label:label})
            pred = np.where(pred > 0.5,1,0)
            self.avgmea.append(pred,label)
            if (batch + 1) % 10 == 0:
                print('{} batches has been evalution'.format(batch+1))
        accuracy = self.avgmea.cal()
        print('the accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    model_path = 'models/MultiLabelLenet-42.meta'
    batch_size = 50
    img_size = [48,22]
    data_file = '/media/song/Bigger_Disk/lingang-data/processed_data_by_label/test_data_file.txt'
    test = TestMultiLabel(model_path,batch_size,img_size)
    test.evalution(data_file)