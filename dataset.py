'''
@Author: Ding Song
@Date: 2019-10-31 00:50:16
@LastEditors: Ding Song
@LastEditTime: 2019-11-08 10:19:45
@Description: A dataset compliment with TensorFlow.
'''
import os
import copy
import cv2
import tensorflow as tf 
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class EvalutionDataset(object):

    def __init__(self,data_file,img_size,batch_size):
        self.data_file = data_file
        self.img_size = img_size
        self.batch_size = batch_size
        path_list,label_list = self.analysis_data_file()
        self.path_list = path_list
        self.label_list = label_list
        self.num_batch = int(np.ceil(float(len(path_list)) / self.batch_size))

    def analysis_data_file(self):
        with open(self.data_file,'r') as f:
            data_lines = f.readlines()
        rd.shuffle(data_lines)
        path_list,label_list = [],[]
        for line in data_lines:
            path,labels = line.strip().split('#')
            labels = list(map(int,labels.split()))
            path_list.append(path)
            label_list.append(labels)
        return path_list,label_list

    def make_data(self):
        file_paths = tf.constant(self.path_list)
        labels = tf.constant(self.label_list)
        dataset = tf.data.Dataset.from_tensor_slices((file_paths,labels))
        dataset = dataset.map(self.parser_function)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def parser_function(self,file_path,label):
        image_string = tf.read_file(file_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded = tf.image.resize_images(image_decoded,self.img_size)
        #image_decoded = tf.reshape(image_decoded,[tf.shape(image_decoded)[0],tf.shape(image_decoded)[1],3])
        return image_decoded * 0.00390625,label
 
class TestDataset(object):

    def __init__(self,data_dir,img_size,batch_size):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.path_list = self.get_data_path()
        self.num_batch = int(np.ceil(float(len(self.path_list)) / self.batch_size))

    def get_data_path(self):
        path_list = []
        for root,dirs,files in os.walk(self.data_dir):
            for filename in files:
                if filename.endswith('.jpg'):
                    path_list.append(os.path.join(root,filename))
        return path_list

    def make_data(self):
        file_paths = tf.constant(self.path_list)
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(self.parser_function)
        dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def parser_function(self,file_path):
        image_string = tf.read_file(file_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded = tf.image.resize_images(image_decoded,self.img_size)
        image_decoded = tf.reshape(image_decoded,[1,tf.shape(image_decoded)[0],tf.shape(image_decoded)[1],3])
        return image_decoded * 0.00390625

class Dataset(object):

    def __init__(self,data_file,img_size,crop_size,batch_size,epoch_num):
        self.data_file = data_file
        self.img_size = img_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        path_list,label_list = self.analysis_data_file()
        self.path_list = path_list
        self.label_list = label_list
        self.num_batch = int(np.ceil(float(len(path_list)) / self.batch_size))

    def analysis_data_file(self):
        with open(self.data_file,'r') as f:
            data_lines = f.readlines()
        rd.shuffle(data_lines)
        path_list,label_list = [],[]
        for line in data_lines:
            path,labels = line.strip().split('#')
            labels = list(map(int,labels.split()))
            path_list.append(path)
            label_list.append(labels)
        return path_list,label_list

    def make_data(self):
        file_paths = tf.constant(self.path_list)
        labels = tf.constant(self.label_list)
        dataset = tf.data.Dataset.from_tensor_slices((file_paths,labels))
        dataset = dataset.map(self.parser_function)
        dataset = dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat(self.epoch_num)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def parser_function(self,file_path,label):
        image_string = tf.read_file(file_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded = tf.image.resize_images(image_decoded,self.img_size)

        image_decoded = tf.reshape(image_decoded,[1,tf.shape(image_decoded)[0],tf.shape(image_decoded)[1],3])
        box_ind = tf.constant([0],dtype=tf.int32)
        crop_rate = [float(self.crop_size[0])/self.img_size[0], float(self.crop_size[1])/self.img_size[1]]
        y,x = rd.uniform(0,1 - crop_rate[0]),rd.uniform(0,1 - crop_rate[1])
        boxes = tf.constant([[y,x,crop_rate[0],crop_rate[1]]],dtype=tf.float32)
        crop_size = tf.constant(self.crop_size,dtype=tf.int32)

        crop_boxes = tf.image.crop_and_resize(image_decoded,boxes,box_ind=box_ind,crop_size=crop_size)
        crop_boxes = crop_boxes
        return crop_boxes[0] * 0.00390625,label
        
def main():
    data_file = '/media/song/Bigger_Disk/lingang-data/processed_data_by_label/balanced_cleaned_data_file.txt'
    data = Dataset(data_file,[50,25],[48,22],2,5)
    iterator = data.make_data()
    one_element =  iterator.get_next()
    with tf.Session() as sess:
        i = 0
        try:
            while i < 10:
                print(i)
                img,label = sess.run(one_element)
                print(np.unique(img))
                i += 1
        except:
            print('done...')


if __name__ == "__main__":
    main()

