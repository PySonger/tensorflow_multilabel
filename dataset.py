'''
@Author: Ding Song
@Date: 2019-10-31 00:50:16
@LastEditors: Ding Song
@LastEditTime: 2019-10-31 01:44:56
@Description: A dataset compliment with TensorFlow.
'''
import os
import cv2
import tensorflow as tf 
import numpy as np


class MakeDataFile(object):

    def __init__(self,img_dir,save_dir):
        self.img_dir = img_dir
        self.save_dir = save_dir

    def get_file_path(self):
        file_path_list = []
        for root,dirs,files in os.walk(self.img_dir):
            for filename in files:
                if filename.endswith('.jpg'):
                    file_path_list.append(os.path.join(root,filename))
        return file_path_list

    def make_file(self):
        path_list = self.get_file_path()
        lines = [i+'#'+'0 1 0 1\n' for i in path_list]
        with open(os.path.join(self.save_dir,'data_file.txt'),'w') as f:
            for line in lines:
                f.write(line)

class Dataset(object):

    def __init__(self,data_file,img_size,batch_size,epoch_num):
        self.data_file = data_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num

    def analysis_data_file(self):
        with open(self.data_file,'r') as f:
            data_lines = f.readlines()
        path_list,label_list = [],[]
        for line in data_lines:
            path,labels = line.strip().split('#')
            labels = list(map(int,labels.split()))
            path_list.append(path)
            label_list.append(labels)
        return path_list,label_list

    def make_data(self):
        path_list,label_list = self.analysis_data_file()
        file_paths = tf.constant(path_list)
        labels = tf.constant(label_list)
        dataset = tf.data.Dataset.from_tensor_slices((file_paths,labels))
        dataset.map(self.parser_function)
        dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat(self.epoch_num)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def parser_function(self,file_path,label):
        image_string = tf.read_file(file_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded = tf.image.resize_images(image_decoded,self.img_size)
        return image_decoded,label
        
def main():
    img_dir = '/media/song/机械盘1/my_python_file/dataset/天气识别/Train'
    save_dir = '/media/song/机械盘1/my_python_file/dataset/天气识别/'
    make_data_file = MakeDataFile(img_dir,save_dir)
    make_data_file.make_file()
    data_file = os.path.join(save_dir,'data_file.txt')
    data = Dataset(data_file,[50,50],2,1)
    iterator = data.make_data()
    one_element =  iterator.get_next()
    with tf.Session() as sess:
        i = 0
        try:
            while True:
                print(i)
                print(sess.run(one_element))
                i += 1
        except:
            print('done...')

if __name__ == "__main__":
    main()

