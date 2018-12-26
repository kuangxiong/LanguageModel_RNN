# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: MakeBatch.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: 2018年12月26日 星期三 20时49分34秒
---------------------------------------------------------
'''
import numpy as np
import tensorflow as tf

TRAIN_DATA = "ptb.train"
TRAIN_BATH_SIZE = 20
TRAIN_NUM_STEP = 35

def read_data(file_path):
    with open(file_path, "r") as fin:
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list

def make_batches(id_list, batch_size, num_step):
    num_batches=(len(id_list)-1)//(batch_size * num_step)
    data = np.array(id_list[:num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    data_batches = np.split(data, num_batches, axis = 1)
    
    label = np.array(id_list[1:num_batches*batch_size*num_step+1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis = 1)

    return list(zip(data_batches, label_batches))

def main():
    train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)





    

