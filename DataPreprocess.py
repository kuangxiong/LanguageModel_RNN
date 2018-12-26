# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: DataPreprocess.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: 2018年12月26日 星期三 22时40分00秒
---------------------------------------------------------
'''

import codecs
import collections
from operator import itemgetter

RAW_DATA = "simple-examples/data/ptb.train.txt"
VOCAB_OUTPUT = 'ptb.vocab'

counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] +=1

sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse = True)

sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words=["<eos>"] + sorted_words

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word+'\n')


#TRAIN_DATA = "ptb.train"
#TRAIN_BATH_SIZE = 20
#TRAIN_NUM_STEP = 35

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

