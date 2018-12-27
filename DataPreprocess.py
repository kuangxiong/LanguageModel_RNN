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
import codecs 
import sys
import numpy as np
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


RAW_DATA="simple-examples/data/ptb.test.txt"
VOCAB = "ptb.vocab"
OUTPUT_DATA = 'ptb.test'

with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]

word_to_id={k:v for (k, v) in zip(vocab, range(len(vocab)))}

def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

fin = codecs.open(RAW_DATA, "r", "utf-8")
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')

for line in fin:
    words = line.strip().split()+["<eos>"]
    out_line=' '.join([str(get_id(w)) for w in words]) +'\n'
    fout.write(out_line)
fin.close()
fout.close()



