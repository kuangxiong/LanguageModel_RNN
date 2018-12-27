# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: main.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: 2018年12月26日 星期三 22时38分25秒
---------------------------------------------------------
'''
from TrainProcess import *
from DataPreprocess import *
import tensorflow as tf
import numpy as np
TRAIN_DATA = "ptb.train"
EVAL_DATA = "ptb.eval"
TEST_DATA = "ptb.test"


def main():
    #define initial function
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    #define RNN model used in train process
    with tf.variable_scope("language_model", reuse=None, initializer= initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    #define RNN model used in test process
    with tf.variable_scope("language_model", reuse = True, initializer = initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        eval_batches = make_batches(read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        test_batches = make_batches(read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        
        step = 0
        for i in range(NUM_EPOCH):
            print("In iteration: %d"%(i+1))
            step, train_pplx = run_epoch(session, train_model, train_batches, train_model.train_op, True, step)
            print("Epoch: %d Train Perplexity:%.3f"%(i+1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches, tf.no_op(), False, 0)
            print("Epoch: %d Eval Perplexity:%.3f"%(i+1, eval_pplx))
        _, test_pplx = run_epoch(session, eval_model, test_batches, tf.no_op(), False, 0)
        print("Test Perplexity:%.3f"%(test_pplx))

if __name__=="__main__":
    main()




