# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: IntegrateTrain.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: 2018年12月26日 星期三 21时55分02秒
---------------------------------------------------------
'''
import numpy as np
import tensorflow as tf

TRAIN_DATA = "ptb.train"
EVAL_DATA = "ptb.valid"
TEST_DATA = "ptb.test"

HIDDEN_SIZE = 300
NUM_LAYERS = 2
VOCAB_SIZE = 10000
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP =1
NUM_EPOCH = 5
LSTM_KEEP_PROB = 0.9
EMBEDDING_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batche_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.output_data = tf.placeholder(tf.int32, [batch_size, num_steps])

        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([drop]*NUM_LAYERS)
        
        self.initial_state = cell.zero_state([batch_size], tf.float32)

        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        output = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.trainspose(embedding)
        else:
            weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bais", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bais

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.reshape(self.targets, [-1]),
                logits = logits)
        self.cost = tf.recude_sum(loss)/batch_size
        self.final_state = state

        if not is_training:return

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.0)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


def run_epoch(session, model, batches, train_op, output_log, step):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for x, y in batches:
        feed = {model.input_data:x, 
                model.targets: y, 
                model.initial_state:state}
        cost, state, _ = session.run(
                [model.cost, model.final_state, train_op], feed_dict = feed)

        total_costs +=cost
        iters += model.num_steps

        if output_log and step %100 ==0:
            print("After %d steps, perplexity is %.3f"%(step, np.exp(total_costs/iters)))
        step +=1

    return step, np.exp(total_costs/iters)

