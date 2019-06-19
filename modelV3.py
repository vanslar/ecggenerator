#!/usr/bin/python
#coding=UTF-8
import tensorflow as tf
import numpy as np
import time
import os

class EcgGenerator:
    def __init__(self, cell_units_count, cell_units_feature_len, layer_count, batch_count, seq_length, learning_rate, train_keep_prob, seq_gen_length):
        self.cell_units_count = cell_units_count
        self.cell_units_feature_len = cell_units_feature_len
        self.layer_count = layer_count
        self.batch_count = batch_count
        self.seq_length  = seq_length
        self.learning_rate = learning_rate

        self.train_keep_prob = train_keep_prob
        self.seq_gen_length = seq_gen_length
        self.build_input()
        self.build_model()
        self.build_loss()
        self.build_optimizer()

        self.saver = tf.train.Saver()
    def build_input(self):
        with tf.name_scope('input'):
            self.inputs_ori = tf.placeholder(tf.float32, shape=(self.batch_count, self.seq_length), name='inputs')
            self.inputs  = tf.expand_dims(self.inputs_ori, 2)

            self.targets = tf.placeholder(tf.float32, shape=(self.batch_count, self.seq_gen_length), name='targets')
            self.keep_prob= tf.placeholder(tf.float32, name='prob')
    
    def build_model(self):
        def _get_cell(cell_units_count, out_keep_prob):
            #cell = tf.nn.rnn_cell.BasicLSTMCell(cell_units_count)
            cell = tf.nn.rnn_cell.BasicRNNCell(cell_units_count)
            drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=out_keep_prob)
            return drop
        with tf.name_scope('lstm_model'):
            self.cells = tf.nn.rnn_cell.MultiRNNCell([_get_cell(self.cell_units_count, self.keep_prob) for _ in range(self.layer_count)])
            self.init_state = self.cells.zero_state(self.batch_count, tf.float32)

            self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(self.cells, self.inputs, initial_state=self.init_state)
            self.loss = tf.Variable(0.0) 
            with tf.name_scope('output'):
                x = tf.reshape(self.lstm_out, [-1, self.cell_units_count])
                w = tf.Variable(tf.truncated_normal([self.cell_units_count, 1], stddev=0.1))
                b = tf.Variable(tf.zeros(1))
                y = tf.matmul(x, w) + b #已知数据的预测
                self.inputs_output = tf.reshape(y, [self.batch_count, -1]) #前缀输入数据的输出

#                self.output = self.inputs_output[:, -1]
#                self.output = tf.expand_dims(self.output, 1)
#                self.inputs_output = self.inputs_output[:, :-1]
                new_input = tf.expand_dims(self.inputs_output[:, -1], 1)
                new_state = self.lstm_state
                
                output = list()
                output.append(self.inputs_output[:, -1])

                for i in range(1, self.seq_gen_length):
                    self.lstm_out, new_state = self.cells(new_input, new_state)
                    x = tf.reshape(self.lstm_out, [-1, self.cell_units_count])
                    y = tf.matmul(x, w) + b 
                    new_input = tf.reshape(y, [self.batch_count, -1])
#                    self.output = tf.concat([self.output, tf.reshape(y, [self.batch_count, -1])], 1)
                    output.append(new_input[:, 0])

                self.output = tf.transpose(output, [1, 0])

    def build_loss(self):
        with tf.name_scope('loss'):
            self.input_loss = tf.reduce_mean(tf.nn.l2_loss(self.inputs_output[:, :-1] - self.inputs_ori[:, 1:]))
            self.predict_loss = tf.reduce_mean(tf.nn.l2_loss(self.output - self.targets))
            self.loss = self.input_loss + self.predict_loss

    def build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

#为了截断梯度，也可以人工对梯度做限制
#       with tf.name_scope('optimizer'):
#           tvars = tf.trainable_variables()
#           grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip) # self.grad_clip= 5..表示梯度最大为5
#           train_op = tf.train.AdamOptimizer(self.learning_rate)
#           self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.sess = tf.Session()
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            step = 0

            new_state =  sess.run(self.init_state)
            print('begin training')
            for x, y in batch_generator:
                start_time = time.time()
                i_loss, p_loss, loss, new_state, opt = sess.run([self.input_loss, self.predict_loss, self.loss, self.lstm_state, self.optimizer], 
                                feed_dict={self.inputs_ori:x, 
                                           self.targets:y,
                                           self.keep_prob:self.train_keep_prob,
                                           self.init_state:new_state})

                end_time = time.time()
                step += 1

                if step % log_every_n == 0:
                    print('step: {}/{}'.format(step, max_steps),
                          'loss: {} {} {}'.format(i_loss, p_loss, loss),
                          '{:.4f} sec/batch'.format(end_time-start_time)
                    )
                if step % save_every_n == 0:
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step > max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def eval(self, ecg_data_gen_len, prefix_ecg_data):
        samples = [d for d in prefix_ecg_data]
        x = np.zeros([1, 1])
        sess = self.sess
        new_state = sess.run(self.init_state)

        for data in prefix_ecg_data:
            x[0, 0] = data
            output, new_state = sess.run([self.output, self.lstm_state], 
                            feed_dict={self.inputs_ori:x,  
                                       self.init_state:new_state,
                                       self.keep_prob:1
                            })
        samples.append(output[0][0])
        for _ in range(ecg_data_gen_len):
            x[0, 0] = samples[-1]
            output, new_state = sess.run([self.output, self.lstm_state], 
                            feed_dict={self.inputs_ori:x,  
                                       self.init_state:new_state,
                                       self.keep_prob:1
                            })
            samples.append(output[0][0])
        return np.array(samples)

    def load(self, checkpoint):
        self.sess = tf.Session()
        self.saver.restore(self.sess, checkpoint)
