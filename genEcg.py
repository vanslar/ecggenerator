#!/usr/bin/python
#import modelCharRnn
import modelV2 as model
import wfdb
import tensorflow as tf


sig, field = wfdb.rdsamp('Data/1201_5')
data = sig[:, 1]
pre_data = data[500:1000]

batch_count = 1
seq_length = 1
#Model = model.EcgGenerator_CharRnn(128, 2, batch_count, seq_length, 0.001, 0.5)
#Model = model.EcgGenerator(128, 2, batch_count, seq_length, 0.001, 0.5, 1000)
Model = model.EcgGenerator(128, 2, batch_count, seq_length, 0.001, 0.5, 1)

Model.load(tf.train.latest_checkpoint('./models'))
result = Model.eval(5000, pre_data)
with open('result.txt', 'w') as fid:
    for i in result:
        fid.write(str(i)+'\n')
