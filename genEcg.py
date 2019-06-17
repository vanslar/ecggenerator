#!/usr/bin/python
import model
import wfdb
import tensorflow as tf

ecg_record = wfdb.rdsamp('1201_5', physical=True)
data = ecg_record.p_signals[:, 1]
pre_data = data[500:1000]

batch_count = 1
seq_length = 1
Model = model.EcgGenerator(128, 2, batch_count, seq_length, 0.001, 0.5)

Model.load(tf.train.latest_checkpoint('./models'))
result = Model.eval(5000, pre_data)
with open('result.txt', 'w') as fid:
    for i in result:
        fid.write(str(i)+'\n')