from utils import batch_generator
import model
import wfdb
import numpy as np

#ecg_record = wfdb.rdsamp('1201_5', physical=False)
#data = ecg_record.d_signals[:, 0]

ecg_record = wfdb.rdsamp('1201_5', physical=True)
data = ecg_record.p_signals[:, 0]

batch_count = 100
seq_length = ecg_record.fs*5
bg = batch_generator(data, batch_count, seq_length)

Model = model.EcgGenerator(128, 2, batch_count, seq_length, 0.001, 0.5)
Model.train(bg, 1000, './models', 100, 5)