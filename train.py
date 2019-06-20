from utils import batch_generator
#import modelCharRnn as model
import modelV3 as model
import wfdb
import numpy as np

sig, field = wfdb.rdsamp('Data/1201_5')
data = sig[:, 0]

batch_count = 100
seq_length = field['fs']*3
feature_length = field['fs']*2

seq_gen_length = 20

bg = batch_generator(data, batch_count, seq_length, seq_gen_length)

Model = model.EcgGenerator(128,  feature_length, 2, batch_count, seq_length, 0.001, 0.5, seq_gen_length)
Model.train(bg, 1000, './models', 100, 5)
