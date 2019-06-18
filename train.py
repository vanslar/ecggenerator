from utils import batch_generator
#import modelCharRnn as model
import modelV2 as model
import wfdb
import numpy as np

sig, field = wfdb.rdsamp('Data/1201_5')
data = sig[:, 0]

batch_count = 100
#seq_length = field['fs']*5
#seq_gen_length = field['fs']*5
seq_length = 10 #field['fs']
seq_gen_length = 10 #field['fs']
bg = batch_generator(data, batch_count, seq_length, seq_gen_length)

Model = model.EcgGenerator(128, 2, batch_count, seq_length, 0.001, 0.5, seq_gen_length)
Model.train(bg, 1000, './models', 100, 5)
