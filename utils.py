#!/usr/bin/python
import numpy as np

def batch_generator(ecg_sample, batch_size, seq_length):
    arr = ecg_sample[:]
    ecg_len = len(arr)
    n_batchs = ecg_len // (batch_size*seq_length)
    ecg_sample = arr[:batch_size*seq_length*n_batchs]
    ecg_sample = ecg_sample.reshape([n_batchs, -1])
    while True:
        for n in range(n_batchs):
            x = ecg_sample[n]
            x = x.reshape([batch_size, seq_length])
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


