# -*- coding:utf-8 -*-
import os

# config
sample_rate = 16000.
n_window = 1024
n_overlap = 360
max_len = 240        # sequence max length is 10 s, 240 frames. 
step_time_in_sec = float(n_window - n_overlap) / sample_rate
model_dir = '.' + os.sep + 'models' + os.sep + 'crnn_sed-16' + os.sep
scaler_path = '.' + os.sep + 'scalers' + os.sep + 'logmel' + os.sep + 'training-16.scaler'

# 类别标签
lbs = ['man', 'woman']

idx_to_lb = {index: lb for index, lb in enumerate(lbs)}
lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
num_classes = len(lbs)
