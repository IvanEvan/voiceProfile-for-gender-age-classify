# -*- coding:utf-8 -*-
import os

# config
sample_rate = 16000.
n_window = 1024
n_overlap = 360
max_len = 240        # sequence max length is 10 s, 240 frames. 
step_time_in_sec = float(n_window - n_overlap) / sample_rate

gender_model = 'models' + os.sep + 'gender_logmel' + os.sep
gender_scaler = 'scalers' + os.sep + 'gender_logmel' + os.sep + 'training-16.scaler'

age_model = 'models' + os.sep + 'age_logmel' + os.sep
age_scaler = 'scalers' + os.sep + 'age_logmel' + os.sep + 'training-9q.scaler'

# 类别标签
gender_lbs = ['man', 'woman']
age_lbs = ['adult', 'child']
