# -*- coding:utf-8 -*-

# config
sample_rate = 16000.
n_window = 1024
n_overlap = 360  # ensure 240 frames in 10 seconds
max_len = 240  # sequence max length is 10 s, 240 frames.
step_time_in_sec = float(n_window - n_overlap) / sample_rate

# Id of classes
def kinds_of_target(intent):
    if intent == 'age':
        lbs = ['adult', 'child']

    elif intent == 'gender':
        lbs = ['man', 'woman']

    else:
        raise Exception("Please use 'age' or 'gender'")

    num_classes = len(lbs)
    lb_to_idx = {lb: index for index, lb in enumerate(lbs)}

    return lbs, num_classes, lb_to_idx




