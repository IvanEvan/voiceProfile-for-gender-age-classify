# -*- coding:utf-8 -*-
import numpy as np
import random

class BalanceDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        n_each = batch_size // n_labs   
        
        index_list = []
        for i1 in range(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in range(n_labs):
            np.random.shuffle(index_list[i1])
        
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            for i1 in range(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_each, len_list[i1])]
                batch_x.append(x[batch_idx])
                batch_y.append(y[batch_idx])
                pointer_list[i1] += n_each
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

class RatioDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100, verbose=1):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self._verbose_ = verbose
            
    def _get_lb_list(self, n_samples_list):
        lb_list = []
        for idx in range(len(n_samples_list)):
            n_samples = n_samples_list[idx]
            if n_samples < 1000:
                lb_list += [idx]
            elif n_samples < 2000:
                lb_list += [idx] * 2
            elif n_samples < 3000:
                lb_list += [idx] * 3
            elif n_samples < 4000:
                lb_list += [idx] * 4
            else:
                lb_list += [idx] * 5
        return lb_list
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        
        if self._verbose_ == 1:
            print("n_samples_list: %s" % (n_samples_list,))
            print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        index_list = []
        for i1 in range(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in range(n_labs):
            np.random.shuffle(index_list[i1])
        
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]
            
            for i1 in range(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_per_class_list[i1], len_list[i1])]
                batch_x.append(x[per_class_batch_idx])
                batch_y.append(y[per_class_batch_idx])
                pointer_list[i1] += n_per_class_list[i1]
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

