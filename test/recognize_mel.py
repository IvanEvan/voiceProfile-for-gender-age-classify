#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/16 14:55
# @Author  : Evan
# @Site    : 
# @File    : recognize_gender_mel.py
# @Software: PyCharm
from __future__ import print_function
import numpy as np
import glob
import os
import librosa
import pickle
import requests
import wave
import gevent

from io import BytesIO
from scipy import signal
from keras.models import load_model

import config as cfg


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 使用 cpu


def pad_trunc_seq(x, max_len):
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]
    return x_new


def url_receiver(wave_url):
    voice_key = wave_url.split('voiceKey=')[-1]  # 取出 url 里的 voice key
    data = requests.get(wave_url)
    bytes_data = data.content  # 得到 url 对应的二进制音频文件
    wave_obj = wave.open(BytesIO(bytes_data))  # 使用 wave 库打开
    n_frames = wave_obj.getnframes()  # 音频采样点总数目，即总帧长

    samp_width = wave_obj.getsampwidth()  # 采样字节长度一定要为 2
    str_data = wave_obj.readframes(n_frames)  # 读取每个采样点对应的幅度值，返回字符串数据
    int_data = np.fromstring(str_data, dtype=np.int16)  # 将字符川数据转化为 int16 类型数据

    scale = 1. / float(1 << ((8 * samp_width) - 1))
    fmt = '<i{:d}'.format(samp_width)
    flt_data = scale * np.frombuffer(int_data, fmt).astype(np.float32)

    return voice_key, wave_obj, flt_data


def wave_transform_2_image(wave_url):
    target_fs = cfg.sample_rate  # 配置文件 采样频率 16000 kHz/s
    max_len = cfg.max_len  # sequence max length is 10 s, 240 frames
    n_window = cfg.n_window  # 1024
    n_overlap = cfg.n_overlap  # 360

    voice_key, wave_obj, flt_data = url_receiver(wave_url)

    n_channel = wave_obj.getnchannels()  # 声道数一定要为 1 即单声道 Mono
    frame_rate = wave_obj.getframerate()  # 采样频率一定要为 16000 kHz/s

    if n_channel > 1:  # 这里考虑双声道的情况，维度为2，在第二个维度上取均值，变成单声道
        flt_data = np.mean(flt_data, axis=1)
    if frame_rate != target_fs:
        flt_data = librosa.resample(flt_data, orig_sr=frame_rate, target_sr=target_fs)  # 重采样输入信号，到目标采样频率
        frame_rate = target_fs

    # 梅尔变换矩阵 梅尔滤波器个数为64
    melW = librosa.filters.mel(sr=target_fs,
                               n_fft=n_window,
                               n_mels=64,
                               fmin=0.,
                               fmax=8000.)

    # 求频谱图
    ham_win = np.hamming(n_window)  # 采用1024长度的汉宁窗滑动
    [_, _, x] = signal.spectral.spectrogram(x=flt_data,
                                            window=ham_win,
                                            nperseg=n_window,
                                            noverlap=n_overlap,
                                            detrend=False,
                                            return_onesided=True,
                                            mode='magnitude')

    x = x.T
    x = np.dot(x, melW.T)  # 梅尔频谱图
    x = np.log(x + 1e-8)  # log 能量域
    x = pad_trunc_seq(x, max_len)  # 输出 64*240 矩阵，不够 240 补全，超过 240 裁剪掉
    input_array = x.astype(np.float32)
    input_array = input_array[np.newaxis, :, :]  # 2维变3维

    return voice_key, input_array


def load_models_scalers(model_dir, scaler_path):
    [model_path] = glob.glob(os.path.join(model_dir, "*.hdf5"))
    model = load_model(model_path)  # 加载模型结构和参数
    scaler = pickle.load(open(scaler_path, 'rb'))  # 加载训练数据的均值方差用以标准化测试数据

    return model, scaler


def recognize_gender(input_array, model, scaler):
    lables = cfg.gender_lbs  # 类别标签

    (n_clips, n_time, n_freq) = input_array.shape
    x_2d = input_array.reshape((n_clips * n_time, n_freq))
    x2d_scaled = scaler.transform(x_2d)  # 利用训练数据的均值方差标准化测试数据
    x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))

    pred = model.predict(x3d_scaled)  # 预测结果
    max_ind = int(np.argmax(pred, axis=1))
    result = lables[max_ind]  # 预测结果的标签
    probability = pred[0][max_ind]  # 预测结果的概率

    return result, probability


def recognize_age(input_array, model, scaler):
    lables = cfg.age_lbs  # 类别标签

    (n_clips, n_time, n_freq) = input_array.shape
    x_2d = input_array.reshape((n_clips * n_time, n_freq))
    x2d_scaled = scaler.transform(x_2d)  # 利用训练数据的均值方差标准化测试数据
    x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))

    pred = model.predict(x3d_scaled)  # 预测结果
    if float(pred[0][1]) >= 0.76:  # 预测为 child 的概率大于 0.74 才为 child，否则为 adult
        result = lables[1]
        probability = pred[0][1]
    else:
        result = lables[0]
        probability = pred[0][0]

    return result, probability


if __name__ == '__main__':
    g_model, g_scaler = load_models_scalers(cfg.gender_model, cfg.gender_scaler)
    a_model, a_scaler = load_models_scalers(cfg.age_model, cfg.age_scaler)

    voice_key, input_array = wave_transform_2_image('http://nggasrvoice.wsd.com/getvoice?voiceKey=c5fe2b99fe2280b314941efdb3dc2c41')

    threads = gevent.joinall([gevent.spawn(recognize_gender, input_array, g_model, g_scaler),
                              gevent.spawn(recognize_age, input_array, a_model, a_scaler)])

    print(voice_key, [thread.value for thread in threads])
