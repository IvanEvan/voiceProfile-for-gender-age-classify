# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import os
import librosa
import cPickle
import pickle
import time
import h5py
import argparse
from scipy import signal
from sklearn import preprocessing

import config as cfg


# Read wav
def read_audio(path, target_fs=None):

    audio, fs = librosa.load(path, sr=None)  # 此处 audio 数据类型为 float34

    if audio.ndim > 1:  # 维度>1，这里考虑双声道的情况，维度为2，在第二个维度上取均值，变成单声道
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)  # 重采样输入信号，到目标采样频率
        fs = target_fs
    audio = audio.astype(np.float32)

    return audio, fs


# Create an empty folder
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


### Feature extraction. 
def extract_features(wav_dir, out_dir, recompute):
    """Extract log mel spectrogram features. 
    
    Args:
      wav_dir: string, directory of wavs. 
      out_dir: string, directory to write out features. 
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features. 
                 
    Returns:
      None
    """
    fs = cfg.sample_rate  # 配置文件 16000.
    n_window = cfg.n_window  # 1024
    n_overlap = cfg.n_overlap  # 360
    
    create_folder(out_dir)
    names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
    names = sorted(names)
    print("Total file number: %d" % len(names))
    
    # Mel filter bank  返回梅尔变换矩阵，梅尔滤波器数目参数为 n_mels ,建议选用64的整数倍
    melW = librosa.filters.mel(sr=fs, 
                               n_fft=n_window, 
                               n_mels=64, 
                               fmin=0., 
                               fmax=8000.)
    
    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = wav_dir + '/' + na
        out_path = out_dir + '/' + os.path.splitext(na)[0] + '.p'
        
        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            print(cnt, out_path)
            (audio, _) = read_audio(wav_path, fs)
            
            # Skip corrupted wavs
            if audio.shape[0] == 0:
                print("File %s is corrupted!" % wav_path)
            else:
                # Compute spectrogram
                ham_win = np.hamming(n_window)  # 采用1024长度的汉宁窗滑动
                [f, t, x] = signal.spectral.spectrogram(x=audio,
                                                        window=ham_win,
                                                        nperseg=n_window,
                                                        noverlap=n_overlap,
                                                        detrend=False,
                                                        return_onesided=True,
                                                        mode='magnitude')
                x = x.T
                x = np.dot(x, melW.T)
                x = np.log(x + 1e-8)
                x = x.astype(np.float32)
                
                # Dump to pickle
                cPickle.dump(x, open(out_path, 'wb'),
                             protocol=cPickle.HIGHEST_PROTOCOL)
                # cPickle.dump(x, open(out_path, 'wb'), protocol=4)
        cnt += 1
    print("Extracting feature time: %s" % (time.time() - t1,))


### Constant-Q-Transform Feature extraction. (by Evan Yang)
def extract_cqt_features(wav_dir, out_dir, recompute):
    """Extract Constant-Q-Transform features.

    Args:
      wav_dir: string, directory of wavs.
      out_dir: string, directory to write out features.
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features.

    Returns:
      None
    """
    fs = cfg.sample_rate  # 配置文件 16000.

    create_folder(out_dir)
    names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
    names = sorted(names)
    print("Total file number: %d" % len(names))

    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = wav_dir + '/' + na
        out_path = out_dir + '/' + os.path.splitext(na)[0] + '.p'

        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            print(cnt, out_path)
            (audio, _) = read_audio(wav_path, fs)

            # Skip corrupted wavs
            if audio.shape[0] == 0:
                print("File %s is corrupted!" % wav_path)
            else:
                # Compute Constant-Q-Transform
                x = librosa.cqt(y=audio, hop_length=512, sr=fs, n_bins=80, bins_per_octave=12, window='hann')
                x = x.T
                x = np.abs(np.log(x + 1e-8))
                x = x.astype(np.float32)

                # Dump to pickle
                cPickle.dump(x, open(out_path, 'wb'),
                             protocol=cPickle.HIGHEST_PROTOCOL)
                # cPickle.dump(x, open(out_path, 'wb'), protocol=4)
        cnt += 1
    print("Extracting feature time: %s" % (time.time() - t1,))


### Harmonic-Percussive Source Separation Feature extraction. (by Evan Yang)
def extract_hpss_features(wav_dir, out_dir, recompute):
    """Extract Harmonic-Percussive Source Separation features.

    Args:
      wav_dir: string, directory of wavs.
      out_dir: string, directory to write out features.
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features.

    Returns:
      None
    """
    # 改 calculate_scaler() 的 (n_clips, n_time, n_freq) 为 (n_clips, n_time, n_freq, n_channel)
    # 改 main_crnn_sed.py 的 (n_time, n_freq, 1) 为 (n_time, n_freq, 2)
    fs = cfg.sample_rate  # 配置文件 16000.
    max_len = cfg.max_len  # sequence max length is 10 s, 240 frames.
    create_folder(out_dir)
    names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
    names = sorted(names)
    print("Total file number: %d" % len(names))

    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = wav_dir + '/' + na
        out_path = out_dir + '/' + os.path.splitext(na)[0] + '.p'

        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            print(cnt, out_path)
            (audio, _) = read_audio(wav_path, fs)

            # Skip corrupted wavs
            if audio.shape[0] == 0:
                print("File %s is corrupted!" % wav_path)
            else:
                # Compute HPSS
                y_harmonic, y_percussive = librosa.effects.hpss(audio)
                # harmonic 分量的梅尔能量谱
                mel_harmonic = librosa.feature.melspectrogram(y_harmonic,
                                                              sr=fs,
                                                              hop_length=1024,
                                                              n_fft=1024,
                                                              n_mels=64,
                                                              fmax=8000)
                mel_harmonic = librosa.power_to_db(mel_harmonic)
                mel_harmonic = (mel_harmonic - np.mean(mel_harmonic)) / np.std(mel_harmonic)
                mel_harmonic = mel_harmonic.T
                mel_harmonic = pad_trunc_seq(mel_harmonic, max_len)
                # percussive 分量的梅尔能量谱
                mel_percussive = librosa.feature.melspectrogram(y_percussive,
                                                                sr=fs,
                                                                hop_length=1024,
                                                                n_fft=1024,
                                                                n_mels=64,
                                                                fmax=8000)
                mel_percussive = librosa.power_to_db(mel_percussive)
                mel_percussive = (mel_percussive - np.mean(mel_percussive)) / np.std(mel_percussive)
                mel_percussive = mel_percussive.T
                mel_percussive = pad_trunc_seq(mel_percussive, max_len)
                # 将连个分量矩阵叠加
                mel_spec = np.stack([mel_harmonic, mel_percussive], axis=2)
                x = mel_spec.astype(np.float32)

                # Dump to pickle
                cPickle.dump(x, open(out_path, 'wb'),
                             protocol=cPickle.HIGHEST_PROTOCOL)
                # cPickle.dump(x, open(out_path, 'wb'), protocol=4)
        cnt += 1
    print("Extracting feature time: %s" % (time.time() - t1,))


### Pack features of hdf5 file
def pack_features_to_hdf5(intent, fe_dir, out_path):
    """Pack extracted features to a single hdf5 file. 
    
    This hdf5 file can speed up loading the features. This hdf5 file has 
    structure:
       na_list: list of names
       x: bool array, (n_clips)
       y: float32 array, (n_clips, n_time, n_freq)
       
    Args:
      intent: string, gender or age intent
      fe_dir: string, directory of features.
      out_path: string, path to write out the created hdf5 file.
    Returns:
      None
    """
    max_len = cfg.max_len  # sequence max length is 10 s, 240 frames.
    create_folder(os.path.dirname(out_path))
    
    t1 = time.time()
    x_all, y_all, na_all = [], [], []

    # Pack from features without ground truth label (dev. data)
    names = os.listdir(fe_dir)  # 上一步保存的 feature
    names = sorted(names)

    for fe_na in names:
        bare_na = os.path.splitext(fe_na)[0]
        fe_path = os.path.join(fe_dir, fe_na)
        na_all.append(bare_na + ".wav")
        x = cPickle.load(open(fe_path, 'rb'))
        x = pad_trunc_seq(x, max_len)  # 如果使用 HPSS 特征，记得注释掉此行
        x_all.append(x)

        lbs, num_classes, lb_to_idx = cfg.kinds_of_target(intent)

        if ('_' + lbs[0]) in bare_na:
            idx = lb_to_multinomial(lbs[0], num_classes, lb_to_idx)
        elif ('_' + lbs[1]) in bare_na:
            idx = lb_to_multinomial(lbs[1], num_classes, lb_to_idx)
        else:
            raise Exception("Note the number of labels in config.py")
        y_all.append(idx)

    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.bool)

    # create_dataset 只接受 ascii 的数据，如果你是Python3或者python使用utf-8的编码就会报错
    # n_na_all = [i.encode() for i in na_all]
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('na_list', data=na_all)
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        
    print("Save hdf5 to %s" % out_path)
    print("Pack features time: %s" % (time.time() - t1,))


def lb_to_multinomial(lables, num_classes, lb_to_idx):
    """Ids of wav to multinomial representation.

    Args:
      lables: list of lable, e.g. ['child', 'child', 'adult', ...]
      num_classes: labels' number
      lb_to_idx: a dict object from config.py, e.g. {'child':1, ''child':1, 'adult':0, ...}
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    y = np.zeros(num_classes)
    index = lb_to_idx[lables]
    y[index] = 1
    return y


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length. 
    
    Args:
      x: ndarray, input sequence data. 
      max_len: integer, length of sequence to be padded or truncated. 
      
    Returns:
      ndarray, Padded or truncated input sequence data. 
    """
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]

    return x_new


### Load data & scale data
def load_hdf5_data(hdf5_path, verbose=1):
    """Load hdf5 data. 
    
    Args:
      hdf5_path: string, path of hdf5 file. 
      verbose: integar, print flag. 
      
    Returns:
      x: ndarray (np.float32), shape: (n_clips, n_time, n_freq)
      y: ndarray (np.bool), shape: (n_clips, n_classes)
      na_list: list, containing wav names. 
    """
    t1 = time.time()
    with h5py.File(hdf5_path, 'r') as hf:
        x = np.array(hf.get('x'))
        y = np.array(hf.get('y'))
        na_list = list(hf.get('na_list'))
        
    if verbose == 1:
        print("--- %s ---" % hdf5_path)
        print("x.shape: %s %s" % (x.shape, x.dtype))
        print("y.shape: %s %s" % (y.shape, y.dtype))
        print("len(na_list): %d" % len(na_list))
        print("Loading time: %s" % (time.time() - t1,))
        
    return x, y, na_list


def calculate_scaler(hdf5_path, out_path):
    """Calculate scaler of input data on each frequency bin. 
    
    Args:
      hdf5_path: string, path of packed hdf5 features file. 
      out_path: string, path to write out the calculated scaler. 
      
    Returns:
      None. 
    """
    create_folder(os.path.dirname(out_path))
    t1 = time.time()
    (x, y, na_list) = load_hdf5_data(hdf5_path, verbose=1)
    (n_clips, n_time, n_freq) = x.shape
    x2d = x.reshape((n_clips * n_time, n_freq))
    scaler = preprocessing.StandardScaler().fit(x2d)
    print("Mean: %s" % (scaler.mean_,))
    print("Std: %s" % (scaler.scale_,))
    print("Calculating scaler time: %s" % (time.time() - t1,))
    pickle.dump(scaler, open(out_path, 'wb'))


def do_scale(x3d, scaler_path, verbose=1):
    """Do scale on the input sequence data. 
    
    Args:
      x3d: ndarray, input sequence data, shape: (n_clips, n_time, n_freq)
      scaler_path: string, path of pre-calculated scaler. 
      verbose: integar, print flag. 
      
    Returns:
      Scaled input sequence data. 
    """
    t1 = time.time()
    scaler = pickle.load(open(scaler_path, 'rb'))
    (n_clips, n_time, n_freq) = x3d.shape
    x2d = x3d.reshape((n_clips * n_time, n_freq))
    x2d_scaled = scaler.transform(x2d)
    x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))
    if verbose == 1:
        print("Scaling time: %s" % (time.time() - t1,))

    return x3d_scaled


### Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    # 提取每个音频的特征，并保存
    parser_ef = subparsers.add_parser('extract_features')
    parser_ef.add_argument('--wav_dir', type=str)
    parser_ef.add_argument('--out_dir', type=str)
    parser_ef.add_argument('--recompute', type=bool)
    # 将所有特征写入 hdf5 文件
    parser_pf = subparsers.add_parser('pack_features')
    parser_pf.add_argument('--intent', type=str)
    parser_pf.add_argument('--fe_dir', type=str)
    parser_pf.add_argument('--out_path', type=str)
    # 计算训练数据的均值方差并保存
    parser_cs = subparsers.add_parser('calculate_scaler')
    parser_cs.add_argument('--hdf5_path', type=str)
    parser_cs.add_argument('--out_path', type=str)

    args = parser.parse_args()
    
    if args.mode == 'extract_features':
        extract_features(wav_dir=args.wav_dir,
                         out_dir=args.out_dir,
                         recompute=args.recompute)
    elif args.mode == 'pack_features':
        pack_features_to_hdf5(intent=args.intent,
                              fe_dir=args.fe_dir,
                              out_path=args.out_path)
    elif args.mode == 'calculate_scaler':
        calculate_scaler(hdf5_path=args.hdf5_path, 
                         out_path=args.out_path)
    else:
        raise Exception("Incorrect argument!")
