# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import argparse
import glob
import time
import os

from keras import backend as K
from keras.models import Model, load_model
from keras.layers.core import Dense, Activation, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply

import config as cfg
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator
import csv_maker as cm


# CNN with Gated linear unit (GLU) block
def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out


def slice1(x):
    return x[:, :, :, 0:64]


def slice2(x):
    return x[:, :, :, 64:128]


def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])


def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])


# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out


# Train model
def train(args):
    _, num_classes, _ = cfg.kinds_of_target(args.intent)  # 选择你的分类目标，目前选项有：gender(男&女)；age(儿童&成人)
    
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x.shape: %s" % (tr_x.shape,))
    print("tr_x.shape: %s" % (tr_y.shape,))
    print("tr_x.shape: %s" % (tr_y[0].shape,))
    n_samples = float(tr_x.shape[0])
    batch_num = 128  # batch size
    if n_samples % batch_num == 0:  #
        steps = int(n_samples/batch_num)
    else:
        steps = int(n_samples/batch_num) + 1
    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    
    # Build model
    (_, n_time, n_freq) = tr_x.shape    # (N, 240, 64)
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel)  # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1)  # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1)  # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1)  # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1)  # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1)  # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1)  # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, "64newMel_240fr.{epoch:02d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = RatioDataGenerator(batch_size=batch_num, type='train')

    # Train
    model.fit_generator(generator=gen.generate([tr_x], [tr_y]),
                        steps_per_epoch=steps,
                        epochs=100,  # Maximum 'epoch' to train
                        verbose=1,
                        callbacks=[save_model],
                        validation_data=(te_x, te_y))


# Run function in mini-batch to save memory. 
def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in range(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all


# Recognize and write probabilites. 
def recognize(args, at_bool):
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    na_list = te_na_list
    
    x = do_scale(x, args.scaler_path, verbose=1)
    
    fusion_at_list = []
    for epoch in range(30, 31):
        t1 = time.time()
        [model_path] = glob.glob(os.path.join(args.model_dir, "*.%02d-0.*.hdf5" % epoch))
        model = load_model(model_path)
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x, batch_size=256)
            fusion_at_list.append(pred)

        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        cm.at_write_prob_mat_to_csv(na_list=na_list,
                                    prob_mat=fusion_at,
                                    out_path=os.path.join(args.out_dir, "at_prob_mat__21-31_11_05.csv.gz"))

    print("Prediction finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    # 训练部分，注意选择正确的 intent
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--intent', type=str)
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    # 识别部分
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--te_hdf5_path', type=str)
    parser_recognize.add_argument('--scaler_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--out_dir', type=str)

    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args, at_bool=True)
    else:
        raise Exception("Incorrect argument!")
