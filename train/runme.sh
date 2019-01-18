#!/bin/bash
export CUDA_VISIBLE_DEVICES='4'

# You can to modify to your own workspace.
WORKSPACE=`pwd`

# You need to modify to your dataset path
TEST_WAV_DIR="data/test"
TRAIN_WAV_DIR="data/train"

# Extract features
python prepare_data.py extract_features --wav_dir=$TEST_WAV_DIR --out_dir=$WORKSPACE"/features/test" --recompute=True
python prepare_data.py extract_features --wav_dir=$TRAIN_WAV_DIR --out_dir=$WORKSPACE"/features/train" --recompute=True

# Pack features
python prepare_data.py pack_features --intent=age --fe_dir=$WORKSPACE"/features/test" --out_path=$WORKSPACE"/packed_features/testing.h5"
python prepare_data.py pack_features --intent=age --fe_dir=$WORKSPACE"/features/train" --out_path=$WORKSPACE"/packed_features/training.h5"

# Calculate scaler
python prepare_data.py calculate_scaler --hdf5_path=$WORKSPACE"/packed_features/training.h5" --out_path=$WORKSPACE"/scalers/training.scaler"

# Train SED
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_sed.py train --intent=age --tr_hdf5_path=$WORKSPACE"/packed_features/training.h5" --te_hdf5_path=$WORKSPACE"/packed_features/testing.h5" --scaler_path=$WORKSPACE"/scalers/training.scaler" --out_model_dir=$WORKSPACE"/models"

# Recognize SED
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_sed.py recognize --te_hdf5_path=$WORKSPACE"/packed_features/testing.h5" --scaler_path=$WORKSPACE"/scalers/training.scaler" --model_dir=$WORKSPACE"/models" --out_dir=$WORKSPACE"/preds"
