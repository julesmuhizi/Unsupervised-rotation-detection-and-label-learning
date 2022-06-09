import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from random import shuffle
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from distilled_model import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yaml
import math

def yaml_load(config):
    with open(config, 'r') as stream:
        param = yaml.safe_load(stream)
    return param

def main(args):

    # get yaml config of model
    config = yaml_load(args.config)


    # generate training data
    img = np.load('02_scan_x256_y256_raw.npy')
    img = img.astype('float32')
    img = np.transpose(img,(2,3,0,1))
    data_r = np.copy(img)
    data_r[data_r>1e3]=1e3
    min_ = np.min(data_r)
    max_ = np.max(data_r)
    data_r = 1.0*(data_r-min_)/(max_-min_)
    data_r = data_r.reshape(-1,1,124,124)
    data_r_cut = data_r[:,:,2:122,2:122]
    data_r_cut = data_r_cut.reshape(256,256,120,120)
    data_r_cut = np.rot90(data_r_cut)
    X = data_r_cut.reshape(-1, 120,120)

    # generate outputs/targets
    # dataset_h5 = h5py.File('unbinned_results.h5','r+')
    # rots = np.array(dataset_h5['rotation'])
    # scal = np.array(dataset_h5['scale'])
    # y = np.concatenate((rots, scal), axis=1)
    y = np.load('unbinned_results.npy')
    sc = StandardScaler()
    y = sc.fit_transform(y)


    # define model
    print(config['name'])
    model = create_small_float_model() if 'small_float' in config['name'] \
        else create_distilled_model() if 'float' in config['name'] \
        else create_quantized_model(config['precision']) if 'quant' in config['name']  \
        else create_small_quantized_model(config['precision']) if 'small' in config['name'] \
        else create_extra_small_quantized_model(config['precision']) if 'extra' in config['name'] \
        else create_mlp_avg_pool(config['precision']) if 'mlp_average_pool' in config['name']  \
        else create_mlp_max_pool(config['precision']) if 'mlp_max_pool' in config['name']  \
        else create_mlp(config['precision']) if 'mlp' in config['name']  \
        else None

    optimizer = 'adam'
    loss = 'mse'
    stopping = EarlyStopping(monitor='val_loss',
                                 patience = 10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                      mode='min', verbose=1, epsilon=0.001,
                                      cooldown=4, min_lr=1e-5)
    callbacks=[
            stopping,
            reduce_lr,
        ]
    model.compile(optimizer=optimizer,
                    loss=loss,)

    model.summary()
    history = model.fit(X,y,
                    epochs=100,
                    batch_size = 32,
                    shuffle=True,
                    validation_split = 0.2,
                    callbacks=callbacks)

    model.save('{}/model.h5'.format(config['model_save_dir']))

    # plot history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/model.png'.format(config['model_save_dir']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='float_model.yml', help="specify yaml config")

    args = parser.parse_args()

    main(args)