#!/usr/bin/env python
# coding: utf-8

'''

functions used in train.ipynb

'''

from tensorflow import keras


import matplotlib.pyplot as plt
import numpy as np
import os, sys
import copy
import imageio
import scipy.io as io
import random
import skimage
import pandas as pd

from glob import glob
from random import shuffle
from datetime import datetime

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Activation, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf

from sklearn.utils import class_weight


def resize_to_224(x_train):

    ''' preprocessing '''
    ns, nr, nc, nchan = x_train.shape
    x_train_list = []
    img3_zp = np.zeros((160, 192, 3))

    for jj in range(ns):
        img3 = np.squeeze(x_train[jj,:,:,:]) # img3 : (96, 192, 3)
        # zero-pad img3
        img3_zp[80-48:80+48, :, :] = img3
        img224 = tf.image.resize(img3_zp, (224, 224))
        img224 = 255.0*img224
        x_train_list.append(img224)

    x_train2 = np.stack(x_train_list, axis=0)

    return x_train2


def resize_to_96(x_train):

    ''' preprocessing '''

    ns, nr, nc, nchan = x_train.shape

    x_train_list = []

    for jj in range(ns):
        img3 = np.squeeze(x_train[jj, :,:,:]) # img3 : (96, 128, 3)
        img224 = tf.image.resize(img3, (96, 128))
        img224 = 255.0*img224
        x_train_list.append(img224)

    x_train2 = np.stack(x_train_list, axis=0)

    return x_train2


def gen_cnn_model(gpu_no, model, model_path, val_f, X_train, y_train, X_val, y_val, batch_size, n_epochs):

    ntrain = X_train.shape[0]
    steps_per_epoch = ntrain // batch_size

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H%M%S")

    y_train_encode = to_categorical(y_train)  # to one-hot
    y_val_encode = to_categorical(y_val)  # to one-hot
    dir1 = "model_val_fold=" + str(val_f) + "_" + dt_string
    dir_model = os.path.join(model_path, dir1)

    if not os.path.exists(dir_model):
        print(f'model path = {dir_model}')
        os.makedirs(dir_model)

    filepath = os.path.join(model_path, dir1) + "/CNN_vf" + str(val_f) + "_e{epoch:02d}_valacc{val_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    print(class_weights)

    with K.tf.device(gpu_no):
        history = model.fit(x=X_train, y=y_train_encode, batch_size=batch_size, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
                            class_weight=class_weights, callbacks=callbacks_list, validation_data=(X_val, y_val_encode))

    return history


def grayscale_to_rgb(images, channel_axis=-1):

    images= K.expand_dims(images, axis=channel_axis)
    tiling = [1] * 4    # 4 dimensions: B, H, W, C
    tiling[channel_axis] *= 3
    images = K.tile(images, tiling)

    return images
