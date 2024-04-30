#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:40:57 2021

@author: francis
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD, Adam, Adadelta, RMSprop

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# define baseline model
def baseline_model(f, y_c):
    # create model
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_dim=f, activation='relu'))
    #model.add(keras.layers.Dense(15, activation = "relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(y_c, activation='softmax'))
    # Compile model
    model.compile(Adam(lr = 0.005), "categorical_crossentropy", metrics = ["accuracy"])
    model.summary()
    return model
