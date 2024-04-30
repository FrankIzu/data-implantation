# -*- coding: utf-8 -*-
"""
Spyder Editor

Dr. Francis Onodueze

This is a temporary script file.
"""

# multi-class classification with Keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD, Adam, Adadelta, RMSprop

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import pandas
import pandas as pd
import numpy as np
import pickle
import time

import matplotlib.pyplot as plt
import time

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import get_TSR_data_functions as tsr

# load dataset
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

title = "With1_3-6"
X_train, X_test, X_val, y_train, y_test, y_val = tsr.get_TSR()

# merge all the datasets for the baseline model ONLY
X_train = np.concatenate((X_train, X_test, X_val))
y_train = np.concatenate((y_train, y_test, y_val))

feature_size = X_train.shape[1]

#dataset = dataframe.to_numpy()
# encode class values as integers


# define baseline model
def baseline_model():
    # create model
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_dim=feature_size, activation='relu'))
    model.add(keras.layers.Dense(15, activation = "relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(7, activation='softmax'))
    # Compile model
    model.compile(Adam(lr = 0.01), "categorical_crossentropy", metrics = ["accuracy"])
    model.summary()
    return model

#X_train = X_train.reshape(-1, 1, np.array(X_train).shape[1])

def BiLSTM_model():
    model = keras.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(feature_size, activation='relu', return_sequences=True), input_shape=(None, feature_size)))
    model.add(keras.layers.LSTM(50, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(7, activation='softmax'))
    #model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


start = time.time() 
# =============================================================================
# # simple early stopping
# es_pre = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
# mc_pre = ModelCheckpoint('models/best_model_early_stopping.h5', monitor='val_loss', verbose=1, save_best_only=True)
#      
# model = baseline_model()
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[es_pre, mc_pre], verbose=1)
# 
# saved_model = load_model('models/best_model_early_stopping.h5')
# model = saved_model
# 
# y_pred_class = model.predict_classes(X_test)
# y_pred = model.predict(X_test)
# y_test_class = np.argmax(y_test, axis=1)
# confusion_matrix(y_test_class, y_pred_class)
# 
# #classification report
# print(classification_report(y_test_class, y_pred_class))
# =============================================================================

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1000, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)

estimator.save('models/Baseline_'+title+'_model')

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

stop = time.time()


print(f"Training time: {stop - start}s")
