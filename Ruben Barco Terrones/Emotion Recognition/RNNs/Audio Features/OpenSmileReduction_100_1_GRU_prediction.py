# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 02:21:34 2019

@author: Rubén
"""

'''

Script to compute the the performance of the model in terms of accuracy for
the training, validation and test sets.

'''


import os
import tensorflow as tf

from keras.utils import Sequence, multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D, TimeDistributed, Lambda, Input, GRU
from keras.callbacks import ModelCheckpoint, Callback
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from keras.layers.merge import concatenate

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from scipy.io import wavfile
import math
import random

import pandas as pd
import numpy as np

from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


'''Definition of the raw-audio generator class'''
class FeatDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class):
        self.df = df
        self.bsz = batch_size
        self.nb_class = nb_class
        self.data_path = data_path

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.csv_seq_list = self.df['seq'].tolist()  # apply(lambda x: os.path.join(data_path, x))

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))
    
    def load_features(self, feat_name):
        data = pd.read_csv(feat_name, header=None).values
        return np.squeeze(data)

    def num_to_vec(self, label):
        v = np.zeros(self.nb_class)
        v[label] = 1
        return v

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.csv_seq_list))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
#        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([np.array([self.load_features(self.data_path+feat_file) for feat_file in seq[1:-2].replace(' ','').replace("'",'').split(',')]) for seq in self.csv_seq_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


''' Load the feat. audio model and add the recurrent layer '''

def build_RNN_feat_audio(model_path, printable):
    
    trained_model = load_model(model_path)
    feat_model = Model(trained_model.inputs, trained_model.layers[-2].output)
    
    input_feat = Input(shape=(None, 5266,), name='seq_input_raw')
    
    feat_counter = 0
    for layer in feat_model.layers:
        layer.trainable = False
        
        if feat_counter == 0:
            fx = input_feat
            feat_counter += 1
        else:
            fx = TimeDistributed(Lambda(lambda x: layer(x)))(fx)


    x = GRU(64, name = 'gru_10', return_sequences=False)(fx)
#    x = Dropout(0.1)(x)
    out = Dense(4, activation='softmax')(x)
    model = Model(inputs = input_feat, outputs = out)
    
    del trained_model
    if printable == 1:
        model.summary()
        
    return model


''' Definition of the architecture and loading the model '''
print('Defining architecture and loading weights...')
model_path = '/app/Experiment1_100_1/weights.000026-0.403-1.253-0.428-1.228.h5'
model = build_RNN_feat_audio(model_path, 0)

weights = '/app/Experiment1_100_1_GRU/weights.000005-0.484-1.141-0.492-1.145.h5'
adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
model.load_weights(weights)
print('Model loaded...')


'''Definition of the variables and the generator and load the data'''
audio_seqs_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train_seqs.csv')
audio_seqs_val   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/val_seqs.csv')
audio_seqs_test  = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test_seqs.csv')

batch_size = 64
nb_class = 4
n_audio_vector = 5266

steps_per_epoch_train = math.ceil(len(audio_seqs_train)/batch_size)
steps_per_epoch_val   = math.ceil(len(audio_seqs_val)/batch_size)
steps_per_epoch_test  = math.ceil(len(audio_seqs_test)/batch_size)

seq_train = FeatDataSequence(audio_seqs_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class)
seq_val   = FeatDataSequence(audio_seqs_val,   '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation/',  
                           batch_size = batch_size, nb_class = nb_class)
seq_test  = FeatDataSequence(audio_seqs_test,  '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test/',  
                           batch_size = batch_size, nb_class = nb_class)


''' Computing the accuracies '''
print('Computing training accuracy...')
evaluations = model.evaluate_generator(generator = seq_train)
print('Training accuracy: {}'.format(evaluations[1]*100))

print('Computing validation accuracy...')
evaluations = model.evaluate_generator(generator = seq_val)
print('Validation accuracy: {}'.format(evaluations[1]*100))

print('Computing test accuracy...')
evaluations = model.evaluate_generator(generator = seq_test)
print('Test accuracy: {}'.format(evaluations[1]*100))

