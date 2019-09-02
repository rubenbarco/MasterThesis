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
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D, TimeDistributed, Lambda, Input, LSTM
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
class AudioDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class, mode, seed = 127):
        self.df = df
        self.bsz = batch_size
        self.mode = mode
        self.seed = seed
        self.nb_class = nb_class
        self.data_path = data_path

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.audio_seq_list = self.df['seq'].tolist()  #.apply(lambda x: os.path.join(data_path, x))
        if self.mode == 'train':
            random.seed(self.seed)

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))
    
    def load_audio(self, audio):
        _, data = wavfile.read(audio)
        return data

    def num_to_vec(self, label):
        v = np.zeros(self.nb_class)
        v[label] = 1
        return v

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.audio_seq_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
#        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])
        
    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([np.array([np.expand_dims(self.load_audio(self.data_path+audio), axis=1) for audio in seq[1:-2].replace(' ','').replace("'",'').split(',')]) for seq in self.audio_seq_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


''' Definition of the architecture and loading the model '''
print('Defining architecture and loading weights...')
#input_raw = Input(shape=(None, 1064, 1), name='seq_input_raw')
#ax = input_raw
#ax = TimeDistributed(Lambda(Conv1D(16, 9, strides=1, activation='relu')))(ax)
#ax = TimeDistributed(Lambda(MaxPooling1D(2)))(ax)
#ax = TimeDistributed(Lambda(Conv1D(32, 9, strides=1, activation='relu')))(ax)
#ax = TimeDistributed(Lambda(MaxPooling1D(2)))(ax)
#ax = TimeDistributed(Lambda(Conv1D(64, 9, strides=1, activation='relu')))(ax)
#ax = TimeDistributed(Lambda(MaxPooling1D(2)))(ax)
#ax = TimeDistributed(Lambda(Conv1D(128, 9, strides=1,activation='relu')))(ax)
#ax = TimeDistributed(Lambda(GlobalAveragePooling1D()))(ax)
#ax = LSTM(64, name = 'lstm_10', return_sequences=False)(ax)
#out = Dense(4, activation='softmax')(ax)
#model = Model(inputs = input_raw, outputs = out)

model_path = '/app/Experiment1_100/weights.000026-0.430-1.224-0.418-1.248.h5'
trained_model = load_model(model_path)
audio_model = Model(trained_model.inputs, trained_model.layers[-3].output)

input_raw = Input(shape=(None, 1600, 1), name='seq_input_raw')

audio_counter = 0
for layer in audio_model.layers:
    layer.trainable = False
    
    if 'global_average' not in layer.name:
        if audio_counter == 0:
            ax = input_raw
            audio_counter += 1
        else:
            ax = TimeDistributed(Lambda(lambda x: layer(x)))(ax)
    else:
        layer.trainable = False
        x = TimeDistributed(Lambda(lambda x: layer(x)))(ax)

x = LSTM(64, name = 'lstm_10', return_sequences=False)(x)
#    x = Dropout(0.1)(x)
out = Dense(4, activation='softmax')(x)
model = Model(inputs = input_raw, outputs = out)


weights = '/app/Experiment1_100_RNN_FINAL_LSTM/weights.000026-0.585-0.964-0.496-1.187.h5'
adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
model.load_weights(weights)
#pesos = model.get_weights()
#np.save('/app/pesos_loaded.npy', pesos)
print('Model loaded...')


'''Definition of the variables and the generator and load the data'''
audio_seqs_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train_seqs.csv')
audio_seqs_val   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/val_seqs.csv')
audio_seqs_test  = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test_seqs.csv')

batch_size = 16
nb_class = 4
n_audio_vector = 1600

steps_per_epoch_train = math.ceil(len(audio_seqs_train)/batch_size)
steps_per_epoch_val   = math.ceil(len(audio_seqs_val)/batch_size)
steps_per_epoch_test  = math.ceil(len(audio_seqs_test)/batch_size)

seq_train = AudioDataSequence(audio_seqs_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'train')
seq_val   = AudioDataSequence(audio_seqs_val,   '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'val')
seq_test  = AudioDataSequence(audio_seqs_test,  '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'val')


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

