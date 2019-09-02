# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 00:39:44 2019

@author: Rubén
"""

'''
Final version of the architecture for thw raw audio network + RNN.
We are going to freeze all the network of raw audio and add a LSTM/GRU layer
with 64 hidden units without any dropout, and the final FC layer (after the 
global average pooling).
'''


''' Libraries '''

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
class AudioDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class, mode = 'train', seed = 127):
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


''' Load the raw audio model and add the recurrent layer '''

def build_RNN_raw_audio(model_path, printable):
    
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

    x = GRU(64, name = 'gru_10', return_sequences=False)(x)
#    x = Dropout(0.1)(x)
    out = Dense(4, activation='softmax')(x)
    model = Model(inputs = input_raw, outputs = out)
    
    del trained_model
    if printable == 1:
        model.summary()
        
    return model


'''Definition of the variables and the generator and load the data'''

model_path = '/app/Experiment1_100/weights.000026-0.430-1.224-0.418-1.248.h5'

audio_seqs_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train_seqs.csv')
audio_seqs_val = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/val_seqs.csv')

batch_size = 16
nb_class = 4
n_audio_vector = 1600
n_epochs = 1000
steps_per_epoch_train = math.ceil(len(audio_seqs_train)/batch_size)
steps_per_epoch_val = math.ceil(len(audio_seqs_val)/batch_size)

seq_train = AudioDataSequence(audio_seqs_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'train')
seq_val = AudioDataSequence(audio_seqs_val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/',  
                         batch_size = batch_size, nb_class = nb_class, mode = 'val')


'''Definition of the model, the optimizer, the checkpointer and the training '''

RNN_raw_model = build_RNN_raw_audio(model_path, 1)
adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
RNN_raw_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

filepath = "weights.{epoch:06d}-{acc:.3f}-{loss:.3f}-{val_acc:.3f}-{val_loss:.3f}.h5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto')

A = RNN_raw_model.fit_generator(
        generator = seq_train, steps_per_epoch = steps_per_epoch_train, epochs = n_epochs,
        callbacks=[checkpointer], validation_data = seq_val, validation_steps = steps_per_epoch_val)



