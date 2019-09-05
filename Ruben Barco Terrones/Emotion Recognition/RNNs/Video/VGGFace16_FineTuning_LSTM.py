# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 02:14:50 2019

@author: Rubén
"""

import os
import tensorflow as tf

from keras.utils import Sequence
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, TimeDistributed, Lambda, Input
from keras.callbacks import ModelCheckpoint
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import math
import random

import pandas as pd
import numpy as np

from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

'''Definition of the image sequence generator class'''

class ImDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class, rescale = False, mode = 'train', seed = 127):
        self.df = df
        self.bsz = batch_size
        self.mode = mode
        self.seed = seed
        self.rescale = rescale
        self.nb_class = nb_class
        self.data_path = data_path

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.im_seq_list = self.df['seq'].tolist()  # apply(lambda x: os.path.join(data_path, x))
        if self.mode == 'train':
            random.seed(self.seed)

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))
    
    def load_image(self, im):
        if self.rescale == True:
            return img_to_array(load_img(im)) / 255.
        else:
            return img_to_array(load_img(im))

    def num_to_vec(self, label):
        v = np.zeros(self.nb_class)
        v[label] = 1
        return v

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_seq_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
#        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([np.array([self.load_image(self.data_path+im) for im in seq[1:-2].replace(' ','').replace("'",'').split(',')]) for seq in self.im_seq_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


''' Load the image model and add the recurrent layer '''

def build_RNN_im_VGG(model_path, printable):
    
    trained_model = load_model(model_path)
    feat_model = Model(trained_model.inputs, trained_model.layers[-2].output)
    
    input_im = Input(shape=(None, 224, 224, 3), name='seq_input_im')
    
    feat_counter = 0
    for layer in feat_model.layers:
        layer.trainable = False
        
        if feat_counter == 0:
            ix = input_im
            feat_counter += 1
        else:
            ix = TimeDistributed(Lambda(lambda x: layer(x)))(ix)


    x = LSTM(64, name = 'lstm_10', return_sequences=False)(ix)
#    x = Dropout(0.1)(x)
    out = Dense(4, activation='softmax')(x)
    model = Model(inputs = input_im, outputs = out)
    
    del trained_model
    if printable == 1:
        model.summary()
        
    return model


'''Definition of the variables and the generator and load the data'''

model_path = '/app/Experiment1/weights.000004-0.785-0.549-0.392-2.273.h5'

im_seqs_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/train_seqs_flip.csv')
im_seqs_val   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/val_seqs.csv')

batch_size = 4
nb_class = 4
im_size = 224
n_epochs = 20
steps_per_epoch_train = math.ceil(len(im_seqs_train)/batch_size)
steps_per_epoch_val   = math.ceil(len(im_seqs_val)/batch_size)

seq_train = ImDataSequence(im_seqs_train, '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class, rescale=True, mode = 'train')
seq_val   = ImDataSequence(im_seqs_val,   '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation/',  
                           batch_size = batch_size, nb_class = nb_class, rescale=True, mode = 'val')


'''Definition of the model, the optimizer, the checkpointer and the training '''

RNN_im_model = build_RNN_im_VGG(model_path, 1)
adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
RNN_im_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

filepath = "weights.{epoch:06d}-{acc:.3f}-{loss:.3f}-{val_acc:.3f}-{val_loss:.3f}.h5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto')

A = RNN_im_model.fit_generator(
        generator = seq_train, steps_per_epoch = steps_per_epoch_train, epochs = n_epochs,
        callbacks=[checkpointer], validation_data = seq_val, validation_steps = steps_per_epoch_val)

