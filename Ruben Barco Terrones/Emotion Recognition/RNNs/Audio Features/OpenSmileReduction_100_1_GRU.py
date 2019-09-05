import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, Input, GRU, TimeDistributed, Lambda
from keras.optimizers import Adam
from keras.utils import Sequence
from keras import backend as K

import pandas as pd
import numpy as np
import math
import random
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

'''Definition of the feat-audio sequence generator class'''

class FeatDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class, mode = 'train', seed = 127):
        self.df = df
        self.bsz = batch_size
        self.mode = mode
        self.seed = seed
        self.nb_class = nb_class
        self.data_path = data_path

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.csv_seq_list = self.df['seq'].tolist()  # apply(lambda x: os.path.join(data_path, x))
        if self.mode == 'train':
            random.seed(self.seed)

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
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

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


'''Definition of the variables and the generator and load the data'''

model_path = '/app/Experiment1_100_1/weights.000026-0.403-1.253-0.428-1.228.h5'

audio_seqs_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train_seqs.csv')
audio_seqs_val   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/val_seqs.csv')

batch_size = 64
nb_class = 4
n_audio_vector = 5266
n_epochs = 1000
steps_per_epoch_train = math.ceil(len(audio_seqs_train)/batch_size)
steps_per_epoch_val   = math.ceil(len(audio_seqs_val)/batch_size)

seq_train = FeatDataSequence(audio_seqs_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'train')
seq_val   = FeatDataSequence(audio_seqs_val,   '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation/',  
                           batch_size = batch_size, nb_class = nb_class, mode = 'val')


'''Definition of the model, the optimizer, the checkpointer and the training '''

RNN_feat_model = build_RNN_feat_audio(model_path, 1)
adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
RNN_feat_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

filepath = "weights.{epoch:06d}-{acc:.3f}-{loss:.3f}-{val_acc:.3f}-{val_loss:.3f}.h5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto')

A = RNN_feat_model.fit_generator(
        generator = seq_train, steps_per_epoch = steps_per_epoch_train, epochs = n_epochs,
        callbacks=[checkpointer], validation_data = seq_val, validation_steps = steps_per_epoch_val)


