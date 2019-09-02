# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 02:21:34 2019

@author: Rubén
"""

'''

Script to compute the the performance of the model in terms of accuracy for
the training, validation and test sets at utterance level and for each label.

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
class AudioDataSequence(Sequence):
    
    def __init__(self, df, data_path, batch_size, nb_class):
        self.df = df
        self.bsz = batch_size
        self.nb_class = nb_class
        self.data_path = data_path

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.audio_seq_list = self.df['seq'].tolist()  #.apply(lambda x: os.path.join(data_path, x))

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
#ax = GRU(64, name = 'gru_10', return_sequences=False)(ax)
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

x = GRU(64, name = 'gru_10', return_sequences=False)(x)
#    x = Dropout(0.1)(x)
out = Dense(4, activation='softmax')(x)
model = Model(inputs = input_raw, outputs = out)


weights = '/app/Experiment1_100_RNN_FINAL_GRU/weights.000026-0.582-0.970-0.499-1.182.h5'
adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
model.load_weights(weights)
print('Model loaded...')


'''Definition of the variables and the generator and load the data'''
audio_seqs_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train_seqs.csv')
audio_seqs_val   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/val_seqs.csv')
audio_seqs_test  = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test_seqs.csv')

batch_size = 16
nb_class = 4
n_audio_vector = 1600
n_epochs = 1000
steps_per_epoch_train = math.ceil(len(audio_seqs_train)/batch_size)
steps_per_epoch_val   = math.ceil(len(audio_seqs_val)/batch_size)
steps_per_epoch_test  = math.ceil(len(audio_seqs_test)/batch_size)

seq_train = AudioDataSequence(audio_seqs_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/',  
                           batch_size = batch_size, nb_class = nb_class)
seq_val   = AudioDataSequence(audio_seqs_val,   '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/',  
                           batch_size = batch_size, nb_class = nb_class)
seq_test  = AudioDataSequence(audio_seqs_test,  '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test/',  
                           batch_size = batch_size, nb_class = nb_class)


''' Computing the accuracies '''
txt_path = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/'

print('Computing training accuracy...') ### Change the following two lines for train/va/test
predictions = model.predict_generator(generator = seq_train)
mode = 'train'
with open(txt_path + mode + '_hap_long.txt', 'r') as f:
    hap_long = [int(line.rstrip('\n')) for line in f]
    
with open(txt_path + mode + '_sad_long.txt', 'r') as f:
    sad_long = [int(line.rstrip('\n')) for line in f]
    
with open(txt_path + mode + '_neu_long.txt', 'r') as f:
    neu_long = [int(line.rstrip('\n')) for line in f]
    
with open(txt_path + mode + '_ang_long.txt', 'r') as f:
    ang_long = [int(line.rstrip('\n')) for line in f]


#true_labels_local = []
#for long in hap_long:
#    for j in range(long):
#        true_labels_local.append(0)
#for long in sad_long:
#    for j in range(long):
#        true_labels_local.append(1)
#for long in neu_long:
#    for j in range(long):
#        true_labels_local.append(2)
#for long in ang_long:
#    for j in range(long):
#        true_labels_local.append(3)
#
#correct_local = 0
#for i in range(len(predictions)):
#    if np.where(predictions[i] == max(predictions[i]))[0][0] == true_labels_local[i]:
#        correct_local += 1
   
cont = 0
true_labels = [] 
pred_labels_maj  = []
pred_labels_mean = []
for long_utt in hap_long:
    true_labels.append(0)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt
    
for long_utt in sad_long:
    true_labels.append(1)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt

for long_utt in neu_long:
    true_labels.append(2)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt

for long_utt in ang_long:
    true_labels.append(3)
    utt_preds = predictions[cont:cont+long_utt]
    
    ### Majority Vote
    votes = []
    for i in range(len(utt_preds)):
        votes.append(np.where(utt_preds[i] == max(utt_preds[i]))[0][0])
    pred_labels_maj.append(max(set(votes), key = votes.count))
    
    ### Mean
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for i in range(len(utt_preds)):
        label_1.append(utt_preds[i][0])
        label_2.append(utt_preds[i][1])
        label_3.append(utt_preds[i][2])
        label_4.append(utt_preds[i][3])
   
    mean_label_1 = np.mean(label_1)
    mean_label_2 = np.mean(label_2)
    mean_label_3 = np.mean(label_3)
    mean_label_4 = np.mean(label_4)
    aux = [mean_label_1,mean_label_2,mean_label_3,mean_label_4]
    pred_labels_mean.append(aux.index(max(aux)))
    
    cont += long_utt
    
print('Set: {}'.format(mode))
print('Number of predictions: {}'.format(len(predictions)))
print('Final value of cont:   {}'.format(cont))
print('Number of utterances pred:  {}'.format(len(pred_labels_mean)))
print('Number of utterances true:  {}'.format(len(true_labels)))

correct_mean = 0
for i in range(len(pred_labels_mean)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(pred_labels_maj)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1
        
#print('Accuracy of model at local level: {} %'.format(100*(correct_local/len(predictions))))
print('Accuracy of model at utterance level (majority vote): {} %'.format(100*(correct_maj/len(pred_labels_maj))))
print('Accuracy of model at utterance level          (mean): {} %'.format(100*(correct_mean/len(pred_labels_mean))))
print('Accuracy of model at utterance level:')

correct_mean = 0
for i in range(0,len(hap_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(0,len(hap_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('hap (majority vote): {} %'.format(100*(correct_maj/len(hap_long))))
print('hap          (mean): {} %'.format(100*(correct_mean/len(hap_long))))

correct_mean = 0
for i in range(len(hap_long),len(hap_long)+len(sad_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(hap_long),len(hap_long)+len(sad_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('sad (majority vote): {} %'.format(100*(correct_maj/len(sad_long))))
print('sad          (mean): {} %'.format(100*(correct_mean/len(sad_long))))

correct_mean = 0
for i in range(len(hap_long)+len(sad_long),len(hap_long)+len(sad_long)+len(neu_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(hap_long)+len(sad_long),len(hap_long)+len(sad_long)+len(neu_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('neu (majority vote): {} %'.format(100*(correct_maj/len(neu_long))))
print('neu          (mean): {} %'.format(100*(correct_mean/len(neu_long))))

correct_mean = 0
for i in range(len(hap_long)+len(sad_long)+len(neu_long),len(hap_long)+len(sad_long)+len(neu_long)+len(ang_long)):
    if pred_labels_mean[i] == true_labels[i]:
        correct_mean += 1
        
correct_maj = 0
for i in range(len(hap_long)+len(sad_long)+len(neu_long),len(hap_long)+len(sad_long)+len(neu_long)+len(ang_long)):
    if pred_labels_maj[i] == true_labels[i]:
        correct_maj += 1

print('ang (majority vote): {} %'.format(100*(correct_maj/len(ang_long))))
print('ang          (mean): {} %'.format(100*(correct_mean/len(ang_long))))


