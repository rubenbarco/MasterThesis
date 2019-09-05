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
from contextlib import redirect_stdout

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


'''Definition of the image-audio generator class'''

class ImAudioFeatSequence(Sequence):
    
    def __init__(self, df_audio, df_features, df_im, data_path_audio, data_path_features, data_path_im, batch_size, nb_class, rescale = False):
        self.df_audio           = df_audio
        self.df_features        = df_features
        self.df_im              = df_im
        self.bsz                = batch_size
        self.nb_class           = nb_class
        self.rescale            = rescale
        self.data_path_audio    = data_path_audio
        self.data_path_features = data_path_features
        self.data_path_im       = data_path_im

        # Take labels and a list of image and audio locations in memory
        self.labels         = self.df_im['label'].values
        self.audio_seq_list = self.df_audio['seq'].tolist()
        self.feat_seq_list  = self.df_features['seq'].tolist()
        self.im_seq_list    = self.df_im['seq'].tolist()

    def __len__(self):
        return int(math.ceil(len(self.df_audio) / float(self.bsz)))

    def load_audio(self, audio):
        _, data = wavfile.read(audio)
        return data

    def load_features(self, feat_name):
        data = pd.read_csv(feat_name, header=None).values
        return np.squeeze(data)
    
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
        self.indexes = range(len(self.audio_seq_list))
    
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array([self.num_to_vec(x) for x in self.labels[idx * self.bsz: (idx + 1) * self.bsz]])
    
    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return [np.array([np.array([self.load_image(self.data_path_im+im) for im in seq[1:-2].replace(' ','').replace("'",'').split(',')]) for seq in self.im_seq_list[idx * self.bsz: (1 + idx) * self.bsz]]), np.array([np.array([np.expand_dims(self.load_audio(self.data_path_audio+audio), axis=1) for audio in seq[1:-2].replace(' ','').replace("'",'').split(',')]) for seq in self.audio_seq_list[idx * self.bsz: (1 + idx) * self.bsz]]), np.array([np.array([self.load_features(self.data_path_features+feat_file) for feat_file in seq[1:-2].replace(' ','').replace("'",'').split(',')]) for seq in self.feat_seq_list[idx * self.bsz: (1 + idx) * self.bsz]])]

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


''' Load the image model and add the recurrent layer '''
    
def build_RNN_VGGAudioFeat(model_path, printable):
    
    trained_model = load_model(model_path)
    aux_model = Model(trained_model.inputs, trained_model.layers[-2].output)
    
    input_im = Input(shape=(None, 224, 224, 3))
    input_raw = Input(shape=(None, 1600, 1))
    input_feat = Input(shape=(None, 5266,))
    
    model_input = [input_im, input_raw, input_feat]
    
    for layer in aux_model.layers:
        layer.trainable = False
        
        if 'input_1' == layer.name:
            ix = input_im
        elif 'conv1d_1_input' == layer.name:
            ax = input_raw
        elif 'dense_1_input' == layer.name:
            fx = input_feat           
        elif 'conv1_1' == layer.name or 'conv1_2' == layer.name or 'pool1' == layer.name or 'conv2_1' == layer.name or 'conv2_2' == layer.name or 'pool2' == layer.name or 'conv3_1' == layer.name or 'conv3_2' == layer.name or 'conv3_3' == layer.name or 'pool3' == layer.name or 'conv4_1' == layer.name or 'conv4_2' == layer.name or 'conv4_3' == layer.name or 'pool4' == layer.name or 'conv5_1' == layer.name or 'conv5_2' == layer.name or 'conv5_3' == layer.name or 'pool5' == layer.name or 'flatten' == layer.name or 'fc6' == layer.name or 'fc7' == layer.name or 'dense_1' == layer.name:
            ix = TimeDistributed(Lambda(lambda x: layer(x)))(ix)
        elif 'conv1d_1' == layer.name or 'max_pooling1d_1' == layer.name or 'conv1d_2' == layer.name or 'max_pooling1d_2' == layer.name or 'conv1d_3' == layer.name or 'max_pooling1d_3' == layer.name or 'conv1d_4' == layer.name or 'global_average_pooling1d_1' == layer.name:
            ax = TimeDistributed(Lambda(lambda x: layer(x)))(ax)
        elif 'dense_aux' == layer.name or 'dense_2' == layer.name:
            fx = TimeDistributed(Lambda(lambda x: layer(x)))(fx)
        elif 'concatenate' in layer.name:
            x = concatenate([ix, ax, fx])
        elif 'fc_class_1' == layer.name:
            x = TimeDistributed(Lambda(lambda x: layer(x)))(x)
            

    x = GRU(64, name = 'gru_10', return_sequences=False)(x)
    out = Dense(4, activation='softmax')(x)
    model = Model(inputs = model_input, outputs = out)
    
    del trained_model
    if printable == 1:
        model.summary()
        
    return model


''' Definition of the architecture and loading the model '''
print('Defining architecture and loading weights...')
model_path = '/app/Fusion_faceraw_features_100/weights.000009-0.865-0.348-0.437-2.212.h5'
model = build_RNN_VGGAudioFeat(model_path, 0)

weights = '/app/Fusion_VGGAudioFeat_100_GRU/weights.000013-0.673-0.830-0.407-1.521.h5'
adam = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
model.load_weights(weights)
print('Model loaded...')


'''Definition of the variables and the generator and load the data'''
im_seqs_train    = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/train_seqs_flip.csv')
im_seqs_val      = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/val_seqs.csv')
im_seqs_test     = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/test_seqs.csv')
audio_seqs_train = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train_seqs_flip.csv')
audio_seqs_val   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/val_seqs.csv')
audio_seqs_test  = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test_seqs.csv')
feat_seqs_train  = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train_seqs_flip.csv')
feat_seqs_val    = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/val_seqs.csv')
feat_seqs_test   = pd.read_csv('/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test_seqs.csv')

batch_size = 4
nb_class = 4
im_size = 224
n_feature_vector = 5266
n_audio_vector = 1600
n_epochs = 20
steps_per_epoch_train = math.ceil(len(im_seqs_train)/batch_size)
steps_per_epoch_val   = math.ceil(len(im_seqs_val)/batch_size)
steps_per_epoch_test  = math.ceil(len(im_seqs_test)/batch_size)

seq_train = ImAudioFeatSequence(audio_seqs_train, feat_seqs_train, im_seqs_train, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/train/', 
                                '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/train/', '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/train/',  
                                batch_size = batch_size, nb_class = nb_class, rescale=True)
seq_val   = ImAudioFeatSequence(audio_seqs_val, feat_seqs_val, im_seqs_val, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/validation/', 
                                '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/validation/', '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/validation/',  
                                batch_size = batch_size, nb_class = nb_class, rescale=True)
seq_test  = ImAudioFeatSequence(audio_seqs_test, feat_seqs_test, im_seqs_test, '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_training_100/4Labels/test/', 
                                '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/test/', '/app/data/IEMOCAP_dat1/IEMOCAP_faces_2_training/4Labels/test/',  
                                batch_size = batch_size, nb_class = nb_class, rescale=True)


''' Computing the accuracies '''
txt_path = '/app/data/IEMOCAP_dat1/IEMOCAP_audio_2_features_100/4Labels/'

print('Computing training accuracy...') ### Change the following two lines for train/val/test
predictions = model.predict_generator(generator = seq_test)
mode = 'test'


with open(txt_path + mode + '_hap_long.txt', 'r') as f:
    hap_long = [int(line.rstrip('\n')) for line in f]
    
with open(txt_path + mode + '_sad_long.txt', 'r') as f:
    sad_long = [int(line.rstrip('\n')) for line in f]
    
with open(txt_path + mode + '_neu_long.txt', 'r') as f:
    neu_long = [int(line.rstrip('\n')) for line in f]
    
with open(txt_path + mode + '_ang_long.txt', 'r') as f:
    ang_long = [int(line.rstrip('\n')) for line in f]

   
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
  
# We have to repeat it if we are evaluating the training set because we have also the fliped faces    
if mode == 'train':
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


