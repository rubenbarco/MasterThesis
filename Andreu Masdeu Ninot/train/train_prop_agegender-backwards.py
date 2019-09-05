#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import pandas as pd
import numpy as np

from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Conv2D, Dense

from leaky_unit import LeakyUnit, ProportionalAddition
import os
from sklearn.model_selection import train_test_split
from PIL import Image

from build_prop_model_backwards import build_model_prop
from keras.callbacks import EarlyStopping


import pickle


DIR_DATA = 'DATA/UTKFace/UTKFace/'
files = os.listdir(DIR_DATA)


images = []
age = []
genders = []
ethnicity = []

for file in files:
    try:
        ethnicity.append(float(file.split('_')[2]))
        images.append(np.asarray(Image.open(DIR_DATA+file))/255)
        age.append(float(file.split('_')[0]))
        genders.append(float(file.split('_')[1]))
        
    except:
        print("Missing label for file {}".format(file))


# In[4]:


split = train_test_split(images, age, genders, ethnicity, test_size=0.2, random_state=42)

(trainX, testX, ages_trainY, ages_testY, gender_trainY,
     gender_testY, ethnicity_trainY, ethnicity_testY) = split

del images



tasks_labels = {'age':[ages_trainY, ages_testY],
                   'gender': [gender_trainY, gender_testY],
                   'ethnicity': [ethnicity_trainY, ethnicity_testY]}

acc = []

for tasks in [['age', 'gender']]:

    for i in range(3):
        
        train_labels = {tasks[0]: np.array(tasks_labels[tasks[0]][0]),
                  tasks[1]: np.array(tasks_labels[tasks[1]][0])
                    }

        test_labels = {tasks[0]: np.array(tasks_labels[tasks[0]][1]),
                  tasks[1]: np.array(tasks_labels[tasks[1]][1])
                    }

        model = build_model_prop(tasks)

        es = EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=1)

        history = model.fit(np.array(trainX), train_labels, epochs=20, 
                  batch_size=64, validation_data=(np.array(testX), test_labels), verbose=2, callbacks=[es])

        current_acc = [max(history.history['val_{}_acc'.format(tasks[0])]), max(history.history['val_{}_acc'.format(tasks[1])])]

        if i == 0 or (sum(current_acc) > np.array(acc).sum(axis=1)).all():
            
            model.save('baseline_{}{}b.h5'.format(tasks[0], tasks[1]))

            weights = {}

            for layer in model.layers:

                weights[layer.name] = layer.get_weights()

        acc.append(current_acc)



    with open("weights_baseline{}{}b.pkl".format(tasks[0], tasks[1]), "wb") as handle:
        pickle.dump(weights, handle)
    

# # Proportional addition
for tasks in [['age', 'gender']]:
    
    train_labels = {tasks[0]: np.array(tasks_labels[tasks[0]][0]),
                  tasks[1]: np.array(tasks_labels[tasks[1]][0])
                    }

    test_labels = {tasks[0]: np.array(tasks_labels[tasks[0]][1]),
                  tasks[1]: np.array(tasks_labels[tasks[1]][1])
                    }
    
    weights = pickle.load(open("weights_baseline{}{}b.pkl".format(tasks[0], tasks[1]), "rb"))
    
    for nb_prop_layers in range(1,6):
        acc = []

        prev_weights = weights

        for i in range(3):

            model = build_model_prop(tasks, nb_prop_layers)

            for layer in model.layers:

                try:
                    layer.set_weights(prev_weights[layer.name])
                except:
                    print("Layer {} weights not available".format(layer.name))


            es = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)

            history = model.fit(np.array(trainX), train_labels, epochs=20, 
                      batch_size=64, validation_data=(np.array(testX), test_labels), verbose=2, callbacks=[es])

            current_acc = [max(history.history['val_{}_acc'.format(tasks[0])]), max(history.history['val_{}_acc'.format(tasks[1])])]

            if i == 0 or (sum(current_acc) > np.array(acc).sum(axis=1)).all():
                
                model.save('prop{}_{}{}b.h5'.format(nb_prop_layers, tasks[0], tasks[1]))

                weights = {}

                for layer in model.layers:

                    weights[layer.name] = layer.get_weights()

            acc.append(current_acc)

            with open("history{}{}{}{}PropAddb.pkl".format(i, tasks[0], tasks[1], nb_prop_layers), "wb") as handle:
                pickle.dump(history.history, handle)


