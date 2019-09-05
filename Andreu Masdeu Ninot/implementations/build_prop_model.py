import keras
import pandas as pd
import numpy as np

from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Conv2D, Dense

from leaky_unit import LeakyUnit, ProportionalAddition
import os
from sklearn.model_selection import train_test_split
from PIL import Image

tasks_losses = {'age':'mse', 'gender':'binary_crossentropy', 
                'ethnicity': 'sparse_categorical_crossentropy'}

tasks_neurons_activations = {'age':[1, 'linear'], 'gender':[1, 'sigmoid'], 
                'ethnicity': [5, 'softmax']}


def build_model_prop(tasks, nb_prop_layers=0):
    
    DIM_IMG = 200

    image = Input(shape=(DIM_IMG, DIM_IMG, 3,))

    x1 = Conv2D(16, 3, activation='relu', name='conv1x1')(image)
    x1 = MaxPooling2D(2,name='pool1x1')(x1)

    x2 = Conv2D(16, 3, activation='relu', name='conv1x2')(image)
    x2 = MaxPooling2D(2, name='pool1x2')(x2)
    
    if nb_prop_layers > 4:
        
        x1, x2 = ProportionalAddition(n_tasks=2)([x1, x2])

    # block 1

    x1 = Conv2D(16, 3, activation='relu', name='conv2x1')(x1)
    x1 = MaxPooling2D(2, name='pool2x1')(x1)

    x2 = Conv2D(16, 3, activation='relu', name='conv2x2')(x2)
    x2 = MaxPooling2D(2, name='pool2x2')(x2)

    if nb_prop_layers > 3:
        
        x1, x2 = ProportionalAddition(n_tasks=2)([x1, x2])
        
    # block 2

    x1 = Conv2D(32, 3, activation='relu', name='conv3x1')(x1)
    x1 = MaxPooling2D(2, name='pool3x1')(x1)

    x2 = Conv2D(32, 3, activation='relu', name='conv3x2')(x2)
    x2 = MaxPooling2D(2, name='pool3x2')(x2)

    if nb_prop_layers > 2:
        
        x1, x2 = ProportionalAddition(n_tasks=2)([x1, x2])
    # block 3

    x1 = Conv2D(32, 3, activation='relu', name='conv4x1')(x1)
    x1 = MaxPooling2D(2, name='pool4x1')(x1)

    x2 = Conv2D(32, 3, activation='relu', name='conv4x2')(x2)
    x2 = MaxPooling2D(2, name='pool4x2')(x2)
    
    if nb_prop_layers > 1:
        
        x1, x2 = ProportionalAddition(n_tasks=2)([x1, x2])

    # block 4

    x1 = Conv2D(64, 3, activation='relu', name='conv5x1')(x1)
    x1 = MaxPooling2D(2, name='pool5x1')(x1)

    x2 = Conv2D(64, 3, activation='relu', name='conv5x2')(x2)
    x2 = MaxPooling2D(2, name='pool5x2')(x2)
    
    if nb_prop_layers > 0:
        
        x1, x2 = ProportionalAddition(n_tasks=2)([x1, x2])

    
    x1 = Flatten(name='flattenx1')(x1)
    x2 = Flatten(name='flattenx2')(x2)

    # dense block 1

    x1 = Dense(128, activation='relu', name='dense1x1')(x1)

    x2 = Dense(128, activation='relu', name='dense1x2')(x2)

    # dense block 2

    x1 = Dense(64, activation='relu', name='dense2x1')(x1)

    x2 = Dense(64, activation='relu', name='dense2x2')(x2)

    # dense block 3

    x1 = Dense(32, activation='relu', name='dense3x1')(x1)

    x2 = Dense(32, activation='relu', name='dense3x2')(x2)

    # dense block 4

    output1 = Dense(tasks_neurons_activations[tasks[0]][0], activation=tasks_neurons_activations[tasks[0]][1], name=tasks[0])(x1)

    output2 = Dense(tasks_neurons_activations[tasks[1]][0], activation=tasks_neurons_activations[tasks[1]][1], name=tasks[1])(x2)

    model = keras.models.Model(inputs=image, outputs=[output1, output2])


    losses = {tasks[0]: tasks_losses[tasks[0]],
              tasks[1]: tasks_losses[tasks[1]]
                }

    model.compile(optimizer='adam', loss=losses, metrics=['acc'])
    
    return model

def build_model_prop_3t(tasks, nb_prop_layers=0):

    DIM_IMG = 200

    image = Input(shape=(DIM_IMG, DIM_IMG, 3,))

    x1 = Conv2D(16, 3, activation='relu', name='conv1x1')(image)
    x1 = MaxPooling2D(2,name='pool1x1')(x1)

    x2 = Conv2D(16, 3, activation='relu', name='conv1x2')(image)
    x2 = MaxPooling2D(2, name='pool1x2')(x2)

    x3 = Conv2D(16, 3, activation='relu', name='conv1x3')(image)
    x3 = MaxPooling2D(2, name='pool1x3')(x3)
    
    if nb_prop_layers > 4:
        
        x1, x2, x3 = ProportionalAddition(n_tasks=3)([x1, x2, x3])

    # block 1

    x1 = Conv2D(16, 3, activation='relu', name='conv2x1')(x1)
    x1 = MaxPooling2D(2, name='pool2x1')(x1)

    x2 = Conv2D(16, 3, activation='relu', name='conv2x2')(x2)
    x2 = MaxPooling2D(2, name='pool2x2')(x2)


    x3 = Conv2D(16, 3, activation='relu', name='conv2x3')(x3)
    x3 = MaxPooling2D(2, name='pool2x3')(x3)

    if nb_prop_layers > 3:
        
        x1, x2, x3 = ProportionalAddition(n_tasks=3)([x1, x2, x3])
        
    # block 2

    x1 = Conv2D(32, 3, activation='relu', name='conv3x1')(x1)
    x1 = MaxPooling2D(2, name='pool3x1')(x1)

    x2 = Conv2D(32, 3, activation='relu', name='conv3x2')(x2)
    x2 = MaxPooling2D(2, name='pool3x2')(x2)

    x3 = Conv2D(32, 3, activation='relu', name='conv3x3')(x3)
    x3 = MaxPooling2D(2, name='pool3x3')(x3)
    
    if nb_prop_layers > 2:
        
        x1, x2, x3 = ProportionalAddition(n_tasks=3)([x1, x2, x3])

    # block 3

    x1 = Conv2D(32, 3, activation='relu', name='conv4x1')(x1)
    x1 = MaxPooling2D(2, name='pool4x1')(x1)

    x2 = Conv2D(32, 3, activation='relu', name='conv4x2')(x2)
    x2 = MaxPooling2D(2, name='pool4x2')(x2)

    x3 = Conv2D(32, 3, activation='relu', name='conv4x3')(x3)
    x3 = MaxPooling2D(2, name='pool4x3')(x3)

    if nb_prop_layers > 1:
        
        x1, x2, x3 = ProportionalAddition(n_tasks=3)([x1, x2, x3])

    # block 4

    x1 = Conv2D(64, 3, activation='relu', name='conv5x1')(x1)
    x1 = MaxPooling2D(2, name='pool5x1')(x1)

    x2 = Conv2D(64, 3, activation='relu', name='conv5x2')(x2)
    x2 = MaxPooling2D(2, name='pool5x2')(x2)

    x3 = Conv2D(64, 3, activation='relu', name='conv5x3')(x3)
    x3 = MaxPooling2D(2, name='pool5x3')(x3)

    if nb_prop_layers > 0:
        
        x1, x2, x3 = ProportionalAddition(n_tasks=3)([x1, x2, x3])

    x1 = Flatten(name='flattenx1')(x1)
    x2 = Flatten(name='flattenx2')(x2)
    x3 = Flatten(name='flattenx3')(x3)


    # dense block 1

    x1 = Dense(128, activation='relu', name='dense1x1')(x1)

    x2 = Dense(128, activation='relu', name='dense1x2')(x2)

    x3 = Dense(128, activation='relu', name='dense1x3')(x3)


    # dense block 2

    x1 = Dense(64, activation='relu', name='dense2x1')(x1)

    x2 = Dense(64, activation='relu', name='dense2x2')(x2)

    x3 = Dense(64, activation='relu', name='dense2x3')(x3)


    # dense block 3

    x1 = Dense(32, activation='relu', name='dense3x1')(x1)

    x2 = Dense(32, activation='relu', name='dense3x2')(x2)

    x3 = Dense(32, activation='relu', name='dense3x3')(x3)


    # dense block 4

    output1 = Dense(5,activation='softmax', name='ethnicity')(x1)

    output2 = Dense(1, activation='sigmoid', name='gender')(x2)

    output3 = Dense(1, activation='linear', name='age')(x3)


    model = keras.models.Model(inputs=image, outputs=[output1, output2, output3])


    losses = {"ethnicity": "sparse_categorical_crossentropy",
              "gender": "binary_crossentropy",
              "age": "mse"
                }


    model.compile(optimizer='adam', loss=losses, metrics=['acc'])
    
    return model