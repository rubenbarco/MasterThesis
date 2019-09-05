import keras
import pandas as pd
import numpy as np

from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Conv2D, Dense, Concatenate

from leaky_unit import LeakyUnit, ProportionalAddition
import os
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace

DIM_IMG = 200

tasks_losses = {'age':'mse', 'gender':'binary_crossentropy', 
                'ethnicity': 'sparse_categorical_crossentropy'}

tasks_neurons_activations = {'age':[1, 'linear'], 'gender':[1, 'sigmoid'], 
                'ethnicity': [5, 'softmax']}

def hard_param_model_2t(tasks, shared_blocks = 4):
    image = Input(shape=(DIM_IMG, DIM_IMG, 3,))
    first_time = True
    
    if  5 > shared_blocks:
               
        x1 = Conv2D(16, 3, activation='relu')(image)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(16, 3, activation='relu')(image)
        x2 = MaxPooling2D(2)(x2)

    else:
        
        x = Conv2D(16, 3, activation='relu')(image)
        x = MaxPooling2D(2)(x)

    # block 1

    if  4 > shared_blocks:
            
        x1 = Conv2D(16, 3, activation='relu')(x1)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(16, 3, activation='relu')(x2)
        x2 = MaxPooling2D(2)(x2)

    else:
        
        if shared_blocks == 4:
            x = Concatenate(axis=2)([x1,x2])
            
        x = Conv2D(16, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)
        
        
    if 3 > shared_blocks:

        x1 = Conv2D(32, 3, activation='relu')(x1)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(32, 3, activation='relu')(x2)
        x2 = MaxPooling2D(2)(x2)

    else:
        
        if shared_blocks == 3:
            x = Concatenate(axis=2)([x1,x2])
            
        x = Conv2D(32, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)


    # bloc
        
    if  2 > shared_blocks:

        x1 = Conv2D(32, 3, activation='relu')(image)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(32, 3, activation='relu')(image)
        x2 = MaxPooling2D(2)(x2)

    else:
        if shared_blocks == 2:
            x = Concatenate(axis=2)([x1,x2])
        x = Conv2D(32, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)


    #
        
    if  1 > shared_blocks:

        x1 = Conv2D(64, 3, activation='relu')(x1)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(64, 3, activation='relu')(x2)
        x2 = MaxPooling2D(2)(x2)

    else:
        if shared_blocks == 1:
            x = Concatenate(axis=2)([x1,x2])
        x = Conv2D(64, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)

        
    x1 = Flatten()(x)
    x2 = Flatten()(x)
   
    # dense block 1

    x1 = Dense(128, activation='relu')(x1)

    x2 = Dense(128, activation='relu')(x2)

    # dense block 2

    x1 = Dense(64, activation='relu')(x1)

    x2 = Dense(64, activation='relu')(x2)

    # dense block 3

    x1 = Dense(32, activation='relu')(x1)

    x2 = Dense(32, activation='relu')(x2)

    # dense block 4

    output1 = Dense(tasks_neurons_activations[tasks[0]][0], activation=tasks_neurons_activations[tasks[0]][1], name=tasks[0])(x1)

    output2 = Dense(tasks_neurons_activations[tasks[1]][0], activation=tasks_neurons_activations[tasks[1]][1], name=tasks[1])(x2)

    model = keras.models.Model(inputs=image, outputs=[output1, output2])


    losses = {tasks[0]: tasks_losses[tasks[0]],
              tasks[1]: tasks_losses[tasks[1]]
                }

    model.compile(optimizer='adam', loss=losses, metrics=['acc'])
    
    return model


def hard_param_model_3t(tasks, shared_blocks = 4):
    image = Input(shape=(DIM_IMG, DIM_IMG, 3,))
    
    first_time = True
    
    if 5 > shared_blocks:
        x1 = Conv2D(16, 3, activation='relu')(image)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(16, 3, activation='relu')(image)
        x2 = MaxPooling2D(2)(x2)
        
        x3 = Conv2D(16, 3, activation='relu')(image)
        x3 = MaxPooling2D(2)(x3)

    else:
        
        x = Conv2D(16, 3, activation='relu')(image)
        x = MaxPooling2D(2)(x)

   

    if 4 > shared_blocks:

        x1 = Conv2D(16, 3, activation='relu')(x1)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(16, 3, activation='relu')(x2)
        x2 = MaxPooling2D(2)(x2)
        
        x3 = Conv2D(16, 3, activation='relu')(x3)
        x3 = MaxPooling2D(2)(x3)

    else:
        if shared_blocks == 4:
            x = Concatenate(axis=2)([x1,x2])
        x = Conv2D(16, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)
        
    # block 2

    if 3 > shared_blocks:

        x1 = Conv2D(32, 3, activation='relu')(x1)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(32, 3, activation='relu')(x2)
        x2 = MaxPooling2D(2)(x2)
        
        x3 = Conv2D(32, 3, activation='relu')(x3)
        x3 = MaxPooling2D(2)(x3)

    else:
        if shared_blocks == 3:
            x = Concatenate(axis=2)([x1,x2])
        x = Conv2D(32, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)

        
    if 2 > shared_blocks:

        x1 = Conv2D(32, 3, activation='relu')(x1)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(32, 3, activation='relu')(x2)
        x2 = MaxPooling2D(2)(x2)
        
        x3 = Conv2D(32, 3, activation='relu')(x3)
        x3 = MaxPooling2D(2)(x3)

    else:
        if shared_blocks == 2:
            x = Concatenate(axis=2)([x1,x2])
        x = Conv2D(32, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)


    # block 4

    if 1 > shared_blocks:

        x1 = Conv2D(64, 3, activation='relu')(x1)
        x1 = MaxPooling2D(2)(x1)

        x2 = Conv2D(64, 3, activation='relu')(x2)
        x2 = MaxPooling2D(2)(x2)
        
        x3 = Conv2D(64, 3, activation='relu')(x3)
        x3 = MaxPooling2D(2)(x3)

    else:
        if shared_blocks == 1:
            x = Concatenate(axis=2)([x1,x2])
        x = Conv2D(64, 3, activation='relu')(x)
        x = MaxPooling2D(2)(x)

        
    x1 = Flatten()(x)
    x2 = Flatten()(x)
    x3 = Flatten()(x)

    # dense block 1

    x1 = Dense(128, activation='relu')(x1)

    x2 = Dense(128, activation='relu')(x2)
    
    x3 = Dense(128, activation='relu')(x3)
    # dense block 2

    x1 = Dense(64, activation='relu')(x1)

    x2 = Dense(64, activation='relu')(x2)
    
    x3 = Dense(64, activation='relu')(x3)


    # dense block 3

    x1 = Dense(32, activation='relu')(x1)

    x2 = Dense(32, activation='relu')(x2)
    
    x3 = Dense(32, activation='relu')(x3)

    # dense block 4

    output1 = Dense(tasks_neurons_activations[tasks[0]][0], activation=tasks_neurons_activations[tasks[0]][1], name=tasks[0])(x1)

    output2 = Dense(tasks_neurons_activations[tasks[1]][0], activation=tasks_neurons_activations[tasks[1]][1], name=tasks[1])(x2)
    
    output3 = Dense(tasks_neurons_activations[tasks[2]][0], activation=tasks_neurons_activations[tasks[2]][1], name=tasks[2])(x2)


    model = keras.models.Model(inputs=image, outputs=[output1, output2, output3])


    losses = {tasks[0]: tasks_losses[tasks[0]],
              tasks[1]: tasks_losses[tasks[1]],
              tasks[2]: tasks_losses[tasks[2]]
                }

    model.compile(optimizer='adam', loss=losses, metrics=['acc'])
    
    return model