import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Conv2D, Dense

from leaky_unit import LeakyUnit
import os
import tqdm
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace


def build_model():

    DIM_IMG = 200
    n_tasks = 3

    img_input = Input(shape=(DIM_IMG, DIM_IMG, 3,))

    # Block 1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1-1')(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2-1')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1-1')(x1)

    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1-2')(img_input)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2-2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1-2')(x2)

    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1-3')(img_input)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2-3')(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1-3')(x3)

    # Block 2
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1-1')(
        x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2-1')(
        x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2-1')(x1)

    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1-2')(
        x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2-2')(
        x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2-2')(x2)

    x3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1-3')(
        x3)
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2-3')(
        x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2-3')(x3)

    # Block 3
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1-1')(
        x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2-1')(
        x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3-1')(
        x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3-1')(x1)

    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1-2')(
        x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2-2')(
        x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3-2')(
        x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3-2')(x2)

    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1-3')(
        x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2-3')(
        x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3-3')(
        x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3-3')(x3)

    # Block 4
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1-1')(
        x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2-1')(
        x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3-1')(
        x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4-1')(x1)


    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1-2')(
        x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2-2')(
        x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3-2')(
        x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4-2')(x2)


    x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1-3')(
        x3)
    x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2-3')(
        x3)
    x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3-3')(
        x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4-3')(x3)

    # Block 5
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1-1')(
        x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2-1')(
        x1)
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3-1')(
        x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5-1')(x1)


    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1-2')(
        x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2-2')(
        x2)
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3-2')(
        x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5-2')(x2)


    x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1-3')(
        x3)
    x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2-3')(
        x3)
    x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3-3')(
        x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5-3')(x3)

    #FC

    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x3 = Flatten()(x3)



    # dense block 4

    output1 = Dense(1, name='age')(x1)

    output2 = Dense(1, activation='sigmoid', name='gender')(x2)

    output3 = Dense(5, activation='softmax', name='ethnicity')(x3)


    model = keras.models.Model(inputs=img_input, outputs=[output1, output2, output3])


    losses = {"age": "mse",
              "gender": "binary_crossentropy",
              "ethnicity" : "sparse_categorical_crossentropy"
                }

    model.compile(optimizer='adam', loss=losses, metrics=['acc'])

    return model
