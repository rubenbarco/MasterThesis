import keras
import pandas as pd
import numpy as np

from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, Conv2D, Dense

from leaky_unit import LeakyUnit, ProportionalAddition
import os
from sklearn.model_selection import train_test_split
from PIL import Image

from build_prop_model import build_model_prop
from keras.callbacks import EarlyStopping
from build_hard_param_backwards import hard_param_model_2t

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

for tasks in [['gender', 'ethnicity']]:
    for nb_shared_blocks in range(1,6):
        
        for i in range(3):

            train_labels = {tasks[0]: np.array(tasks_labels[tasks[0]][0]),
                  tasks[1]: np.array(tasks_labels[tasks[1]][0])
                    }

            test_labels = {tasks[0]: np.array(tasks_labels[tasks[0]][1]),
                  tasks[1]: np.array(tasks_labels[tasks[1]][1])
                    }

            model = hard_param_model_2t(tasks, nb_shared_blocks)
            

            es = EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=1)

            history = model.fit(np.array(trainX), train_labels, epochs=20, 
                      batch_size=64, validation_data=(np.array(testX), test_labels), verbose=2, callbacks=[es])
            if i == 0:
                model.save('hardparam{}{}{}b.h5'.format(nb_shared_blocks, tasks[0], tasks[1]))
                
            with open("history_hardparam{}{}{}{}back.pkl".format(tasks[0], tasks[1], nb_shared_blocks, i), "wb") as handle:
                pickle.dump(history.history, handle)



