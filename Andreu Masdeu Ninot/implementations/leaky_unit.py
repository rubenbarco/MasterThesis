



import keras
import numpy as np
import pandas as pd



from keras.models import Model
from keras.layers import Input
import numpy as np
from keras.layers import dot

from keras import backend as K
from keras.layers import Layer

class ProportionalAddition(Layer):
    def __init__(self, n_tasks, **kwargs):
        super(ProportionalAddition, self).__init__(**kwargs)
        self.n_tasks = n_tasks
        
    def build(self, input_shape):
        batch_size, H, T, F = input_shape[0]
        self.alpha = self.add_weight(name='alpha',
                                     shape=(self.n_tasks, self.n_tasks), 
                                     initializer='normal',  # Try also 'ones' and 'uniform'
                                     trainable=True)
        super(ProportionalAddition, self).build(input_shape)

    def call(self, x):
        # x is a list of n tensors with shape=(batch_size, H, T, F)
        
        tensors = []
        for j in range(self.n_tasks):
            tensor = self.alpha[j, 0]*x[0]
            for i in range(1, len(x)):
                tensor += self.alpha[j, i]*x[i]
            tensors.append(tensor)
        return tensors

    def compute_output_shape(self, input_shape):
        output_shapes = []
        
        for i in range(self.n_tasks):
            output_shapes.append(input_shape[0])
        
        return output_shapes








class LeakyUnit(Layer):
    def __init__(self, n_tasks, **kwargs):
        
        super(LeakyUnit, self).__init__(**kwargs)
        self.n_tasks = n_tasks
        count = 0
        self.dict_indexs = {}
        
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if j != i:
                    self.dict_indexs[(i, j)] = count
                    count += 1
        
    def build(self, input_shape):
        batch_size, H, T, F = input_shape[0]
        self.Wr = self.add_weight(name='Wr',
                                     shape=(self.n_tasks*(self.n_tasks-1), H, T, F), 
                                     initializer='normal',  # Try also 'ones' and 'uniform'
                                     trainable=True)
        
        self.Wz = self.add_weight(name='Wz',
                                     shape=(self.n_tasks*(self.n_tasks-1), H, T, F), 
                                     initializer='normal',  # Try also 'ones' and 'uniform'
                                     trainable=True)
        
        self.W = self.add_weight(name='W',
                                     shape=(self.n_tasks*(self.n_tasks-1), H, T, F), 
                                     initializer='normal',  # Try also 'ones' and 'uniform'
                                     trainable=True)
        
        self.U = self.add_weight(name='U',
                                     shape=(self.n_tasks*(self.n_tasks-1), H, T, F), 
                                     initializer='normal',  # Try also 'ones' and 'uniform'
                                     trainable=True)
        super(LeakyUnit, self).build(input_shape)

    def call(self, x):
        # x is a list of n tensors with shape=(batch_size, H, T, F)
        
        tensors = []
        for i in range(self.n_tasks):
            
            zetas = []
            new_Fs = []
            
            for j in range(self.n_tasks):
                
                if j != i:
                    
                    r = K.sigmoid(self.matmul(self.Wr[self.dict_indexs[(i,j)]], x[i])+
                                self.matmul(self.Wr[self.dict_indexs[(i,j)]], x[j]))
                    
                    element_wise = r*x[j] #multply layer in keras does this
                    
                    
                    new_map = K.tanh(self.matmul(self.U[self.dict_indexs[(i,j)]], x[i]) +
                                  self.matmul(self.W[self.dict_indexs[(i,j)]], element_wise ))
                    
                    
                    zeta = K.sigmoid(self.matmul(self.Wz[self.dict_indexs[(i,j)]], x[i])+
                                self.matmul(self.Wz[self.dict_indexs[(i,j)]], x[j]))
                    
                    zetas.append(zeta)
                    new_Fs.append(new_map)
            
            tensor = self.matmul_2(zetas[0],x[i])+self.matmul_2((1-zetas[0]), new_Fs[0])
            
            for zeta1, new_map1 in zip(zetas[1:], new_Fs[1:]):
                
                tensor = self.matmul_2(zeta1,x[i])+self.matmul_2((1-zeta1), new_map1) + tensor
            
            tensors.append(tensor/self.n_tasks)
        
        return tensors
    
    def matmul(self, kernel, x):
            
        
        feature_maps = []
        
        for channel in range(int(x.shape[-1])):
            
            feature_maps.append(K.dot( x[:, :,:,channel], kernel[:,:, channel]))
            
        output = K.stack(feature_maps, axis=3)
        return output

    def matmul_2(self, y, x):
            
        
        feature_maps = []
        
        for channel in range(int(x.shape[-1])):
            
            feature_maps.append(dot([ x[:, :,:,channel], y[:, :,:, channel]], axes=-1))

        output = K.stack(feature_maps, axis=3)
        return output

    def compute_output_shape(self, input_shape):
        output_shapes = []
        
        for i in range(self.n_tasks):
            output_shapes.append(input_shape[0])
        
        return output_shapes


