
import sys 
sys.path.append('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/')
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import models 
import os
from keras.regularizers import l2
import matplotlib
from multiprocessing import Pool
import Hyperparam_Testing.Testing_Notebooks.Common_Functions as cf 
import Hyperparam_Testing.Testing_Notebooks.chirallag as cL

matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'font.size': 16})

###Slight alteration on the previous function, I shouldnt code it like this in future make a function that generates data then train test split 
def gen_and_load_noisy(n_pred,n_val,sigma):
    number_predictions= n_pred
    N = 3
    F0 = 1
    gens = cL.gen_gellman(3)
    pi=np.random.rand(number_predictions,N*N-1)**0.25
    dpi=np.random.rand(number_predictions,N*N-1)**0.25
    orig_V = abs(cL.get_V(pi,dpi,gens,F0).real)
    #Make the xvals noisey
    pi = cf.apply_noise(pi,sigma)
    dpi = cf.apply_noise(dpi,sigma)
    output = np.hstack((pi,dpi,np.expand_dims(orig_V,axis=1)))
    
    return [(output[:-n_val,:-1],output[:-n_val,-1]),(output[-n_val:,:-1],output[-n_val:,-1])]

points  =10**7
val_points = 10**5
sigma = 0.03
data = gen_and_load_noisy(points,val_points,sigma)
train_x ,train_y = data[0]
val_x,val_y = data[1]

opt = keras.optimizers.Adam(beta_1= 0.9, beta_2= 0.98)


model = models.Sequential()

model.add(layers.Input(train_x.shape[1]))
model.add(layers.Dense(300,activation= 'relu',kernel_initializer= 'random_normal'))
model.add(layers.Dense(300,activation= 'relu',kernel_initializer= 'random_normal'))
model.add(layers.Dense(300,activation= 'relu',kernel_initializer= 'random_normal'))
model.add(layers.Dense(300,activation= 'relu',kernel_initializer= 'random_normal'))
model.add(layers.Dense(300,activation= 'relu',kernel_initializer= 'random_normal'))
model.add(layers.Dense(1))
model.compile(optimizer = opt,loss = 'mape', metrics = [['mean_absolute_error'],['mean_absolute_percentage_error']])
model.summary()

epoch_num = 100


history = model.fit(train_x,train_y,validation_data=(val_x,val_y),batch_size= 32 , epochs = epoch_num)

model.save('./5_10{}datapoints_noise_{}.h5'.format((np.log10(train_y.shape[0]+val_y.shape[0])),sigma))

mape = history.history['val_mean_absolute_percentage_error']
print('Best Network Mape with {} points and noise $\sigma = {}$: {}'.format(points,sigma,mape))

