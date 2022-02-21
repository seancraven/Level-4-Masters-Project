
from importlib import import_module
import sys 
sys.path.append('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/')
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import Hyperparam_Testing.Testing_Notebooks.Common_Functions as cf 
from sklearn.model_selection import train_test_split as tts
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import models 
import datetime

time = datetime.datetime.now()
time_string = time.strftime('%d_%m_%y')

points =10**7
sigma = 0.03
cutoff = 0.1
#Gen Data
data = cf.noisy(sigma).data(points,cutoff = cutoff)
#Split to input output 
x,y = data[:,:16],data[:,16]
train_x, val_x, train_y, val_y =  tts(x,y,test_size = 0.1)

power = np.log10(len(y))

#build model
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

#fit model
fname = './Saved_Optimised_Networks/10{}datapoints_noise_{}_cutoff_{}'.format(power,sigma,cutoff)+'_date'+time_string+'_best_cp_.h5'
ts = datetime.datetime.now()
#best model callback
checkpoint = tf.keras.callbacks.ModelCheckpoint(fname,save_best_only=True)
history = model.fit(train_x,train_y,validation_data=(val_x,val_y),batch_size= 32 , epochs = epoch_num,callbacks = [checkpoint],verbose = 0)
#save model
te = datetime.datetime.now()
td = te-ts

fname = './Saved_Optimised_Networks/10{}datapoints_noise_{}_cutoff_{}'.format(power,sigma,cutoff)+'_date'+time_string+'.h5'
model.save(fname)
hrs = td.seconds//3600
mins = (td.seconds//60)%60
sec = (td.seconds%60)%60
print('Training Time {}:{}:{}'.format(hrs,mins,sec))
mape = history.history['val_loss']
print('Best Network Mape with {} points and noise $\sigma = {}$: {}'.format(points,sigma,np.min(mape)))

