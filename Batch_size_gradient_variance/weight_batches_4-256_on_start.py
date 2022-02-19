import sys
sys.path.append('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/')
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import models 
import os
import matplotlib


matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'font.size': 16})
from keras import backend as K
import os


##Load Training Data
treaining_data_array = np.loadtxt('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Batch_size_gradient_variance/Weights_Data/training_data.csv', delimiter = ',')
training_data = ([treaining_data_array[:-1000,:16],treaining_data_array[:-1000,16]],[treaining_data_array[-1000:,:16],treaining_data_array[-1000:,16]])


#Hyperparam Which is edited

# Default hyperparams for network 
epochs = 10
nodes_per_layer = 300
optimizer = keras.optimizers.Adam(beta_1 = 0.9,beta_2 = 0.98)

#split_training_data
train_x, train_y = training_data[0]
val_x,val_y = training_data[1]

#Dictionary to append weights to 
weights_dict = {}
#Callback to record weights

class GetWeights(tf.keras.callbacks.Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        #weights_dict = {}

    def on_epoch_start(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            #print('Layer %s has weights of shape %s and biases of shape %s' %(
                #layer_i, np.shape(w), np.shape(b)))

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                weights_dict['w_'+str(layer_i+1)] = w
                weights_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                weights_dict['w_'+str(layer_i+1)] = np.dstack(
                    (weights_dict['w_'+str(layer_i+1)], w))
                # append new weights to previously-created weights array
                weights_dict['b_'+str(layer_i+1)] = np.dstack(
                    (weights_dict['b_'+str(layer_i+1)], b))
for batchsize in [2**j for j in range(2,9)]:
    print('Training Batch Size {}'.format(batchsize))
    K.clear_session()
    #Compile network
    model = models.Sequential()
    model.add(layers.Dense(nodes_per_layer, activation ='relu', kernel_initializer= 'random_normal', input_shape = [train_x.shape[1]]))
    model.add(layers.Dense(nodes_per_layer, activation ='relu', kernel_initializer= 'random_normal'))
    model.add(layers.Dense(nodes_per_layer, activation ='relu', kernel_initializer= 'random_normal'))
    model.add(layers.Dense(nodes_per_layer, activation ='relu', kernel_initializer= 'random_normal'))
    model.add(layers.Dense(nodes_per_layer, activation ='relu', kernel_initializer= 'random_normal'))
    model.add(layers.Dense(1))
    model.compile(optimizer = optimizer,loss = 'mape', metrics = ['mean_absolute_percentage_error'])
    #model.summary()

    # Train Network 
    model.fit(train_x,train_y,validation_data=(val_x,val_y),batch_size= batchsize , epochs = epochs, callbacks= GetWeights(),verbose= 0)



    #Arrays to store mean and std of the change in weight per epoch data 
    
    
    rms_weight = np.zeros((epochs,6))
    std_weight = np.zeros((epochs,6))
    rms_wieght_delta = np.zeros((epochs,6))
    std_wieght_delta = np.zeros((epochs,6))
    
    
    for j,_ in enumerate(weights_dict.items()):
        if j%2 ==0 :#Only Weights not Biases
            key = list(weights_dict.keys())[j]
            for i in range(epochs):
                #calculate mean absolute value of weights 
                rs_weight = abs(((weights_dict[key])[:,:,i]))
                rms_weight[i,j//2] = np.mean(rs_weight)
                std_weight[i,j//2] = np.std(rs_weight)
                #calculate mean absolute value of change in weights between epochs
                if i ==0: 
                    rs_weight_delta = abs(((weights_dict[key])[:,:,1])-((weights_dict[key])[:,:,0]))
                else:
                    rs_weight_delta = abs(((weights_dict[key])[:,:,i])-((weights_dict[key])[:,:,i-1]))
                rms_wieght_delta[i,j//2] = np.mean(rs_weight_delta)
                std_wieght_delta[i,j//2] = np.std(rs_weight_delta)
    ##Saving Weights and Weight Deltas            
    directory ='/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Batch_size_gradient_variance/Weights_Data_Start'
    extention_w = 'weight/weigth_batch_size_{}'.format(batchsize)
    joined_w = os.path.join(directory,extention_w)
    extention_d= 'delta/weight_delta_batch_size_{}'.format(batchsize)
    joined_d = os.path.join(directory,extention_d)

    np.savetxt(joined_d+'rms.csv',rms_weight,delimiter = ',')
    np.savetxt(joined_d+'std.csv',std_weight,delimiter = ',')
    np.savetxt(joined_w+'rms.csv',rms_wieght_delta,delimiter = ',')
    np.savetxt(joined_w+'std.csv',std_wieght_delta,delimiter = ',')

