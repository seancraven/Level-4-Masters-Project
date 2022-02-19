import sys
sys.path.append('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/')

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import models 
import numpy as np
import Hyperparam_Testing.Testing_Notebooks.chirallag as cL
from scipy import stats
import matplotlib.pyplot as plt
#Function which makes data 
#mean 0 std 1
def data_normaliser(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data,axis = 0)
    return (data -mean)/std 


###Main class which shortens the variable network process 
###network class builds a sequential network with a relu activ according to hyperparam arg 
class network():
    def __init__(self,train_x,train_y,val_x,val_y, layer_shapes, optimizer = 'Adam', callback  = 1 ):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.optimizer = optimizer
        self.layer_shapes = layer_shapes
        self.callback = callback

    def build(self,model_summary = False, initializer = 'he_normal'):
        
        model = models.Sequential()
        model.add(layers.Dense(self.layer_shapes[0],activation= 'relu',input_shape = (self.train_x.shape[1],),kernel_initializer= initializer))
        for i in range(1,len(self.layer_shapes)):
            model.add(layers.Dense(self.layer_shapes[i],activation = 'relu',kernel_initializer= initializer))
        model.add(layers.Dense(1))
        model.compile(optimizer = self.optimizer,loss = 'mape', metrics = [['mean_absolute_error'],['mean_absolute_percentage_error']])
        
        if model_summary:
            model.summary()
        
        return model

#Trains the network 
class trained_network(network):
    def __init__(self,train_x,train_y,val_x,val_y, layer_shapes, optimizer = 'Adam', verbose = 0,epochs = 30,batch_size = 32,model_summary = False,initializer = 'he_normal',callback = 1):

        super().__init__(train_x,train_y,val_x,val_y, layer_shapes, optimizer)
        #print(layer_shapes)
        super().build()
        self.verbose  = verbose
        self.epochs = epochs 
        self.batch_size = batch_size
        self.model_summary = model_summary
        self.callback = callback
        self.initializer = initializer
        network = self.build(model_summary= self.model_summary, initializer = self.initializer)
        
        def fit(self,net):
            net_hist = net.fit( self.train_x, self.train_y, validation_data = (self.val_x,self.val_y), verbose  = self.verbose, epochs = self.epochs, use_multiprocessing = True,batch_size = self.batch_size)
            return net_hist
        self.history = fit(self,network).history 
        
#plot for layer testing        
class plotter():
    def __init__(self, history):
        self.history = history
    def basic(self):
        mape = self.history['mean_absolute_percentage_error']
        epochs = range(1,len(mape)+1) 
        plt.plot(epochs, mape)
    



## Basic smoothing implementation
def exponetial_smoothing(array,smoothing_factor):
    for i in range(1,len(array)):
        array[i] = array[i]*smoothing_factor + array[i-1]*(1-smoothing_factor)
    return array

## Memory reset, having some problems with implementaion of keras
def reset():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    session.close()
    print('Memory Reset')

##Determines if the gradient of the last third of the decay is less than loss
def neg_grad_tester(val_array, array):
    quart_length = int(len(val_array)/3)
    x = np.arange(len(val_array[:-quart_length]))
    lin_reg_val = stats.linregress(x,val_array[:-quart_length])
    lin_reg = stats.linregress(x,array[:-quart_length])
   
    if lin_reg_val.slope < lin_reg.slope*0.6:
        return True
    else:
        return False

#Makes the data noisy inside the bounds of the co-ordinates
def apply_noise(xs,noise_level):
    xs = xs+np.random.normal(size = xs.shape, scale= noise_level)
    xs[np.where(xs>1)] = 1
    xs[np.where(xs<0)] = 0
    return xs

##Generates pion 16 fields on interval 0-1 raised to the 1/4 
def gen_and_load(n_pred,n_val, onearr = False):
    number_predictions= n_pred
    N = 3
    F0 = 1
    gens = cL.gen_gellman(3)
    pi=np.random.rand(number_predictions,N*N-1)**0.25
    dpi=np.random.rand(number_predictions,N*N-1)**0.25
    orig_V = abs(cL.get_V(pi,dpi,gens,F0).real)
    output = np.hstack((pi,dpi,np.expand_dims(orig_V,axis=1)))
    if onearr:
        return output
    else:
        return [(output[:-n_val,:-1],output[:-n_val,-1]),(output[-n_val:,:-1],output[-n_val:,-1])]


def field_plotter(pions,pots):
    fig , ax = plt.subplots(2,8,sharey= True, figsize = (10,8))
    plt.subplots_adjust(hspace= 0.4,wspace= 0.5)
    for i in range(8):
        ax[0,i].hist(pions[:,i], density = True, color = 'black') 
        ax[0,i].set_xticks([0,1])
        ax[1,i].set_xticks([0,1])
        ax[1,i].hist(pions[:,8+i],density = True, color = 'black')
        ax[0,i].set_xlabel('$\phi_{{{}}}$'.format(i))
        ax[1,i].set_xlabel('$d\phi_{{{}}}$'.format(i))
    fig.supylabel('Frequency Density')
    fig_2 = plt.figure(figsize= (10,8))
    plt.hist(pots,density= True, bins = 1000, color = 'black',label=  '$\phi \sim U^{1/4}$')
    #plt.hist(df_pions[:,-1],density= True, bins = 1000, color = 'blue',label=  '$\phi \sim U^{1/4}$' )
    plt.xlabel('$V(\phi)$')
    plt.ylabel('Frequency Density')
    plt.legend()
