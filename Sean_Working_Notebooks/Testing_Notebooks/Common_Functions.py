import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import models 
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from scipy import stats

def data_normaliser(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data,axis = 0)
    return (data -mean)/std 

####  This iscommon class/ functions notebook to call out of so that ecerything is easily accessible

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
        #print(self.layer_shapes)
        model = models.Sequential()
        ##Layers 
        #print(self.train_x.shape[1])
    
        model.add(layers.Dense(self.layer_shapes[0],activation= 'relu',input_shape = (self.train_x.shape[1],),kernel_initializer= initializer))
        for i in range(1,len(self.layer_shapes)):
            #print(i)
            model.add(layers.Dense(self.layer_shapes[i],activation = 'relu',kernel_initializer= initializer))
        model.add(layers.Dense(1))
        model.compile(optimizer = self.optimizer,loss = 'mape', metrics = [['mean_absolute_error'],['mean_absolute_percentage_error']])
        
        if model_summary:
            model.summary()
        
        return model

def scheduler_dead(epoch,lr):
    return lr

#### I fucking Hate How many hyperparameters there are 
class trained_network(network):
    def __init__(self,train_x,train_y,val_x,val_y, layer_shapes, optimizer = 'Adam', verbose = 0,epochs = 30,batch_size = 32,model_summary = False, callback = scheduler_dead,initializer = 'he_normal'):

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
        
        
class plotter():
    def __init__(self, history):
        self.history = history
    def basic(self):
        mape = self.history['mean_absolute_percentage_error']
        #percentageloss = 
        epochs = range(1,len(mape)+1) 
        plt.plot(epochs, mape)
    
### What do I want out of this??
#- Be able to vary shape of network easily 
#- Output Basic plot for fast comparison
#- Output History data for more in depth comparison 
#- I would quite like to automate my testing, so say I pass list of [[3,36],[2.36]...etc] it runs and stores some measure of how good these were, often interpreting the data requires a graph so maybe it needs to plot a bunch of subplots
#- Save Fig 
#- Diff Optimisers 



## Definin
def exponetial_smoothing(array,smoothing_factor):
    for i in range(1,len(array)):
        array[i] = array[i]*smoothing_factor + array[i-1]*(1-smoothing_factor)
    return array


def reset():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    session.close()
    print('Memory Reset')

def neg_grad_tester(val_array, array):
    quart_length = int(len(val_array)/3)
    #print(quart_length)
    x = np.arange(len(val_array[:-quart_length]))
    lin_reg_val = stats.linregress(x,val_array[:-quart_length])
    lin_reg = stats.linregress(x,array[:-quart_length])
   
    if lin_reg_val.slope < lin_reg.slope*0.6:
        return True
    else:
        return False