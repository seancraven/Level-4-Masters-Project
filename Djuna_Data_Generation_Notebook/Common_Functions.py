import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import models 
import numpy as np

def data_normaliser(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data,axis = 0)
    return (data -mean)/std 

####  This iscommon class/ functions notebook to call out of so that ecerything is easily accessible

class network():
    def __init__(self,train_x,train_y,val_x,val_y, layer_shapes, optimizer = 'Adam', ):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.optimizer = optimizer
        self.layer_shapes = layer_shapes
    def build(self,model_summary = False):
        #print(self.layer_shapes)
        model = models.Sequential()
        ##Layers 
        model.add(layers.Dense(self.layer_shapes[0],activation= 'relu',input_shape = (self.train_x.shape[1],)))
        for i in range(1,len(self.layer_shapes)):
            #print(i)
            model.add(layers.Dense(self.layer_shapes[i],activation = 'relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer = self.optimizer,loss = 'mse', metrics = ['mean_absolute_percentage_error'])
        
        if model_summary:
            model.summary()
        
        return model

#### I fucking Hate How many hyperparameters there are 
class trained_network(network):
    def __init__(self,train_x,train_y,val_x,val_y, layer_shapes, optimizer = 'Adam', verbose = 0,epochs = 100):
        super().__init__(train_x,train_y,val_x,val_y, layer_shapes, optimizer)
        #print(layer_shapes)
        super().build()
        self.verbose  = verbose
        self.epochs = epochs 
        network = self.build()
        
        def fit(self,net):
            net_hist = net.fit( self.train_x, self.train_y, validation_data = (self.val_x,self.val_y), verbose  = self.verbose, epochs = self.epochs, use_multiprocessing = True)
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

