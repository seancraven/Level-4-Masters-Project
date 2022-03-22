import sys
sys.path.append('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/')
import random
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
    xs[np.where(xs>1)] = 2-xs[np.where(xs>1)]
    xs[np.where(xs<0)] = abs(xs[np.where(xs<0)])
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


##Functions to build Noisy Data Sets
class noisy():
    def __init__(self,sigma):
        self.sigma = sigma
    def keep_dims_with_cut(self,data,cutoff,buffer = 1.1):
        #The Cutoff functionality removes data below the point 
        #Running some fast tests There seems to be a reasonable amount of noise with the cutoff of the order 1% or so
        #If i want a fully populated dataset
        no_points =data.shape[0]
        indexes = np.where(data[:,16]>0.1)
        data = data[indexes]
        fraction = data.shape[0]/no_points
        #Math Found in Notebook Guassian_noise_trained_netowrk
        no_points_to_gen = int(buffer*(1-fraction**2)/fraction*data.shape[0])
        print('Training data cut for potential values below {}'.format(cutoff))
        print('Remaining data fraction after cut  = {}'.format(fraction))
        print('To retain {} training points generating {} more '.format(no_points,(no_points_to_gen)))
        ##Repopulate the array
        print('This produces {} usefull points'.format(data.shape[0]+int(no_points_to_gen*fraction)))
        
        
        new_points = self.data(no_points_to_gen)
        indexes = np.where(new_points[:,16]>0.1)
        new_points_to_append = new_points[indexes]
        output = np.vstack((data,new_points_to_append))
        return output[:no_points]
    
    def data(self,number_predictions,power= 0.25,cutoff = 0, keep_dim = True): #Define sigma globally 
        N = 3
        F0 = 1
        #Get Generator Matricies 
        gens = cL.gen_gellman(3)
        #Generate Fields
        pi=np.random.rand(number_predictions,N*N-1)**power
        dpi=np.random.rand(number_predictions,N*N-1)**power
        #Calculate V
        orig_V = abs(cL.get_V(pi,dpi,gens,F0).real)
        #Make the xvals noisey
        pi = apply_noise(pi,self.sigma)
        dpi = apply_noise(dpi,self.sigma)
        output = np.hstack((pi,dpi,np.expand_dims(orig_V,axis=1)))
        if keep_dim & (cutoff != 0):
            output =  self.keep_dims_with_cut(output,cutoff)
        return output





### Daniels Transformation and preparation stuff
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def adjointSUN(dim,liststruc):
    dimSUN=dim**2-1
#     admat = np.zeros((8,8,8))
    admat = np.zeros((dimSUN,dimSUN,dimSUN))
    for i in range(liststruc.shape[0]):
        strucc = liststruc[i]
        strucc1 = int(strucc[0])-1
        strucc2 = int(strucc[1])-1
        strucc3 = int(strucc[2])-1
        admat[strucc1,strucc2,strucc3]=strucc[3]
        admat[strucc1,strucc3,strucc2]=-strucc[3]
        admat[strucc3,strucc1,strucc2]=strucc[3]
        admat[strucc3,strucc2,strucc1]=-strucc[3]
        admat[strucc2,strucc3,strucc1]=strucc[3]
        admat[strucc2,strucc1,strucc3]=-strucc[3]
    return admat


# To do, make these work when not supplying stacks of epsilon and transformations.

def get_SO_trans(eps,N):
    ''' Produces 'infinitesimal' transformation matrices for SO(N) given epsilon.
        Epsilon is size of transformation and should be a numpy array of dimension 1.
        N must be larger than 1.
    '''
    
    trans_number = len(eps) 
    # Generate rotation matrices
    SO_samp = np.zeros((trans_number,N,N))

    # Set random off diagonal element of each matrix to one (surely must be a better way to do this)
    poss_inds = range(N)
    SO_inds = [random.sample(poss_inds,2) for i in range(trans_number)]
    SO_indsflat = [i*(N)**2+x*(N)+y for i,(x,y) in enumerate(SO_inds)]
    SO_samp.flat[SO_indsflat] += 1

    # Is this transpose faster than repeating assignment above?
    SO_samp = SO_samp - np.transpose(SO_samp,axes=[0,2,1])
    SO_trans = np.identity(N) + eps[:,None,None]*SO_samp
    # Normalise so that det = 1
    #norm = np.power(np.linalg.det(SO_trans),-1/(N))
    #SO_trans = norm[:,None,None]*SO_trans
    return SO_trans
  
def get_SU_trans(eps,N,liststruct,number_predictions):
    ''' Produces 'infinitesimal' transformation matrices for SO(N) given epsilon.
        Epsilon is size of transformation and should be a numpy array of dimension 1.
        N must be larger than 1.
        Calls adjointSUN which requires liststruct
    '''
    adjointSU = adjointSUN(N,liststruct)
    
    genno=np.random.randint(0,N*N-2,number_predictions)
    SU_trans = np.identity(N*N-1)+eps[:,None,None]*adjointSU[genno]
    # Normalise so that det = 1
    #norm = np.power(np.linalg.det(SU_trans),-1/(N*N-1))
    #SU_trans = norm[:,None,None]*SU_trans
    return SU_trans
    
def apply_trans(trans,vec):
    ''' Apply transformation trans to vector. Assumes both trans and vec are stacks.'''
    trans_vec=np.matmul(trans,vec[:,:,None]).squeeze()
    return trans_vec
    
