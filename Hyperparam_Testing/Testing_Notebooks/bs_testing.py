# %%
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Common_Functions as cf 
from mpl_toolkits import mplot3d
plt.rc('font', family='serif')
import matplotlib
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'font.size': 16})

# %%
#Load Data
td = cf.gen_and_load(10**5,10**3)
#Hold Out Validation
train_x, train_y = td[0]
val_x, val_y = td[1]

# %%
#Adam Optimiser
opt = keras.optimizers.Adam( amsgrad= True)

# %%
## Function Creates multiple Identical networks while varying batch size.
## Returns numpy array of Batch sizes 
## Returns numpy array of min MAPE of size len(batch_size),repeats
def batch_size_tester(start,step,no_items,repeats = 5,section = None):
    
    batch_sizes = [start+step*j for j in range(no_items)]
    is_last_val_bool = np.zeros_like(batch_sizes)
    min_mape = np.zeros((len(batch_sizes),repeats))
    for i,batch_size in enumerate(batch_sizes):
        print('Batch Size ',batch_size)
        last_val_bool = 0
        for j in range(repeats):
            #train network get MAPE Histories
            mape_df = cf.trained_network(train_x[:section],train_y[:section],val_x,val_y,[512,512], optimizer= opt, verbose= 1,batch_size = batch_size).history
            #Smooth Arrays
            val_mape_ar_smoothed = cf.exponetial_smoothing(np.array(mape_df['val_mean_absolute_percentage_error']),0.4)
            loss_ar_smoothed = cf.exponetial_smoothing(np.array(mape_df['loss']),0.4)
            min_mape[i,j] = np.min(val_mape_ar_smoothed)
            #convergance test 
            if cf.neg_grad_tester(val_mape_ar_smoothed,loss_ar_smoothed):
                last_val_bool +=1
                print('Negative Grad Identified')
            else:
                pass
            tf.keras.backend.clear_session()  
                
        if last_val_bool >= 2:
            is_last_val_bool[i] = 1
        else:
            is_last_val_bool[i] = 0
    return min_mape, batch_sizes, last_val_bool

# %%
min_mape, batch_sizes , last_val_bool = batch_size_tester(4,20,20, repeats= 10)

# %%

##For very low batch sizes some networks dont converge
##This removes them, they are set to the mean plus one std some kind of average bad point
def clean_min_mape(min_mape):
    orig_min = min_mape.copy()
    mean_min_mape = np.mean(min_mape,axis =1 )
    mean_min_mape_array = (np.ones_like(min_mape).transpose()*mean_min_mape).transpose()
    std = np.std(min_mape,axis = 1)
    mean_plus_1_sigma = mean_min_mape+std
    mean_plus_1_sigma_array = (np.ones_like(min_mape).transpose()*mean_plus_1_sigma).transpose()
    #print(mean_plus_1_sigma_array)
    index = np.where(min_mape> mean_plus_1_sigma_array)
    min_mape[index] = mean_min_mape_array[index]

    return min_mape
cleaned_min_mape = clean_min_mape(min_mape)
#print(clean_mean)
clean_mean = np.mean(cleaned_min_mape,axis=1)
clean_ste = np.std(cleaned_min_mape,axis = 1)/10**0.5
print(min_mape[0,:])

# %%
#print(last_val_bool)
#Calculation of plotting quantities
np.savetxt('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Hyperparam_Testing/Testing_Notebooks/Testing_Data/bs/batch_size_18_02.csv',min_mape,delimiter= ',')
np.savetxt('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Hyperparam_Testing/Testing_Notebooks/Testing_Data/bs/batch_size_bool_18_02.csv',last_val_bool,delimiter= ',')
min_mape = np.loadtxt('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Hyperparam_Testing/Testing_Notebooks/Testing_Data/bs/batch_size_18_02.csv',delimiter= ',')
last_val_bool = np.loadtxt('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Hyperparam_Testing/Testing_Notebooks/Testing_Data/bs/batch_size_bool_18_02.csv',delimiter= ',')
batch_sizes = np.arange(4,501,step = 16)

mean_min_mape = np.mean(min_mape,axis= 1)
std_min_mape = np.std(min_mape,axis= 1)
mse = std_min_mape/min_mape.shape[1]**0.5
print(batch_sizes.shape)
#print(clean_mean.shape)
print(min_mape)

# %%
batch_sizes[np.argmin(mean_min_mape)]

# %%
#plotting
fig, ax = plt.subplots(1,1, figsize= (10,5))
#ax = fig.add_axes((0,0,1,1))
ax.errorbar(batch_sizes,clean_mean,clean_ste,color = 'black',capsize= 2)
ax.plot(4,clean_mean[0],marker = 'x', linestyle = '', c = 'red')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Mean Minimum MAPE')
ax.set_xticklabels(np.linspace(4,500,6,dtype= int))
ax.set_xticks(np.linspace(4,500,6,dtype= int))


# %%
fig.savefig('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Figures/Batch_size_2.png',dpi = 300, transparent= False)

# %%



