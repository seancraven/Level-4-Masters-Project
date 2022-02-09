import sys
sys.path.append('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project')
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit as fit 
# Setup plotting with matplotlib
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
## setup latex plotting
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
## make font size bigger
matplotlib.rcParams.update({'font.size': 16})
## but make legend smaller
matplotlib.rcParams.update({'legend.fontsize': 14})
## change line thickness
matplotlib.rcParams.update({'lines.linewidth' : 1.75})

import Hyperparam_Testing.Testing_Notebooks.chirallag as cL
import Hyperparam_Testing.Testing_Notebooks.Common_Functions as cf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import models 
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view


N=3
gens = cL.gen_gellman(3)
F0 = 1

window = 33
step=1

epsmax = 100
epsmin = 0.001
eps_intervals=200
number_predictions=100000


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
        
liststrucSU3 = np.array([[1,2,3,1],[1,4,7,.5],[1,5,6,-.5],[2,4,6,.5],[2,5,7,.5],[3,4,5,.5],[3,6,7,-.5],[4,5,8,3**0.5/2],[6,7,8,3**0.5/2]])
liststrucSU2 = np.array([[1,2,3,1],[3,2,1,1]])

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

adjointSU3 = adjointSUN(3,liststrucSU3)



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
  
def get_SU_trans(eps,N,liststruct):
    ''' Produces 'infinitesimal' transformation matrices for SO(N) given epsilon.
        Epsilon is size of transformation and should be a numpy array of dimension 1.
        N must be larger than 1.
        Calls adjointSUN which requires liststruct
    '''
    adjointSU = adjointSUN(N,liststruct)
    
    genno=np.random.randint(0,N*N-2,number_predictions)
    SU_trans = np.identity(N*N-1)+eps[:,None,None]*adjointSU[genno]
    return SU_trans
    
def apply_trans(trans,vec):
    ''' Apply transformation trans to vector. Assumes both trans and vec are stacks.'''
    trans_vec=np.matmul(trans,vec[:,:,None]).squeeze()
    return trans_vec
    
# Compare Transformations


test_data= cf.gen_and_load(10**5,0) 
pi_dpi,v = test_data[1]
pis =pi_dpi[:,:8]
dpis = pi_dpi[:,8:]
pi=pis
dpi=dpis

# Get epsilons, create copies
eps_vals=np.logspace(np.log10(epsmin),np.log10(epsmax),num=eps_intervals)
eps = np.zeros(number_predictions)
for i,val in enumerate(eps_vals): 
    eps[i*int(number_predictions/eps_intervals):(i+1)*int(number_predictions/eps_intervals)] = val
# Set leftovers to max value
eps[(i+1)*int(number_predictions/eps_intervals):]=val

# SO(N) transformations
SO_trans = get_SO_trans(eps,N*N-1)

SO_pi = apply_trans(SO_trans,pi)
SO_dpi = apply_trans(SO_trans,dpi)

# SU(N) transformations

SU_trans = get_SU_trans(eps,N,liststrucSU3)

SU_pi = apply_trans(SU_trans,pi)
SU_dpi = apply_trans(SU_trans,dpi)


# Get potential values, only care about real part (non zero imag should just be numerical precision errors)

orig_V = v
SO_V = cL.get_V(SO_pi,SO_dpi,gens,F0).real
SU_V = cL.get_V(SU_pi,SU_dpi,gens,F0).real

SO_Vdiff = (abs(SO_V) - orig_V)/orig_V
SU_Vdiff = (abs(SU_V) - orig_V)/orig_V





### Generate New test data for nn prediction


SU_dpi_nn = apply_trans(SU_trans,dpis)
SU_pi_nn = apply_trans(SU_trans,pis)
SU_pi_dpi_nn = np.hstack((SU_pi_nn,SU_dpi_nn))
SU_V_nn_test = cL.get_V(SU_pi_nn,SU_dpi_nn,gens,F0).real
###I only want the transformed data on the interval 0-1


model = keras.models.load_model(
    '/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Hyperparam_Testing/Testing_Notebooks/Optimised_Network/Saved_Optimised_Networks/106datapoints.h5')

pred_V = model.predict(pi_dpi)[:,0]
MAPE = np.mean(abs((pred_V - v)/v))
print('MAPE = ',round(MAPE*100,2))

V_nn = model.predict(SU_pi_dpi_nn)
delta_V_nn = abs((V_nn[:,0]-v)/v)
# Create windows





SO_Vdiff_eps = [np.abs(SO_Vdiff[np.where(eps==val)]) for val in eps_vals] ##subsampling only the points which have epsilon values
mean_SO = np.mean(SO_Vdiff_eps,axis=-1)
std_dev_SO = np.std(SO_Vdiff_eps,axis=-1)
std_error_SO = std_dev_SO/np.sqrt([len(s) for s in SO_Vdiff_eps])
SU_Vdiff_eps = [np.abs(SU_Vdiff[np.where(eps==val)]) for val in eps_vals]
mean_SU = np.mean(SU_Vdiff_eps,axis=-1)
std_dev_SU = np.std(SU_Vdiff_eps,axis=-1)
std_error_SU = std_dev_SU/np.sqrt([len(s) for s in SU_Vdiff_eps])
nn_eps = [np.abs(delta_V_nn[np.where(eps==val)]) for val in eps_vals]
mean_nn = np.mean(nn_eps,axis = -1)
std_nn = np.std(nn_eps,axis =-1)
ste_nn = std_nn/np.sqrt([len(s) for s in nn_eps])




eps_windowed = sliding_window_view(eps_vals,window)[::step]
mean_SO_windowed = sliding_window_view(mean_SO,window)[::step]
mean_SU_windowed = sliding_window_view(mean_SU,window)[::step]
#mean_arb_windowed = sliding_window_view(mean_arb,window)[::step]

#mean_SO_network_windowed = sliding_window_view(mean_SO_network,window)[::step]
mean_SU_network_windowed = sliding_window_view(mean_nn,window)[::step]
#mean_arb_network_windowed = sliding_window_view(mean_arb_network,window)[::step]


std_error_SO_windowed = sliding_window_view(std_error_SO,window)[::step]
std_error_SU_windowed = sliding_window_view(std_error_SU,window)[::step]
#std_error_arb_windowed = sliding_window_view(std_error_arb,window)[::step]

#std_error_SO_network_windowed = sliding_window_view(std_error_SO_network,window)[::step]
std_error_SU_network_windowed = sliding_window_view(mean_nn,window)[::step]
#td_error_arb_network_windowed = sliding_window_view(std_error_arb_network,window)[::step]

p0 = np.zeros(6)

SO_fits=np.zeros((eps_windowed.shape[0],len(p0)))
SU_fits=np.zeros((eps_windowed.shape[0],len(p0)))
arb_fits=np.zeros((eps_windowed.shape[0],len(p0)))

SO_network_fits=np.zeros((eps_windowed.shape[0],len(p0)))
SU_network_fits=np.zeros((eps_windowed.shape[0],len(p0)))
arb_network_fits=np.zeros((eps_windowed.shape[0],len(p0)))


SO_covs=np.zeros((eps_windowed.shape[0],len(p0),len(p0)))
SU_covs=np.zeros((eps_windowed.shape[0],len(p0),len(p0)))


SU_network_covs=np.zeros((eps_windowed.shape[0],len(p0),len(p0)))


bound = (-.001,10)
def p5(x,a0,a1,a2,a3,a4,a5):
    return a0+x*a1+a2*x**2+a3*x**3+a4*x**4+a5*x**5

for i,these_eps_vals in enumerate(eps_windowed):

    SO_fit_params,SO_fit_cov=fit(p5,these_eps_vals,mean_SO_windowed[i],p0=p0,
                                                      sigma=std_error_SO_windowed[i],absolute_sigma=True,bounds=bound)
    SU_fit_params,SU_fit_cov=fit(p5,these_eps_vals,mean_SU_windowed[i],p0=p0,
                                                      sigma=std_error_SU_windowed[i],absolute_sigma=True,bounds=bound)
    
    
    SU_network_fit_params,SU_network_fit_cov=fit(p5,these_eps_vals,mean_SU_network_windowed[i],
                                                                    p0=p0,sigma=std_error_SU_network_windowed[i],absolute_sigma=True,
                                                                    bounds=bound)
    

    SO_fits[i]=(SO_fit_params)
    SO_covs[i]=(SO_fit_cov)
    SU_fits[i]=(SU_fit_params)
    SU_covs[i]=(SU_fit_cov)
    

   
    SU_network_fits[i]=(SU_network_fit_params)
    SU_network_covs[i]=(SU_network_fit_cov)
  


# Now SU(3)
fig,ax = plt.subplots(1,figsize=(10,6))
colors = sb.color_palette('colorblind',4)
eps_mid=np.median(eps_windowed,axis=-1)

coeff = 1
y=np.array(SU_fits)[:,coeff]
ci = np.sqrt(np.array(SU_covs)[:,coeff,coeff])
ax.semilogx(eps_mid, y*np.power(eps_mid,0),label=r'SU(3)',color='red')
ax.fill_between(eps_mid, (y-ci)*np.power(eps_mid,0), (y+ci)*np.power(eps_mid,0), color='red', alpha=.1)


y=np.array(SO_fits)[:,coeff]
ci = np.sqrt(np.array(SO_covs)[:,coeff,coeff])
ax.semilogx(eps_mid, y*np.power(eps_mid,0),label=r'SO(8)',color='black')
ax.fill_between(eps_mid, (y-ci)*np.power(eps_mid,0), (y+ci)*np.power(eps_mid,0), color='black', alpha=.1)

y=np.array(SU_network_fits)[:,coeff]
ci = 0.05*np.sqrt(np.array(SU_network_covs)[:,coeff,coeff])
ax.semilogx(eps_mid, y*np.power(eps_mid,0),label=r'SU(3$)_{nn}$',color='purple')
ax.fill_between(eps_mid, (y-ci)*np.power(eps_mid,0), (y+ci)*np.power(eps_mid,0), color='purple', alpha=.1)
ax.set_xlabel('$\epsilon$')
ax.set_ylabel('a{}'.format(coeff))
ax.set_ylim(-1,1)
ax.set_xlim(0.01,10)
plt.legend()
plt.show()
fig.savefig('/home/sean/Documents/Work/Level 4/Level-4-Masters-Project/Figures/sliding_window_a_{}.png'.format(coeff),dpi = 300)