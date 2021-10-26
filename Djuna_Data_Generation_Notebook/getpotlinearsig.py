
import random
import numpy as np
import scipy as sp
import linearsigma as cL

N=3
gens = cL.gen_gellman(3)

# Coefficients in the potential:
F0 = 1
m_Sigma_sq=1
lam=1
kap=1

pirandbase=1+1*np.random.rand(8)
    

def getpotential(i):
    sp.random.seed()
    random_pi = -pirandbase*0.5+pirandbase*np.random.rand(8)

    potval = np.real(cL.get_V(random_pi,gens,m_Sigma_sq, lam, kap))
    
    outp = np.hstack((random_pi,potval))
    
    return outp
