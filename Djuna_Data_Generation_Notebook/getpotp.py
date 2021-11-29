
import random
import numpy as np
import scipy as sp
import chirallag as cL

N=3
gens = cL.gen_gellman(3)
F0 = 1

pirandbase=1+1*np.random.rand(8)
    

def getpotential(i):
    sp.random.seed()
    random_pi = -pirandbase*0.5+pirandbase*np.random.rand(8)
    random_p = (-1+2*np.random.normal(0,.1,8))
    random_dpi = random_pi*random_p

    potval = np.real(cL.get_V(random_pi,random_dpi,gens,F0))
    
    outp = np.hstack((random_p,random_pi,potval))
    
    return outp
