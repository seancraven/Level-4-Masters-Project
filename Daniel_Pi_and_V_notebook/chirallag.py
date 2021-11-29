"""
Chiral lagrangian helper module, provides various functions to produce
generators for the chiral lagrangian, and also to obtain the potential, kinetic
and H values for arrays of pi, dpi/dx, a set of generators, and a given value of
F_0.
Main functions are:
- gen_gellman to obtain generators
- get_V, get_H and get_kinetic_term to obtain potential parameters.
"""

import numpy as np

_pauli_matrix = [[[0,1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]

def append_to(arry, entry):
    if len(arry)==0:
        arry = np.concatenate((arry,[entry]), axis=1)
    else:
        arry = np.concatenate((arry, [entry]), axis=0)
    return arry

def permute(oldlist, rule):
    newlist=np.copy(oldlist)
    for index in range(len(rule)):
        newlist[rule[index]-1]= oldlist[index]
    return newlist

def gen_perms(n):
    perms=[]
    for i in range(n-1):
        perm=[]
        for j in range(n):
            if j==0:
                perm = np.append(perm, i+1)
            elif j==1:
                perm = np.append(perm, n)
            elif j<=i+1:
                perm = np.append(perm, j-1)
            else:
                perm = np.append(perm, j)
        if len(perms)==0:
            perms = np.append(perms,perm, axis=0)
        else:
            perms = np.vstack([perms, perm])
    perms = perms.astype(int)
    return perms



def gen_gellman(n):
    baselist=np.copy(_pauli_matrix)
    if n > 2:
        gens = np.copy(gen_gellman(n-1))
        gens = np.pad(gens, ((0, 0), (0, 1), (0, 1)))
        perms = gen_perms(n)
        for perm in perms:
            for a in range(2):
                matrix=[[]]
                for entry in gens[a]:
                    if np.array_equal(matrix, [[]]):
                        matrix=np.concatenate((matrix, [permute(entry, perm)]), axis=1)
                    else:
                        matrix=np.concatenate((matrix, [permute(entry, perm)]), axis=0)
                gens=np.concatenate((gens,[permute(matrix, perm)]), axis=0)
        matrix=[[]]
        for q in range(n):
            row=[]
            for r in range(n):
                if q != r:
                    row=np.append(row,0)
                elif q==r and q+1 !=n:
                    row=np.append(row, 1)
                else:
                    row=np.append(row,-n + 1)
            if np.array_equal(matrix, [[]]):
                matrix=np.concatenate((matrix, [row]), axis=1)
            else:
                matrix=np.concatenate((matrix, [row]), axis=0)
        matrix=1. / np.sqrt(1/2*(n-1 + (n-1)**2))* matrix
        gens=np.concatenate((gens,[matrix]), axis=0)
    elif n <=2:
        gens = np.copy(baselist)
    return gens

def get_V(pi_fs,dpi_fs,gens,F_0):
    """
    Computes V given array of pi fields, array of dpi/dx fields, and array
    of generators.
    """
    if len(pi_fs.shape)>1:
        sum_string = 'aij,bjk,ckl,dli,na,nb,nc,nd->n'
        sum_arrays = [gens,gens,gens,gens,pi_fs,pi_fs,dpi_fs,dpi_fs]
        path = np.einsum_path(sum_string,*sum_arrays,optimize='optimal')[0]
        result = -np.einsum(sum_string,*sum_arrays,optimize=path)
        sum_string = 'aij,bjk,ckl,dli,nd,nb,na,nc->n'
        path = np.einsum_path(sum_string,*sum_arrays,optimize='optimal')[0]
        result += np.einsum(sum_string,*sum_arrays,optimize=path)
        result *= 1/(24*F_0**2)
    else:
        sum_string = 'aij,bjk,ckl,dli,a,b,c,d->'
        sum_arrays = [gens,gens,gens,gens,pi_fs,pi_fs,dpi_fs,dpi_fs]
        path = np.einsum_path(sum_string,*sum_arrays,optimize='optimal')[0]
        result = -np.einsum(sum_string,*sum_arrays,optimize=path)
        sum_string = 'aij,bjk,ckl,dli,d,b,a,c->'
        path = np.einsum_path(sum_string,*sum_arrays,optimize='optimal')[0]
        result += np.einsum(sum_string,*sum_arrays,optimize=path)
        result *= 1/(24*F_0**2)
    return result
def get_kinetic_term(dpi_fs):
    """Returns kinetic term given a set of dpi/dx fields."""
    return 0.5*np.sum(dpi_fs*dpi_fs,axis = -1)

def get_H(pi_fs,dpi_fs,gens,F_0):
    """
    Computes H given array of pi fields, array of dpi/dx fields, and array
    of generators.
    Note this calls both get_V and get_kinetic_term so better to just do this
    addition by hand if also computing those.
    """
    return get_V(pi_fs,dpi_fs,gens,F_0) + get_kinetic_term(dpi_fs)
