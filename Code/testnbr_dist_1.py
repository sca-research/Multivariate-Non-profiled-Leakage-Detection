#!/usr/bin/python3
import numpy as np
import math
import random



from scipy.stats import dlaplace # simulation from discrete lapalce
from joblib import Parallel, delayed

# Sbox look-up tables
AES_Sbox = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]

DES_Sbox = [
    14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
    0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
    4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
    15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]

# Different Leakage Models on intermediate-----------------------------------------------------------
def lkg_6LSB(a):
    return a & 0x3f

def ham_wt(a):
    return bin(a).count("1")

def Non_lin(a):
    return DES_Sbox[lkg_6LSB(a)]


## Multivariate Leakage Simulation-------------------------------------------------------------------------------------
def mv_trace(n_trace, n_dim, sbox, ckey, sigma, fix_input = False):
    '''
    Parameters
    ----------
    n_trace: Number of traces required
    sbox : Subbyte operation based on lookup table
    ckey : Correct key (dtype= np.uint8)
    pt: n-bit plaintext
    sigma : standard deviation of error vector
    dim: dimension of the traces or the number of sample points

    Returns
    -------
    Simulated Multivariate Traces for fixed or random inrtermediate 
    '''
    
    ## We fix or randomized the 'plain_text(X)' (input) to generate the traces from fixed or random input 
    
    
    if fix_input == False:
        pt = np.random.randint(0, 256, n_trace, dtype = np.uint8)
    if fix_input == True:
        p_text = np.random.randint(0, 256, 1, dtype = np.uint8)
        pt = np.repeat(p_text, n_trace)
    
    round_key = np.random.randint(0, 256, n_dim-1, dtype = np.uint8)
    
           
    Tr = np.empty((n_trace, n_dim))

    for i in np.arange(0, n_trace):
        s_out = np.empty(n_dim, dtype = np.uint8)
        s_out[0] = sbox[pt[i] ^ ckey]
               
        Tr[i, 0] =  float(ham_wt(s_out[0])) + np.random.normal(0,sigma,1) 
        # Tr[i, 0] = float(Non_lin(s_out[0])) + np.random.normal(0,sigma,1)
        # Tr[i, 0] =  int(ham_wt(s_out[0])) + dlaplace.rvs(sigma, 1)                 
        # Tr[i, 0] =  int(Non_lin(s_out[0]) + dlaplace.rvs(sigma, 1))                 
        
        for j in range(1, n_dim):
            s_out[j] = sbox[s_out[j-1] ^ round_key[j-1]]
            # s_out[j] = s_out[j-1] ^ round_key[j-1]
            
            Tr[i, j] =  float( ham_wt(s_out[j])) + np.random.normal(0,sigma,1)
            # Tr[i, j] =  float( Non_lin(s_out[j])) + np.random.normal(0,sigma,1)
            # Tr[i, j] =  int(ham_wt(s_out[j])) + dlaplace.rvs(sigma,1)
            # Tr[i, j] =  int(Non_lin(s_out[j])) + dlaplace.rvs(sigma,1)
        del s_out
    return Tr



class Digitizer:
    """
        This class is used to make the trace discrete
    """
    def __init__(self,nbins,min,max):
        self._nbins = nbins
        self._min = min
        self._max = max
    def digitize(self,X):
        return np.digitize(np.minimum(X,self._max),np.linspace(self._min,self._max,self._nbins-1))


## MI plugin estimator for univarate and multivariate G test-------
def mi_plug_in(pred_leakage, ot):
    '''

    Parameters
    ----------
    pred_leakage : A list of univariate 'discrete' predicted leakage
    ot : A list of Univariate 'discrete' Observable Traces

    Returns
    -------
    Plug-in estimate of MI between two set of discrete data

    '''

    unique_ot, count_uni = np.unique( ot, return_counts = True )
    ot_prob = count_uni / np.sum(count_uni)
    del unique_ot , count_uni
    
    entrop_trc = -np.sum(ot_prob * np.log2(ot_prob))
    
    
    
    unique_Y, count_Y = np.unique(pred_leakage, return_counts = True)
    Y_prob = count_Y / np.sum(count_Y)
    
    all_cond_probs = []
    for i in unique_Y:        
        cond_tr = ot[pred_leakage == i]
        unique2, cond_tr_count = np.unique(cond_tr,  return_counts = True)
        all_cond_probs.append(cond_tr_count / np.sum(cond_tr_count))
        pass

    N = len(all_cond_probs)
    cond_entrps = [0]*N
    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(all_cond_probs[i]
                                          [j]) * all_cond_probs[i][j]
                pass
            pass
        pass


    # Vectorize above computation--------------------------------------
    cond_entrp = np.sum(cond_entrps * Y_prob)    
    return entrop_trc - cond_entrp


# Multidimensional plug-in MI estimator-----------------------------------------------------

def mi_plug_indd(pred_lkg, ot):
    '''
    Parameters
    ----------
    pred_lkg : An array of univariate Pred_leakage having N elements  
               (an arbitrary functional output of the intermediate value)
    ot : A 2-D array of multivariate 
                 Trace data ( N X D ) 

    Returns: MI plugin estimator of the multivariate discrete leakage and 
                  a univariate function of intermediate (discrete in nature)
    -------
    Note: This function is not applicable for non-discrete data

    '''
    unique_ot, count_multi = np.unique(ot, axis = 0, return_counts=True)
    
    ot_prob = count_multi / np.sum(count_multi)
    
    del count_multi, unique_ot
    
    entrop_trc = -np.sum(ot_prob * np.log2(ot_prob))
    
        
    unique_Y, count_Y = np.unique(pred_lkg, return_counts = True)
    Y_prob = count_Y / np.sum(count_Y)

    all_cond_probs = Parallel(n_jobs=-1)(delayed(calculate_cond_probs)(ot, pred_lkg, unique_Y, i) for i in unique_Y)


    N = len(all_cond_probs)
    cond_entrps = [0]*N
    for i in range(N):
        for j in range(len(all_cond_probs[i])):
            if(all_cond_probs[i][j] != 0):
                cond_entrps[i] -= np.log2(
                    all_cond_probs[i][j]) * all_cond_probs[i][j]
                pass
            pass            
    del all_cond_probs
    
    
    # Vectorize above computation--------------------------------------    
    cond_entrp = np.sum(cond_entrps * Y_prob)

    return entrop_trc - cond_entrp


def calculate_cond_probs(ot, pred_lkg, unique_Y, i):
    cond_ot = ot[ pred_lkg == i]
    unique_cond_ot, cond_ot_count = np.unique(cond_ot, axis=0, return_counts=True)
    return cond_ot_count / np.sum(cond_ot_count)


# ====================================================================================