#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:14:15 2024

@author: IWAS\choaak

Multivariate Leakage Detection Methods-------------------------------------------------------
"""

from  testnbr_dist_1 import *

from scipy.stats import  chi2, f
from scipy.stats import ttest_ind, chi2_contingency


from RP_dcor import* # For dcor_adjusted_alpha
from pandas import crosstab # For making frequency table for chi-square test


# Diagonal Test (D-test) -------------------------------------------------------------
def diag_test( n_x, n_y, p, X, Y, alpha = 0.05):
    '''
    Parameters
    ----------
    n_trace : Number of traces 
    p : Number of dimensions (of X and Y)
    X : a 2-D array (of shape: n_x x p)
    Y : a 2-D array (of shape: n_y x q)
    alpha : float TYPE, the assumed False Positive value
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    D : Diagonal t-test statistic from Multi-tuple leakage detection paper (https://tches.iacr.org/index.php/TCHES/article/view/7394/6566)
    cutoff : Upper alpha point of chi-square distribution

    '''
        
    delta = (np.mean( X, axis = 0) - np.mean(Y, axis = 0)) ** 2
    var_x = np.var(X, axis = 0) / n_x
    var_y = np.var(Y, axis = 0) / n_y

    
    D = 0
    for i in range(p):
        denom = var_x[i]  + var_y[i]

        D += delta[i] / denom 
    
    chi_ = chi2(df = p, loc = 0, scale = 1) 
    cutoff = chi_.ppf(1-alpha)
    
    return D, cutoff 


# Hotelling's T-square test ------------------------------------------------------------------------
def TwoSampleT2Test(n_x, n_y, p, X, Y, alpha = 0.05):
    '''
    Parameters
    ----------
    n_trace : Number of traces 
    p : Number of dimensions (of X and Y)
    X : a 2-D array (of shape: n_x x p)
    Y : a 2-D array (of shape: n_y x q)
    alpha : float TYPE, the assumed False Positive value
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    D : Hotelling's T-square test statistic (https://tches.iacr.org/index.php/TCHES/article/view/7394/6566)
    cutoff : Upper alpha point of chi-square distribution

    '''

    
    delta = np.mean(X, axis = 0) - np.mean(Y, axis = 0)
    Sx = np.cov(X, rowvar = False)
    Sy = np.cov(Y, rowvar = False)
    S_pooled = (( n_x - 1) * Sx + ( n_y - 1) * Sy) / (n_x + n_y - 2)

    t_squared = (n_x * n_y) / (n_x + n_y) * ((delta.T @ np.linalg.inv(S_pooled)) @ delta)
    statistic = t_squared * (n_x + n_y - p - 1)/(p * (n_x + n_y - 2))
    F = f(p, n_x + n_y - p - 1)
    
    # p_value = 1 - F.cdf(statistic)
    cutoff = F.ppf( 1 - alpha)

    return statistic, cutoff



# Multivariate g-test---------------------------------------------------------------------------
def mv_gtest(X, Y, alpha = 0.05):
    '''
    Parameters
    ----------
    X : An NxD array, D>1
    Y : An Nx1 array 
    alpha : level of significance

    Returns
    -------
    MI_based test statistic
    cutoff of upper alpha point of chi_square distribution

    '''
    
    n_trace = len(Y)
        
    D_MI_value = mi_plug_indd(Y, X) * np.log(2)
    
    unique_X = np.unique(X, axis = 0)
    
    unique_Y = np.unique(Y)   
    d_f = (len(unique_X.flatten()) - 1)  * (len(unique_Y) - 1)
    
    MI_cutoff = chi2.ppf(1- alpha, d_f) / ( 2 * n_trace )

    return D_MI_value, MI_cutoff


# Multiplicity Correction------------------------------------------- 
# TVLA after adjusting the level of significance using Bonferonni's correction------------------------------
def TVLA_adjusted_alpha( p, X, Y, alpha = 0.05):
    '''
    Parameters
    ----------
    p: dimension X
    X : An NxD array
    Y : An NxD array
    alpha : level of significance

    Returns
    -------
    count of rejected null hypotheses (count of p-values < the alpha_adjusted value)

    ''' 
    
    alpha_new = alpha / p
    
    p_values = np.empty(p)
    count  = 0
    for i in range(p):
        
        p_values[i] = ttest_ind(X[:,i], Y[:,i], equal_var= False)[1]
        
        if p_values[i] < alpha_new:
            count += 1               
            pass
        pass
        
    return  count  


# dcor after ajusting the level of significance using Bonferonni's correction-----------------------------
def dcor_adjusted_alpha(p, X, Y, alpha_ = 0.05):
    '''
    Parameters
    ----------
    p: dimension X
    X : An NxD array
    Y : An Nx1 array (vector of zeros and ones)
    alpha : level of significance 

    Returns
    -------
    count of rejected null hypotheses 

    '''
    
    n_samp = len(Y)
    alpha_new = alpha_ / p
    p_values = np.empty(p)
    count_dcorr = 0
    
    cut_off = float(chi2.ppf(1 - alpha_new, df = 1) - 1)
    
    for i in range(p):
        dcor_statistic = n_samp * u_distance_correlation_sqr( X[:, i], Y, method = 'mergesort')        
        if dcor_statistic > cut_off:
            count_dcorr += 1
            pass
        pass
    return count_dcorr



# chi-square test after ajusting the level of significance using Bonferonni's correction-------------------
def chi_sqr_adjusted_alpha( p, X, Y, alpha_ = 0.05):
    '''
    Parameters
    ----------
    p: dimension X
    X : An NxD array
    Y : An Nx1 array (vector of zeros and ones)
    alpha : level of significance 

    Returns
    -------
    count of rejected null hypotheses 

    '''
    
    alpha_new = alpha_ / p
    p_values = np.empty(p)
    count_chi_sqr = 0
    
    for i in range(p):
        freq_table = crosstab(X[:, i], Y)
        if chi2_contingency(freq_table).pvalue < alpha_new:
            count_chi_sqr += 1
            pass
        pass
    return count_chi_sqr


# MI-based g-test after ajusting the level of significance using Bonferonni's correction-------------------
def gtest_adjusted_alpha(p, X, Y, alpha_ = 0.05):
    '''
    Parameters
    ----------
    p: dimension X (or number of trace points)
    X : An NxD array
    Y : An Nx1 array (vector of zeros and ones)
    alpha : level of significance 

    Returns
    -------
    count of rejected null hypotheses 

    '''
    
    n_samp = len(Y)
    alpha_new = alpha_ / p
 
      
    Y_length =  len(np.unique(Y))
    
    
    count_gtest = 0 
    for i in range(p):
        D_MI_value = mi_plug_in( Y, X[:, i]) * np.log(2) 
        
        d_f_ = (len(np.unique(X[:, i])) - 1)  * (Y_length - 1)      
        
        chi2_upper_alpha = chi2.ppf( 1 - alpha_new , d_f_)
        
        if   D_MI_value >  float( chi2_upper_alpha / (2 * n_samp)) :
            count_gtest += 1
            pass
        del D_MI_value, d_f_
        pass
    return count_gtest