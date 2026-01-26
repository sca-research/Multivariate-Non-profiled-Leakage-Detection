#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 19:04:28 2026

@author: IWAS\choaak
"""

import numpy as np
from tqdm import tqdm

from testnbr_dist_1 import *
from leakage_test_runner import run_all_tests
from dcor import u_distance_correlation_sqr
from scipy.stats import  chi2, chi2_contingency,  ttest_ind
from pandas import crosstab
import matplotlib.pyplot as plt


## for PRESENT_RC_pointwise()
from mpmath import* 
from scipy.special import gammaln


def chisquare_cdf(x, k):
    # This is related to mpmath specification, only affect the mpmath related functions
    mp.dps = 25
    mp.pretty = True
    return gammainc(k/2, a=0, b=float(x/2))/np.exp(gammaln(k/2))



def simulated_exp():
    """
    Computing True Positive rates for Leakage Detection 
    on simulated traces based on small set of sets
    USING leakage_test_runner.py
    """

    n_dim = 50
    ckey = 210
    sigma = 14.14

    n_trial = 50


    z = []

    for n_trace in range(50, 250, 50):
        print(f"Number of Traces = {n_trace}")
        
        count_mv_dcov = 0
        count_tvla = 0
        count_chi2 = 0
        
        for _ in tqdm(range(n_trial)):

            Tr_random = mv_trace(
                n_trace, n_dim, AES_Sbox, ckey, sigma, fix_input=False)
            Tr_fixed = mv_trace(
                n_trace, n_dim, AES_Sbox, ckey, sigma, fix_input=True) 
            
            
            ## Discretize the continuous traces for chi2 and gtest (For HW model nbins = 9)
            digit_1 = Digitizer( 9, np.min(Tr_random), np.max(Tr_random))
            digit_2 = Digitizer( 9, np.min(Tr_fixed) , np.max(Tr_fixed))
           
            
        ##------------------------------------------------------------------
        ## Specify the subsets of methods in enabled_tests------------------
        ##------------------------------------------------------------------ 
            results = run_all_tests(
               Tr_random,
               Tr_fixed,
               n_dim = 50,
               enabled_tests=["mv_dcov", "tvla"],
                )
            
            ## Separate results for chi2, gtest, mvgtest (for discretized traces) 
            results_ = run_all_tests(
               digit_1.digitize(Tr_random),
               digit_2.digitize(Tr_fixed),
               n_dim = 50,
               enabled_tests=["chi2"],
                )
            
            
            ## Specify the result of leakage detection whether it detect the leaks or not
            if results["mv_dcov"]:
               count_mv_dcov += 1
            
            if results["tvla"]:    
                count_tvla += 1
                
            if results_["chi2"]:    
                count_chi2 += 1
            
            
        ## Computing Empirical True positive rate--------------
        count_mv_dcov /= n_trial
        count_tvla /= n_trial
        count_chi2 /= n_trial
                
        print(f"True Positive Rate with mv_dcov_test = {count_mv_dcov}")
        print(f"True Positive Rate with TVLA + adjusted_pvalue = {count_tvla}")
        print(f"True Positive Rate with chi_sqr + adjusted_pvalue = {count_chi2}")
        
            
        z.append([n_trace, count_mv_dcov, count_tvla, count_chi2])
        ###------------------------------------------------------------------------------
        
        
        
        # Same save semantics
    np.save(
        f"HW_Norm_{round(1 / (sigma ** 2), 3)}_{n_dim}.npy",
        z,
    )
    



def PRESENT_RC_pointwise():
    
    ''' Point-wise Leakage Detection of PRESENT-RC dataset '''
    
    # load PRESENT data (heavily misaligned)------------------------------------------------------

    PRESENT_random_clock = np.load('Traces_PRESENT_RC.npy')
    print(np.shape(PRESENT_random_clock))
    Traces = PRESENT_random_clock[:, :5000]
    zero_one_vector = PRESENT_random_clock[:, 5000]

    del PRESENT_random_clock

    n_points = int(5e03)
    n_traces = int(5e03)
    
    
    sub_traces = Traces[: n_traces]
    zero_one_vector_ = zero_one_vector[: n_traces]
    del Traces, zero_one_vector
    
    # # Producing traces corresponding Fix and Random inputs---------------------------------
    Trace_random = sub_traces[np.where(  zero_one_vector_ == 1)]
    Trace_fix = sub_traces[np.where( zero_one_vector_ == 0)]
    

    test_pvalues1 = np.empty(n_points)
    test_pvalues2 = np.empty(n_points)
    test_pvalues3 = np.empty(n_points)
    test_pvalues4 = np.empty(n_points)

    for i in tqdm(range(n_points)):

        # # using dcor-----------------------------------------------------------------------
        dcor_statistic =  (2 * n_traces) * u_distance_correlation_sqr( sub_traces[:, i].astype('float'),
                                            zero_one_vector_.astype('float'), method = "mergesort")
        test_pvalues1[i] = 1 - chi2.cdf(dcor_statistic, 1)
        del dcor_statistic

        # # using g-test---------------------------------------------------------------------
        D_MI_value = mi_plug_in(
            zero_one_vector_, sub_traces[:, i]) * np.log(2)  # for univariate
        d_f = (len(np.unique(sub_traces[:, i])) - 1) * \
                (len(np.unique(zero_one_vector_))-1)
        test_pvalues2[i] = 1 - chisquare_cdf(D_MI_value * 2 * n_traces, d_f)
        del D_MI_value

        # # using chi-square test------------------------------------------------
        freq_table = crosstab(sub_traces[:, i], zero_one_vector_)
        test_pvalues3[i] = chi2_contingency(freq_table).pvalue

        # using TVLA----------------------------------------------------
        test_pvalues4[i] = ttest_ind(Trace_random[:, i], Trace_fix[:, i], equal_var= False)[1]

        pass

    del sub_traces, zero_one_vector_
    
    
    # Adjusting for vary low p-values------------------------
    update_pvalues1 = np.where(
        test_pvalues1 > 1e-16, np.log10(np.abs(test_pvalues1)) * (-1), 10)
    update_pvalues2 = np.where(
        test_pvalues2 > 1e-16, np.log10(np.abs(test_pvalues2)) * (-1), 10)
    update_pvalues3 = np.where(
        test_pvalues3 > 1e-16, np.log10(np.abs(test_pvalues3)) * (-1), 10)
    update_pvalues4 = np.where(
        test_pvalues4 > 1e-16, np.log10(np.abs(test_pvalues4)) * (-1), 10)
    
    ''' Plotting the result '''
    
    ## Fig and plot specifications-----------------------------------------------
    TINY_SIZE = 5
    SMALL_SIZE = 5
    MEDIUM_SIZE = 5
    
    cm_ = 1 / 2.54 # For plotting within specific centimeters
    fig, axs = plt.subplots(4, 1, figsize = ( 7.5 * cm_, 9.2 * cm_ ), sharex=True)
    
    
    plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize = MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = TINY_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = TINY_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize = MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize = MEDIUM_SIZE)  # fontsize of the figure title
    
    
    
    # # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0.2)

    # Plot each graph, and manually set the y tick values---------------------------
    axs[0].plot(np.array(range(n_points)), update_pvalues1, '-', linewidth=0.3)
    # axs[0].set_ylabel(r'$-\log_{10}(p)$')
    axs[0].set_yticks(np.arange(0, 20, 4))
    axs[0].grid(linestyle='--', linewidth=0.5)
    axs[0].axhline(y=5, color='r', linestyle='-', linewidth=0.3, zorder=0)
    # axs[0].set_ylim(-1, 1)
    # ------------
    axs[1].plot(np.array(range(n_points)),
                update_pvalues2, '-', linewidth=0.3)
    # axs[1].set_ylabel(r'$-\log_{10}(p)$')
    axs[1].axhline(y=5, color='r', linestyle='-', linewidth=0.3, zorder=0)
    axs[1].set_yticks(np.arange(0, 8, 3))
    axs[1].grid(linestyle='--', linewidth=0.5)
    # ------------
    axs[2].plot(np.array(range(n_points)),
                update_pvalues3, '-', linewidth=0.3)
    # axs[2].set_ylabel(r'$-\log_{10}(p)$')
    axs[2].axhline(y=5, color='r', linestyle='-', linewidth=0.3, zorder=0)
    axs[2].set_yticks(np.arange(0, 8, 3))
    axs[2].grid(linestyle='--', linewidth=0.5)
    # ------------
    axs[3].plot(np.array(range(n_points)), update_pvalues4, '-', linewidth=0.3)
    axs[3].set_yticks(np.arange(0, 12, 4))
    axs[3].axhline(y=5, color='r', linestyle='-', linewidth=0.3, zorder=0)
    axs[3].grid(linestyle='--', linewidth=0.5)
    fig.supylabel(r'$-\log_{10}(p)$')
    fig.supxlabel("Trace points")

    plt.show()




def PRESENT_RC_multivariate():
    ''' Multivariate Leakage Detection for PRESENT-RC data ( by exactly following the steps as in simulated_exp() ) '''
    
    # load PRESENT data (heavily misaligned)------------------------------------------------------

    PRESENT_random_clock = np.load('Traces_PRESENT_RC.npy')
    Traces = PRESENT_random_clock[:, :5000]
    zero_one_vector = PRESENT_random_clock[:, 5000]

    del PRESENT_random_clock

    n_points = int(5e03)
    
    # to replicate the exact experimental outcome like paper please consider n_trial = 500
    n_trial = 50 

    z = []
    for n_trace in range(100, 1800, 200):
        print("Number of Traces = {}".format(n_trace))

        count_diag = 0  # Related to D-test
        count_gtest = 0
       
        for m in tqdm(range(n_trial)):
            
            ''' select random traces '''
            n_index = np.random.choice( len(Traces[:, 0]) , n_trace, replace = False)
            sub_traces = Traces[n_index]
            zero_one_vector_ = zero_one_vector[n_index]

            Tr_random = sub_traces[np.where( zero_one_vector_ == 1)]
            Tr_fixed = sub_traces[np.where( zero_one_vector_ == 0)]

            
            ##------------------------------------------------------------------
            ## Specify the subsets of methods in enabled_tests------------------
            ##------------------------------------------------------------------ 
            results = run_all_tests(
                   Tr_random,
                   Tr_fixed,
                   n_dim = n_points,
                   enabled_tests=["diag", "gtest"],
                    )
                
                
                
            ## Specify the result of leakage detection whether it detect the leaks or not
            if results["diag"]:
                count_diag += 1
                
            if results["gtest"]:    
                count_gtest += 1

        count_diag = count_diag / n_trial
        count_gtest = count_gtest / n_trial

        
        print("True positive rate with D-test (Diagonal T-test) = {}".format(count_diag))
        # -------------------------------------------------------------------------------------
        print("True positive rate. with multi. uni gtest = {}".format(count_gtest))
        

        z.append([n_trace, count_diag, count_gtest])
        pass

    np.save('MV_Leakage_Detection_PRESENT.npy', z)






if __name__== '__main__':
    simulated_exp()
    # PRESENT_RC_pointwise()
    # PRESENT_RC_multivariate()