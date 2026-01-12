#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:00:57 2025

@author: IWAS\choaak
"""

from tqdm import tqdm

from MV_tests import *
from dcor.independence import distance_correlation_t_test
from dcor import u_distance_correlation_sqr
import matplotlib.pyplot as plt
from scalib.metrics import SNR
import h5py






def chisquare_cdf(x, k):
    # This is related to mpmath specification, only affect the mpmath related functions
    mp.dps = 25
    mp.pretty = True
    return gammainc(k/2, a=0, b=float(x/2))/np.exp(gammaln(k/2))






def simulated_exp():
    ''' Computing True Positive Rates for Simulated Multivariate Leakage Detection '''
        
    n_dim = 50
    ckey = 210  # Specify a fixed key to generate the 

    # sigma = 7.5 * math.sqrt(2)
    sigma = 12

    print("HW() leakage with Gaussian noise having SNR = {} and n_d = {}\n".format( 1 / (sigma ** 2), n_dim))
    # print("Non_lin() leakage with Gaussian noise having SNR = {} and n_d = {}\n".format(
    #     21.25 / (sigma ** 2), n_dim))

    # print("HW() Leakage with discrete laplacian noise having SNR = {}".format( 1 / ( 2 * (dlaplace.std(sigma) ** 2 ) ) ))
    # print("Non_lin() leakage with Gaussian noise having SNR = {} and n_d = {}\n".format( 21.25 / ( 2 * (dlaplace.std(sigma) ** 2 ) ), n_dim))
    
    
    x = [1, 0]

    n_trial = 200  # it should be increased upto 500 to avoid the variability

    z = []
    for n_trace in range(50, 250, 50):
        print("Number of Traces = {}".format(n_trace))
        count_1 = 0
        # count_2 = 0
        # count_3 = 0
        # count_MI = 0

        count_TVLA = 0
        # count_gtest = 0
        # count_chi_sqr = 0
        # count_dcor = 0

        V_vector = np.repeat(x, n_trace) ## 0-1 valued variable

        for m in tqdm(range(n_trial)):

            Tr_random = mv_trace(n_trace, n_dim, AES_Sbox,
                                 ckey, sigma, fix_input=False)
            Tr_fixed = mv_trace(n_trace, n_dim, AES_Sbox,
                                ckey, sigma, fix_input=True)

            # Tr = np.concatenate((Tr_fixed, Tr_random), dtype = np.float64)

            Tr = np.concatenate((Tr_fixed, Tr_random))

            # number of projections----------------------
            # k = int(n_trace/np.log(n_trace)) as a default we can consider k = 50 for n_trace in range(20, 160, 20)
            
            ############################## 
            '''Considering Multivariate Tests from MV_tests.py''' 
            # make sure input of the dcov metric is NxD and Nx1
            statistic_1, cutoff_1 = u_dist_cov_sqr_mv_test(Tr.astype('float'),
                                                           V_vector.astype('float'), n_projs = 100)
            # statistic_2, cutoff_2 = TwoSampleT2Test( 
            #    n_trace, n_dim, Tr_random, Tr_fixed)
            # statistic_3, cutoff_3 = diag_test(
            #    n_trace, n_dim, Tr_random,  Tr_fixed)
            # To use mv_gtest one have to Digitise the trace please see Digitizer class in testnbr_dist_1.py
            # statistic_4, cutoff_4 = mv_gtest(Tr, V_vector)    
            

            if statistic_1 > cutoff_1:
                count_1 += 1
            # if statistic_2 > cutoff_2:
            #    count_2 += 1
            # if statistic_3 > cutoff_3:
            #    count_3 += 1
            # if statistic_4 > cutoff_4:
            #     count_MI += 1



            '''Considering Multiplicity Corrections for leakage detections from MV_tests.py''' 
            if TVLA_adjusted_alpha(n_dim, np.array(Tr_random), np.array(Tr_fixed)) >= 1:
               count_TVLA += 1
            # if gtest_adjusted_alpha(n_dim, Tr, V_vector ) >= 1:
            #     count_gtest += 1
            # if chi_sqr_adjusted_alpha( n_dim, Tr, V_vector) >= 1:
            #     count_chi_sqr += 1
            # if dcor_adjusted_alpha(n_dim, Tr.astype('float'), V_vector.astype('float')) >= 1:
            #     count_dcor += 1
        
                
        
        ''' Computing True Positive rates for n_trial number of iterations '''
        count_1 = count_1 / n_trial
        # count_2 = count_2 / n_trial
        # count_3 = count_3 / n_trial
        # count_MI = count_MI / n_trial

        count_TVLA = count_TVLA / n_trial
        # count_gtest = count_gtest / n_trial
        # count_chi_sqr = count_chi_sqr / n_trial
        # count_dcor = count_dcor / n_trial

        print("\nTrue Positive Rate with mv_dcov_test = {}".format(count_1))
        # print("True Positive Rate with mv_gtest = {}".format(count_MI))
        # print("True Positive Rate with hotelling's $T^2$ test = {}".format(count_2))
        # print("True Positive Rate with D-test = {}".format(count_3))

        print(
           "True Positive Rate with TVLA + adjusted_pvalue  = {}".format(count_TVLA))
        # print("True Positive Rate with gtest + adjusted_pvalue  = {}".format(count_gtest))
        # print("True Positive Rate with chi_square + adjusted_pvalue  = {}".format(count_chi_sqr))
        # print("True Positive Rate with dcor + adjusted_pvalue  = {}".format(count_dcor))
        
        
        z.append([n_trace, count_1 , count_TVLA])
        # z.append([n_trace, count_1, count_2, count_3, count_MI])
        # z.append([n_trace, count_1, count_2, count_3])
        # z.append([n_trace, count_TVLA, count_gtest, count_chi_sqr, count_dcor])
        # z.append([n_trace, count_1, count_2, count_3, count_TVLA, count_dcor])
    
    
    ''' Saving the true positive rates w.r.t number of traces in a .npy file '''
    # np.save('Non_lin_MV(lin)_dLap_{}_{}.npy'.format(round(  21.25 / ( 2 * (dlaplace.std(sigma) ** 2 )) , 3), n_dim) , z)
    # np.save('Non_lin_Multi_test(lin)_dLap_{}_{}.npy'.format(round(  21.25 / ( 2 * (dlaplace.std(sigma) ** 2 )) , 3), n_dim) , z)
    np.save('HW_Norm_{}_{}.npy'.format(round(  1 / (sigma ** 2) , 3), n_dim) , z)
    # np.save('Non_lin_Norm_{}_{}.npy'.format(
    #     round(21.25 / (2 * (dlaplace.std(sigma) ** 2)), 3), n_dim), z)




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
    print(np.shape(PRESENT_random_clock))
    Traces = PRESENT_random_clock[:, :5000]
    zero_one_vector = PRESENT_random_clock[:, 5000]

    del PRESENT_random_clock

    n_points = int(5e03)
    
    
    n_trial = 50 # to replicate the exact experiment like paper please consider n_trial = 500

    z = []
    for n_trace in range(100, 1800, 200):
        print("Number of Traces = {}".format(n_trace))

        # count_1 = 0  # Related to Multivariate distance covariance
        count_2 = 0  # Related to D-test

        # count_TVLA = 0
        count_gtest = 0
        # count_chi_sqr = 0
        # count_dcor = 0
        for m in tqdm(range(n_trial)):
            
            ''' select random traces '''
            n_index = np.random.choice( len(Traces[:, 0]) , n_trace, replace = False)
            sub_traces = Traces[n_index]
            zero_one_vector_ = zero_one_vector[n_index]

            Tr_random = sub_traces[np.where( zero_one_vector_ == 1)]
            Tr_fixed = sub_traces[np.where( zero_one_vector_ == 0)]
            
            ''' Computing true positive rate  for multivariate testing methods'''
            # MVdcov test (For efficient performance of MVdcov please consider larger number of cpu cores)
            # statistic_1, cutoff_1  =  u_dist_cov_sqr_mv_test(sub_traces.astype('float'),
            #                                 zero_one_vector_.astype('float'), n_projs = 500)
            
            # D-test------------------------------------------------------------
            n_fix = len(Tr_random[:, 0])
            n_random = len(Tr_fixed[:, 0])
            statistic_2, cutoff_2  = diag_test(n_fix, n_random, n_points, Tr_random,  Tr_fixed)
            
            
            
            # if statistic_1 > cutoff_1: count_1 += 1
            
            if statistic_2 > cutoff_2: count_2 += 1

            ''' Computing True positive rate after Bonferonni's multiplicity corrrections'''
            # if TVLA_adjusted_alpha( n_points, np.array(Tr_random), np.array(Tr_fixed) ) >= 1:
            #    count_TVLA += 1
            if gtest_adjusted_alpha( n_points, sub_traces, zero_one_vector_ ) >= 1:
               count_gtest += 1
            # if chi_sqr_adjusted_alpha( n_points, sub_traces, zero_one_vector_) >= 1:
            #    count_chi_sqr += 1
            # if dcor_adjusted_alpha(n_points, sub_traces.astype('float'), zero_one_vector_.astype('float')) >= 1:
            #    count_dcor += 1



        # count_1 =  count_1 / n_trial
        count_2 = count_2 / n_trial
        
        
        
        # count_TVLA = count_TVLA / n_trial
        count_gtest = count_gtest / n_trial
        # count_chi_sqr = count_chi_sqr / n_trial
        # count_dcor = count_dcor / n_trial

        # print("True positive rate with multivariate-dcor Test = {}".format(count_1))
        print("True positive rate with D-test = {}".format(count_2))
        # -------------------------------------------------------------------------------------
        # print(" True positive rate. with multi. uni TVLA = {}".format(count_TVLA))
        print("True positive rate. with multi. uni gtest = {}".format(count_gtest))
        # print(" True positive rate. with multi. uni chi_sqr = {}".format(count_chi_sqr))
        # print(" True positive rate. with multi. uni dcor = {}".format(count_dcor))

        z.append([n_trace, count_2, count_gtest])
        # z.append([n_trace, count_TVLA, count_gtest, count_chi_sqr, count_dcor])
        # z.append([n_trace, count_dcor])
        pass

    np.save('MV_PRESENT.npy', z)




if __name__== '__main__':
    simulated_exp()
    # PRESENT_RC_pointwise()
    # PRESENT_RC_multivariate()
