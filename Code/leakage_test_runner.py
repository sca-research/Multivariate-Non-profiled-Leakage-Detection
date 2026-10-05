#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leakage test runner---------------------------------
"""

from typing import Dict, List, Optional
import numpy as np
from joblib import Parallel, delayed

from MV_tests import (
    TVLA_adjusted_alpha,
    TwoSampleT2Test,
    diag_test,
    mv_gtest,
    dcor_adjusted_alpha,
    chi_sqr_adjusted_alpha,
    gtest_adjusted_alpha,
)

from RP_dcor import u_dist_cov_sqr_mv_test




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



def _build_Tr_and_V_exact(Tr_fixed: np.ndarray, Tr_random: np.ndarray):
    """
    Construct pooled traces for tests of independence, e.g. dcor, chi_sqr, gtest, etc. 

        Tr = concatenate((Tr_fixed, Tr_random))
        V  = repeat([1, 0], n_trace)

    This function exists ONLY to prevent future semantic drift.
    """
    
    if Tr_fixed.shape[0] == Tr_random.shape[0]:
        n_trace = Tr_fixed.shape[0] 
        V = np.repeat([1, 0], n_trace)
        pass
    else:
        V = np.repeat([1, 0], ( Tr_fixed.shape[0], Tr_random.shape[0]  )  ) 
    
    Tr = np.concatenate((Tr_fixed, Tr_random), axis=0)
    
    
    return Tr, V


# =============================================================================
# Public API
# =============================================================================

def run_all_tests(
    Tr_random: np.ndarray,
    Tr_fixed: np.ndarray,
    *,
    n_dim: int,
    enabled_tests: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Leakage Detection Tests 
    (4 Multivariate tests and 4 Univariate tests with multiplicity corrections)
    
    4 Multivariate tests: "mv_dcov", "hotelling", "diag", "mv_gtest"
    4 Univariate tests: "tvla", "dcor", "chi2", "gtest"
    
    
    Parameters
    ----------
    Tr_random : (N x D) ndarray
    Tr_fixed  : (N x D) ndarray
    n_dim     : int
        MUST be the experiment dimension (same value passed in simulated_exp)
    enabled_tests : list[str]
        Subset of tests to run

    Returns
    -------
    Dict[str, bool]
        test_name -> reject H0
    """

    if enabled_tests is None:
        enabled_tests = []

    results: Dict[str, bool] = {}

    # -------------------------------------------------------------------------
    # Multivariate distance covariance (RP-dCov)
    # -------------------------------------------------------------------------
    if "mv_dcov" in enabled_tests:
        Tr, V = _build_Tr_and_V_exact(Tr_fixed, Tr_random)

        stat, cutoff = u_dist_cov_sqr_mv_test(
            Tr.astype(float),
            V.astype(float),
            n_projs=100,
        )

        results["mv_dcov"] = stat > cutoff

    # -------------------------------------------------------------------------
    # TVLA with Bonferroni correction (STRICT)
    # -------------------------------------------------------------------------
    if "tvla" in enabled_tests:
        results["tvla"] = (
            TVLA_adjusted_alpha(
                n_dim,
                np.array(Tr_random),
                np.array(Tr_fixed),
            ) >= 1
        )

    # -------------------------------------------------------------------------
    # OPTIONAL TESTS (safe but disabled by default)
    # -------------------------------------------------------------------------
    if "hotelling" in enabled_tests:
        n_1 = Tr_random.shape[0]
        n_2 = Tr_fixed.shape[0]
        stat, cutoff = TwoSampleT2Test(n_1, n_2, n_dim, Tr_random, Tr_fixed)
        results["hotelling"] = stat > cutoff

    if "diag" in enabled_tests:
        n_1 = Tr_random.shape[0]
        n_2 = Tr_fixed.shape[0]
        stat, cutoff = diag_test(n_1, n_2, n_dim, Tr_random, Tr_fixed)
        results["diag"] = stat > cutoff
        
     ## To use mv_gtest one have to Digitise the trace 
     ## please see Digitizer class in trace_simulation.py  
     
    if "mv_gtest" in enabled_tests:  
        Tr, V = _build_Tr_and_V_exact(Tr_fixed, Tr_random)
        stat, cutoff = mv_gtest(Tr, V)
        results["mv_gtest"] = stat > cutoff

    if "dcor" in enabled_tests:
        Tr, V = _build_Tr_and_V_exact(Tr_fixed, Tr_random)
        results["dcor"] = dcor_adjusted_alpha(n_dim, Tr.astype(float), V.astype(float)) >= 1

    if "chi2" in enabled_tests:
        Tr, V = _build_Tr_and_V_exact(Tr_fixed, Tr_random)
        results["chi2"] = chi_sqr_adjusted_alpha(n_dim, Tr, V) >= 1
        
    if "gtest" in enabled_tests:
        Tr, V = _build_Tr_and_V_exact(Tr_fixed, Tr_random)
        results["gtest"] = gtest_adjusted_alpha(n_dim, Tr, V) >= 1

    return results
