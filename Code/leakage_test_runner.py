#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leakage test runner---------------------------------
"""

from typing import Dict, List, Optional
import numpy as np

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
        
     ## To use mv_gtest one have to Digitise
     ## the trace please see Digitizer class in testnbr_dist_1.py  
     
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
