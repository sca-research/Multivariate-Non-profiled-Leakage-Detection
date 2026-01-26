### REQUIREMENTS:
This code is written in Python 3.9.21 and is mainly based on Numpy = 1.23.5, pandas = 2.2.3, and Scipy 1.12 running on a Linux or Windows distribution. The required packages are:
- numpy
- scipy
- pandas
- [dcor](https://dcor.readthedocs.io/en/stable/apilist.html)
- [mpmath](https://mpmath.org/)
- [numba >= 0.51](https://numba.pydata.org/)
- matplotlib
- tqdm
  
These ``Python`` modules can be installed by running: 
```
sudo pip install numpy scipy pandas dcor mpmath numba matplotlib tqdm
``` 
### FILES DESCRIPTION:
- RP_dcor.py: Contains the Mv-dcov computation and the corresponding test of independence.
- testnbr_dist_1.py: Contains the Multivariate trace generation and the MI "plug-in" estimators (for implementing $G$-test).
- MV_tests.py: All multivariate tests and the multiplicity corrections are (after adjusting the p-values via Bonferroni's correction) defined here.
- leakage_test_runner.py: For out-of-the-box implementation, it is considered so that any subsets of tests are callable.
- out_of_the_box_exp.py: This is ``main`` file for leakage detection.  

### USAGE:
We are mainly examining three types of experiments: 
1. The Scalability Test of the parallel implementation of the multivariate distance covariance metric will be obtained by running:
```
python3 RP_dcor.py
```
Apart from scalability, the correctness of our parallel implementation of Mv-dcov was verified by uncommenting the ``checking_correctness()`` main function in RP_dcor.py. We have implemented the fast MV-dcov utilizing the published paper on [A Statistically and Numerically Efficient Independence Test Based on Random Projections and Distance Covariance](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2021.779841/full#supplementary-material).

2. Simulated Multivariate Leakage Detection:
Uncomment the ``simulated_exp()`` function at the last part of ``out_of_the_box_exp.py`` and then run:
```
python3 out_of_the_box_exp.py
```
You will observe a comparison of the True positive rate between MV-dcov and the multiplicity correction for the classical TVLA (Welch's $T$-test), as shown in **figures 1 and 2** of the published version. 
To use any subset of tests, please change the callable ``run_all_tests()``:
```
results = run_all_tests(
               Tr_random,
               Tr_fixed,
               n_dim = 50,
               enabled_tests=["mv_dcov", "tvla"],
                )
```
At present, we call only one multivariate test: ``"mv_dcov"`` and one univariate test with multiplicity correction: `` "tvla" ``. You can call all 8 tests from ``leakage_test_runner.py``.

3. Leakage Detection on PRESENT-RC dataset:
- First, consider the data `` Traces_PRESENT_RC.npy `` from  [PRESENT-RC](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/tree/main/PRESENT-RC) folder
- For point-wise leakage detection, uncomment ``PRESENT_RC_pointwise()`` in ``out_of_the_box_exp.py`` and then run:
```
python3 out_of_the_box_exp.py
```
- For multivariate leakage detection (i.e., comparing True positive rates), uncomment ``PRESENT_RC_multivariate()`` in ``out_of_the_box_exp.py`` and then run:
```
python3 out_of_the_box_exp.py
```
Like experiment 2, you can make changes to the callable ``run_all_tests()`` to replicate our results, as given in **Figure 8**. 
This repository is limited only to the non-profiled leakage detection tests. To get the results of Deep-net models, we recommend using the publicly available [DL-LA](https://github.com/Chair-for-Security-Engineering/DL-LA?tab=readme-ov-file) git repository. 
