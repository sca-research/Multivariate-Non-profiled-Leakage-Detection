### REQUIREMENTS:
This code is written in Python 3.9.21 and is primarily based on Numpy 1.23.5, pandas 2.2.3, and Scipy 1.12, running on Linux or Windows. The required packages are:
- numpy==1.2.5
- scipy==1.12
- pandas==2.2.3
- [dcor==0.6](https://dcor.readthedocs.io/en/stable/apilist.html)
- [mpmath==1.3.0](https://mpmath.org/)
- [numba==0.60.0](https://numba.pydata.org/)
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
For the simulated leakage detection, please run:
```
python3 out_of_the_box_exp.py --exp simulated_exp
```
You will observe a comparison of the True positive rate between MV-dcov and the multiplicity correction for the classical TVLA (Welch's $T$-test) and $\chi^2$-test.
To use any subset of tests, please change the callable ``run_all_tests()``:
```
results = run_all_tests(
               Tr_random,
               Tr_fixed,
               n_dim = 50,
               enabled_tests=["mv_dcov", "tvla"],
                )
```
At present, we call only one multivariate test: ``"mv_dcov"`` and two univariate tests with multiplicity correction: `` "tvla" `` and `` "chi2" ``. Implementing `` "chi2" `` requires digitization of the continuous trace thus needed a separate ``run_all_tests()``:
```
results_ = run_all_tests(
               digit_1.digitize(Tr_random),
               digit_2.digitize(Tr_fixed),
               n_dim = 50,
               enabled_tests=["chi2"],
                )
```
The class ``Digitizer`` is defined in [testnbr_dist_1.py](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/testnbr_dist_1.py) .

You can call any subset of 8 tests from ``[ "mv_dcov", "hotelling", "diag", "mv_gtest", "tvla", "dcor", "chi2", "gtest"]``. This simulation experiment should be considered to reproduce figures **$1$ and $2$** of the paper.

**Figure $1$** is related to univariate tests, i.e., `` ["tvla", "dcor", "chi2", "gtest"] ``:
![Figure 1](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/Figure_1.png)

**Figure $2$** is related to multivariate tests. i.e. ``["mv_dcov", "hotelling", "diag", "mv_gtest"]``:
![Figure 2](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/Figure_2.png)


3. Leakage Detection on PRESENT-RC dataset:
- First, consider the data `` Traces_PRESENT_RC.npy `` from  [PRESENT-RC](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/tree/main/PRESENT-RC) folder
- For point-wise leakage detection, run:
```
python3 out_of_the_box_exp.py --exp present_pointwise
```
You can see, it will reproduce the figure **8a.** of the paper (see the attached figure at the bottom).

- For multivariate leakage detection (i.e., comparing True positive rates), run:
```
python3 out_of_the_box_exp.py --exp present_multivariate
```
Like experiment 2, you can make changes to the callable ``run_all_tests()`` to replicate our results, as given in figures **8b and 8c**. 
At present, we only run for the best (in terms of producing better true positive rate) multivariate test (i.e., the $D$-test, the red solid line in 8c), and the best univariate test ( $G$-test, the green dashed line in 8b).

Figure 8 in the paper is represented as follows:
![Figure 8a](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/Figure_8.png)

It is important to mention that this repository is limited only to the non-profiled leakage detection tests. To get the results corresponding to the Deep-net models, we recommend using the publicly available [DL-LA](https://github.com/Chair-for-Security-Engineering/DL-LA?tab=readme-ov-file) git repository. 
