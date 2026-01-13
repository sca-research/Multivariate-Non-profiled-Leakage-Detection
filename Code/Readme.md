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
- MV_tests.py: Contains all the multivariate tests and the multiplicity corrections(after adjusting the p-values via Bonferonni's correction)
- Leakage_Detection.py: This is the 'main' file to run all the leakage detection tests

### USAGE:
We are mainly examining three types of experiments: 
1. The Scalability Test of the parallel implementation of the multivariate distance covariance metric will be obtained by running:
```
python3 RP_dcor.py
```
Apart from scalability, the correctness of our parallel implementation of Mv-dcov was also obtained by uncommenting ``checking_correctness()`` main function inside RP_dcor.py. We have implemented the fast MV-dcov utilizing the published paper on [A Statistically and Numerically Efficient Independence Test Based on Random Projections and Distance Covariance](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2021.779841/full#supplementary-material).

2. Simulated Multivariate Leakage Detection:
Uncomment the ``simulated_exp()`` function at the last part of ``Leakage_Detection.py`` and then run:
```
python3 Leakage_Detection.py
```
You will observe a comparison in the True positive rate between MV-dcov and the multiplicity correction related to classical TVLA (Welch's $T$-test), representing **figures 1 and 2** of the published version. 

3. Leakage Detection on PRESENT-RC dataset:
- For point-wise leakage detection, uncomment 'PRESENT_RC_pointwise()' in 'Leakage_Detection.py' and then run:
```
python3 Leakage_Detection.py
```
- For multivariate leakage detection (i.e., comparing True positive rates), uncomment 'PRESENT_RC_multivariate()' in 'Leakage_Detection.py' and then run:
```
python3 Leakage_Detection.py
```
We only commented out a few among all the tests to run the experiments. For users, we welcome them to comment out the other tests to exactly replicate our results, as given in **Figure 8**. 
This repository is limited only to the non-profiled leakage detection tests. To get the results of Deep-net models, we recommend using the publicly available [DL-LA](https://github.com/Chair-for-Security-Engineering/DL-LA?tab=readme-ov-file) git repository. 
