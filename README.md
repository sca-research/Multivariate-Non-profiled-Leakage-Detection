## Multivariate-Non-profiled-Leakage-Detection
This repository contains the practical implementation of several multivariate leakage detection tests considered in the IACR-Tches 2026, Volume 2 paper titled **Multivariate Leakage Detection**. The research was conducted by the Cybersecurity research group at the University of Klagenfurt, Austria. In this repository, we primarily focused on a comparative study of different non-profiled multivariate detection methods, along with multiplicity corrections for existing univariate detection methods.      

## General Introduction 
- The project contains the `Python` implementation of three multivariate leakage detection tests, namely, the distance covariance estimator (aka MV-dcov), the Diagonal $T$-test, and Hotelling's $T^2$. We have also considered Bonferroni's multiplicity correction techniques for four univariate leakage detection tests, namely Welch's $t$--test, $\chi^2$-test, mutual information-based $G$-test, and distance correlation-based test of independence (aka dcor).  
- We have analysed both the False positive rate and the True positive rate of the aforementioned tests via p-value computation and then computing the statistical power of the tests for a certain number of iterations.  
User guide and detailed instructions of our `Python` implementations are provided in [Code](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/tree/main/Code) folder.

## Datasets
We have considered both simulated and practical case studies for our implementation.
- In simulation experiments, we have considered different linear leakage models, like hamming weight, hamming distance, weighted hamming weight, and one non-linear model (by considering the double permutation). Along with leakage models, we also consider the Gaussian and non-Gaussian additive noises. The multivariate leakage simulation is provided in [testnbr_dist_1.py](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/testnbr_dist_1.py) Python script.
- We have considered a practical case study for the side-channel traces from an unprotected implementation of PRESENT block cipher as provided by [DL-LA](https://github.com/Chair-for-Security-Engineering/DL-LA). The download instructions for this dataset are available in [PRESENT-RC](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/tree/main/PRESENT-RC)   

# Acknowledgment
This project is supported in part by the Austrian Science Fund (FWF) 10.55776/F85 (SFB SpyCode) and by the  EU Horizon project (enCrypton, grant agreement number 101079319).

![Spycode Logo](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/spycode.png)
![EU Logo](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/CERV-Acknowlegments.png)
