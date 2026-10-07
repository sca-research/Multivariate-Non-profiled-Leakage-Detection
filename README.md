## Multivariate-Non-profiled-Leakage-Detection
<!-- This repository contains the practical implementation of several multivariate leakage detection tests considered in the IACR-Tches 2026, Volume 2 paper titled **Multivariate Leakage Detection**. The research was conducted by the Cybersecurity research group at the University of Klagenfurt, Austria. -->

In this repository, we primarily focused on a comparative study of different non-profiled multivariate detection methods, along with multiplicity corrections for existing univariate detection methods.      

## General Introduction 
- The project contains the `Python` implementation of three multivariate leakage detection tests, namely, the distance covariance estimator (aka MV-dcov), the Diagonal $T$-test, and Hotelling's $T^2$. We have also considered Bonferroni's multiplicity correction techniques for four univariate leakage detection tests: Welch's $t$-test, the $\chi^2$-test, the mutual information-based $G$-test, and the distance correlation-based test of independence (aka dcor).  
- We have analysed both the False positive rate and the True positive rate of the aforementioned tests via p-value computation and then computing the statistical power of the tests for a certain number of iterations.  
User guide and detailed instructions of our `Python` implementations are provided in [Code](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/tree/main/Code) folder.

## Installation
Python 3.9.x is the tested version (validated with Python 3.9.25). The current pinned NumPy version is not compatible with Python 3.14; other Python versions have not been validated.

Clone the repository and move into its root directory:
```bash
git clone https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection.git
cd Multivariate-Non-profiled-Leakage-Detection
```

Create and activate a Conda environment, then install the repository requirements from the repository root:
```bash
conda create -n leakage-detection python=3.9
conda activate leakage-detection
python -m pip install -r requirements.txt
```

## Datasets
We have considered both simulated and practical case studies for our implementation.
- In simulation experiments, we have considered different linear leakage models, like hamming weight, hamming distance, weighted hamming weight, and one non-linear model (by considering the double permutation). Along with leakage models, we also consider the Gaussian and non-Gaussian additive noises. The multivariate leakage simulation is provided in the [trace_simulation.py](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/trace_simulation.py) Python script.
- We have considered a practical case study for the side-channel traces from an unprotected implementation of PRESENT block cypher as provided by [DL-LA](https://github.com/Chair-for-Security-Engineering/DL-LA). The download instructions for this dataset are available in [PRESENT-RC](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/tree/main/PRESENT-RC)   

## References
- Olivier Bronchain, Tobias Schneider, and François-Xavier Standaert. [“Multi-Tuple Leakage Detection and the Dependent Signal Issue.”](https://tches.iacr.org/index.php/TCHES/article/view/7394) *IACR Transactions on Cryptographic Hardware and Embedded Systems*, 2019(2), 318–345. DOI: [10.13154/tches.v2019.i2.318-345](https://doi.org/10.13154/tches.v2019.i2.318-345).
- Aakash Chowdhury and Elisabeth Oswald. [“Multivariate Leakage Detection.”](https://tches.iacr.org/index.php/TCHES/article/view/12890) *IACR Transactions on Cryptographic Hardware and Embedded Systems*, 2026(2), 296–324. DOI: [10.46586/tches.v2026.i2.296-324](https://doi.org/10.46586/tches.v2026.i2.296-324).

## Acknowledgement
This project is supported in part by the Austrian Science Fund (FWF) 10.55776/F85 (SFB SpyCode) and by the  EU Horizon project (enCrypton, grant agreement number 101079319).

![Spycode Logo](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/spycode.png)
![EU Logo](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/CERV-Acknowlegments.png)
