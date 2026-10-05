### REQUIREMENTS AND SETUP:
Python 3.9.x is the tested version (validated with Python 3.9.25). Install the pinned dependencies from the repository root in a Conda environment:
```
conda create -n leakage-detection python=3.9
conda activate leakage-detection
python -m pip install -r requirements.txt
``` 
The dependency versions are maintained in the repository-root `requirements.txt`. 
### FILES DESCRIPTION:
- RP_dcor.py: Contains the Mv-dcov computation and the corresponding test of independence.
- trace_simulation.py: Contains the Multivariate trace generation and the `Digitizer`.
- MV_tests.py: All multivariate tests and the multiplicity corrections are (after adjusting the p-values via Bonferroni's correction) defined here.
- leakage_test_runner.py: Also contains the MI "plug-in" estimators (`mi_plug_in`, `mi_plug_indd`, for implementing $G$-test). For out-of-the-box implementation, it is considered so that any subsets of tests are callable.
- out_of_the_box_exp.py: This is ``main`` file for leakage detection.  

### USAGE:
We are mainly examining three types of experiments: 
1. The **scalability** test of the parallel implementation of the multivariate distance covariance (**MV-dcov**) checks both **correctness** and **runtime**. From the `Code` folder, run:
```
python RP_dcor.py --exp runtime
```
To check correctness of the parallel MV-dcov implementation, run:
```
python RP_dcor.py --exp correctness
```

Here, We have implemented the fast MV-dcov utilising the published paper on [A Statistically and Numerically Efficient Independence Test Based on Random Projections and Distance Covariance](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2021.779841/full#supplementary-material). 

The snippet of the **run-time** scalability test of MV-dcov:

<div style="height:300px; width:500px; overflow:auto; border:1px solid #ccc;">
  <img 
    src="https://raw.githubusercontent.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/main/Code/Screenshot_run_time.png"
    style="min-width:700px; min-height:800px;"
  >
</div>

2. Simulated Multivariate Leakage Detection:
From the `Code` folder, run the simulated leakage detection with:
```
python out_of_the_box_exp.py --exp simulated_exp
```
The defaults are the `hamming_weight` leakage model, `gaussian` noise, and `sigma=14.14`. Choose other built-in models from the command line:
```
python out_of_the_box_exp.py --exp simulated_exp --leakage-model nonlinear --noise-model discrete_laplace --sigma 14.14
```
Leakage choices are `hamming_weight` and `nonlinear` (the DES S-box output); noise choices are `gaussian`, `laplace` (continuous Laplace with scale `sigma/√2`, so `sigma` is the standard deviation), `discrete_laplace`, and `none`. For custom models, pass a callable as `leakage_model` to `mv_trace`; it receives one intermediate value. A custom `noise_model` callable receives `(shape, sigma)` and must return an array with that shape. The result filename records the selected leakage model, noise model, and sigma.
```
import numpy as np
from trace_simulation import AES_Sbox, mv_trace

def custom_leakage(value):
  return bin(value).count("1")

def custom_noise(shape, sigma):
  return np.random.uniform(-sigma, sigma, size=shape)

traces = mv_trace(
  100, 10, AES_Sbox, 210, 0.5,
  leakage_model=custom_leakage,
  noise_model=custom_noise,
)
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
The class ``Digitizer`` is defined in [trace_simulation.py](https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/trace_simulation.py) .

Each simulated run saves results to `simulated_<leakage-model>_<noise-model>_sigma_<sigma>_<dimension>.npy`. The tracked [simulated_hamming_weight_gaussian_sigma_14.14_50.npy](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/simulated_hamming_weight_gaussian_sigma_14.14_50.npy) is a generated Hamming-weight/Gaussian result (`sigma=14.14`). The snippet of that result is shown below:

<div style="height:300px; width:500px; overflow:auto; border:1px solid #ccc;">
  <img 
    src="https://raw.githubusercontent.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/main/Code/Screenshot_simulated_exp.png"
    style="min-width:700px; min-height:800px;"
  >
</div>


You can call any subset of 8 tests from ``[ "mv_dcov", "hotelling", "diag", "mv_gtest", "tvla", "dcor", "chi2", "gtest"]`` and be able to reproduce figures **$1$ and $2$**.

**Figure $1$** is related to univariate tests, i.e., `` ["tvla", "dcor", "chi2", "gtest"] ``:
![Figure 1](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/Figure_1.png)

**Figure $2$** is related to multivariate tests. i.e. ``["mv_dcov", "hotelling", "diag", "mv_gtest"]``:
![Figure 2](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/Figure_2.png)


3. Leakage Detection on PRESENT-RC dataset:
- After downloading and unpacking the dataset as described in `PRESENT-RC/Readme.md`, generate `Traces_PRESENT_RC.npy` from the `PRESENT-RC` directory:
```
cd ../PRESENT-RC && python DUT.py
```
- Run point-wise leakage detection:
```
cd ../PRESENT-RC && python ../Code/out_of_the_box_exp.py --exp present_pointwise
```
You can see that it reproduces figure **8a** (see the attached figure at the bottom).

- Run multivariate leakage detection (i.e., comparing True positive rates):
```
cd ../PRESENT-RC && python ../Code/out_of_the_box_exp.py --exp present_multivariate
``` 
At present, we only run for the best (in terms of producing a better true positive rate) multivariate test (i.e., the $D$-test, the red solid line in **8c**), and the best univariate test ( $G$-test, the green dashed line in **8b**). The snippets of the multivariate tests are as follows:
<div style="height:300px; width:500px; overflow-y:auto;">
  <img src="https://raw.githubusercontent.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/main/Code/Screenshot_PRESENT_RC_1.png" style="width:100%;">
  <img src="https://raw.githubusercontent.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/main/Code/Screenshot_PRESENT_RC_2.png" style="width:100%;">
  <img src="https://raw.githubusercontent.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/main/Code/Screenshot_PRESENT_RC_3.png" style="width:100%;">
</div>

Like experiment 2, you can make changes to the callable ``run_all_tests()`` to replicate our results, as given in figures **8b and 8c**. Figure 8 is represented as follows:
![Figure 8a](https://github.com/sca-research/Multivariate-Non-profiled-Leakage-Detection/blob/main/Code/Figure_8.png)

It is important to note that this repository is limited to non-profiled leakage detection tests. To get the results corresponding to the Deep-net models, we recommend using the publicly available [DL-LA](https://github.com/Chair-for-Security-Engineering/DL-LA?tab=readme-ov-file) git repository. 
