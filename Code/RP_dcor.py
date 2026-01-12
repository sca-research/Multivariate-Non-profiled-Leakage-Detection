''' 
A Statistically and Numerically Efficient Independence Test Based on Distance Covariance of Randomly Projected Data 

url: https://www.frontiersin.org/articles/10.3389/fams.2021.779841/full
'''
import numpy as np
from dcor import u_distance_covariance_sqr, rowwise
from scipy.stats import multivariate_normal
from scipy.stats import gamma as GM
import math
from scipy.special import gammaln

# from tqdm import tqdm
from numba import njit, prange


from mpmath import* 

from timeit import default_timer as timer # For evaluating the computation time


# Required to calculate the upper alpha point (cutoff) of gamma distribution
def gamma_ratio(p): return np.exp(gammaln((p+1)/2) - gammaln(p/2))  # For Calculating C_p and C_q

def gamma_cdf(x, shape,  scale):
    mp.dps = 25; mp.pretty = True  # This is related to mpmath specification, only affect the mpmath related functions 
    return gammainc(shape, a = 0, b = float(x/scale))/np.exp(gammaln(shape))
    # return gammainc(shape, float(x/scale))


@njit(fastmath=True, parallel=True, cache=True)
def dist_sum(X):  # It is nothing but the parallelized version of pdist function-----------------------
    '''
    Parameters
    ----------
    X : 1D array.

    Returns
    -------
    res : sum of distinct Euclidean distances corresponding to the elements of X.

    Note: To implement numba, one needs to consider "numpy==1.25" or less. 
    '''
    res = 0

    n_x = len(X)
    for i in prange(n_x):
        for j in range(i+1, n_x):
            res += np.abs(X[i] - X[j])
            pass
        pass
    return res


def rndm_projection(X, p):
    '''
    Parameters
    ----------
    X : N x p, array of arrays
    where, p: number of dimensions (p >= 1) and N: number of samples 
    p : number of dimensions (p >= 1)      

    Returns
    -------
    X_new : an array of size N
    DESCRIPTION: Random projection of multivariate array
    '''
    
    X_std = np.random.standard_normal(p)
            
    X_norm = np.linalg.norm(X_std)
    
    U_sphere = np.array(X_std)/X_norm  # Normalize X_std
    
    if p > 1:
        X_new = U_sphere @ X.T
    else:
        X_new = U_sphere * X
        
    return X_new


def u_dist_cov_sqr_mv(X, Y, n_projs = 1000, method ='mergesort'):
    '''
    Parameters
    ----------
    X : N x p, array of arrays, where p > 1
    Y : N x q, array of arrays, where q >= 1 ( when q = 1, it's only an array )
    where p and q: number of dimensions of variable X and Y, respectively and N: number of samples

    n_projs : Number of projections (integer type), optional
        DESCRIPTION. The default is 500.
    method : fast computation method either 'mergesort' or 'avl', optional
        DESCRIPTION. The default is 'mergesort'.

    Returns
    -------
    omega_bar : Float type
        DESCRIPTION: Produce fastly computed unbiased distance covariance between X and Y

    '''
    
    n_samples = np.shape(X)[0]
    p = np.shape(X)[1]
    
    
    if Y.T.ndim == 1:
        q = 1
    else:
        q = np.shape(Y)[1]
    
    sqrt_pi_value = math.sqrt(math.pi)
    C_p = sqrt_pi_value*gamma_ratio(p)
    C_q = sqrt_pi_value*gamma_ratio(q)

            
    X_proj = np.empty(( n_projs, n_samples))    
    Y_proj = np.empty(( n_projs, n_samples))
    
    for i in range(n_projs):
        X_proj[i, :] = rndm_projection(X, p)
        Y_proj[i, :] = rndm_projection(Y, q)    
        pass
          
    omega_ = rowwise(u_distance_covariance_sqr, 
                      X_proj, Y_proj, rowwise_mode = method)
    omega_bar = C_p * C_q * np.mean(omega_)
    
    return omega_bar


def u_dist_cov_sqr_mv_test(X, Y, n_projs = 1000, method ='mergesort'):
    '''

    Parameters
    ----------
    X : N x p, array of arrays, where p > 1
    Y : N x q, array of arrays, where q >= 1 ( when q = 1, it's only an array )
    where p and q are the number of dimensions of X and Y, respectively and N: number of samples

    n_projs : Number of projections (integer type), optional
        DESCRIPTION. The default is 500.
    method : fast computation method either 'mergesort' or 'avl', optional
        DESCRIPTION. The default is 'mergesort'.
    

    Returns
    -------
    Test_statistic : TYPE float
        DESCRIPTION: Test statistic based on fast distance covariance measure
    cutoff (critical value) or p_value : TYPE float 
        DESCRIPTION: Upper alpha (0.05) point (Cutoff) of gamma distribution / p_value corresponding to gamma distribution 
    '''

    n_samples = np.shape(X)[0]
    p = np.shape(X)[1]
    
    
    if Y.T.ndim == 1:
        q = 1
    else:
        q = np.shape(Y)[1]
    
    
    sqrt_pi_value = math.sqrt(math.pi)
    C_p = sqrt_pi_value * gamma_ratio(p)
    C_q = sqrt_pi_value * gamma_ratio(q)
    

    
    X_proj_1 = np.empty(( n_projs, n_samples))    
    Y_proj_1 = np.empty(( n_projs, n_samples))
    X_proj_2 = np.empty(( n_projs, n_samples))    
    Y_proj_2 = np.empty(( n_projs, n_samples))
    S2_n = 0
    S3_n = 0
    
    for i in range(n_projs):
        X_proj_1[ i, :] = rndm_projection(X, p)
        Y_proj_1[ i, :] = rndm_projection(Y, q)
        S2_n += (2 * dist_sum(X_proj_1[ i, :]))
        S3_n += (2 * dist_sum(Y_proj_1[ i, :]))
        X_proj_2[ i, :] = rndm_projection(X, p)
        Y_proj_2[ i, :] = rndm_projection(Y, q)

        
    omega1_ = rowwise(u_distance_covariance_sqr, 
                      X_proj_1, Y_proj_1, rowwise_mode= method)
    omega1_bar = C_p * C_q * np.mean(omega1_)
    
    S11_ =  np.array(rowwise(u_distance_covariance_sqr, 
                             X_proj_1, X_proj_1, rowwise_mode= method))
    S12_ =  np.array(rowwise(u_distance_covariance_sqr,
                             Y_proj_1, Y_proj_1, rowwise_mode= method))   
    S1_bar = C_p * C_q * np.mean(S11_* S12_)
    
    S2_bar = (C_p * S2_n) / (n_projs * n_samples * (n_samples-1))
    S3_bar = (C_q * S3_n) / (n_projs * n_samples * (n_samples-1))
    
    omega2_ = rowwise(u_distance_covariance_sqr, 
                      X_proj_1, X_proj_2, rowwise_mode = method)
    omega2_bar = (C_p ** 2) * np.mean(omega2_)
    
    omega3_ = rowwise(u_distance_covariance_sqr, 
                      Y_proj_1, Y_proj_2, rowwise_mode = method)
    omega3_bar = (C_q ** 2) * np.mean(omega3_)
    
    # calculate alpha and beta--------------------------------------
    denom = (((n_projs-1) * omega2_bar * omega3_bar) + S1_bar) / n_projs
    alpha = (0.5 * ((S2_bar * S3_bar) ** 2)) / denom
    beta = (0.5 * S2_bar * S3_bar) / denom

    # calculate test statistic and the p-value--------------
    Test_statistic = ((n_samples * omega1_bar) + (S2_bar * S3_bar))
    cutoff = GM.ppf(1 - 0.05, a = alpha, loc = 0, scale = float(1 / beta))
    # p_val = 1 - gamma_cdf(Test_statistic, 
    #           shape = alpha,  scale = float(1 / beta))
    
    # if p_val < 0: p_val = 0
        
    return Test_statistic, cutoff
    # return Test_statistic, p_val




def checking_correctness():
    
    '''
    Checking the correctnes of fast evaluated  "u_dist_cov_sqr_mv()" comparing with the naive $\Omega(X,Y)$ "u_distance_covariance_sqr" 
    '''
    
    # Specifiy the dimension n_d
    dim_data = 5 # Increase the dimension size to check the scalability of u_dist_cov_sqr_mv()
    
    # Define the mean vector and 'symmetric' Covariance matrix for multivariate Gaussian simulation
    mean_vector = np.repeat(3, dim_data)
    A = 0.01 * np.random.rand(dim_data, dim_data) # increase the 0.01 to higher value will increase the variance values in Cov_matrix
    Cov_matrix = np.dot(A, A.transpose())  
    
    n_samples = 3000  
    print("Data size = {}".format(n_samples)) 
    
    # Simulated Data from Multivariate Gaussian Distribution----------------
    X = multivariate_normal.rvs( mean_vector, Cov_matrix, size = n_samples)
    
    Tr = X.T[:(dim_data-2)]
    pred_lkg = X.T[(dim_data-2):]
    
    dim_Tr = np.shape(Tr)[0]
    print("dimension of Tr = {}".format(dim_Tr))
    
    # For checking accuracy of the 'fast' dcov estimator with the 'naive' one    
    print("distance covariance based on naive approach ={} ".format(  # Uncomment this part for higher(>5) dim_data
       u_distance_covariance_sqr(Tr.T, pred_lkg.T, method='naive')))  
    
    start = timer()
    print("Computing fast distance covariance = {}".format(
        u_dist_cov_sqr_mv(Tr.T, pred_lkg.T)))
    end = timer()
    print("Time to compute fast distance covariance in seconds = {}".format(end - start))



def scalability_testing():
    '''
    Computing runtime for fast "u_dist_cov_sqr_mv()" for different number of dimensions

    '''
    
    # Specifiy the dimension n_d
    n_samples = int(2e03)
    for dim_data in range(10, 1500, 100): 
    
        # Define the mean vector and 'symmetric' Covariance matrix for multivariate Gaussian simulation
        mean_vector = np.repeat(3, dim_data)
        A = 0.01 * np.random.rand(dim_data, dim_data) 
        Cov_matrix = np.dot(A, A.transpose())  
        
        # Simulated Data from Multivariate Gaussian Distribution----------------
        X = multivariate_normal.rvs( mean_vector, Cov_matrix, size = n_samples)
        Tr = X.T[:(dim_data-2)]
        pred_lkg = X.T[(dim_data-2):]
        
        
        ## Computing Runtime of u_dist_cov_sqr_mv()
        start = timer()
        u_dist_cov_sqr_mv(Tr.T, pred_lkg.T)
        end = timer()
        
        print("No. of dimensions = {}, Run_time ={}".format( dim_data ,   end - start))








if __name__ == '__main__':
    # checking_correctness() 
    scalability_testing()
