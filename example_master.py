'''
This file contains the master parameters for the plate example and other examples so that the examples and figures shown in various notebooks are consistent.
'''
import numpy as np
import scipy.stats as sps

from distr_tools import mixture_dist

# set a random seed
np.random.seed(24513)

####################################################
# Fixed and Wobbly-Plate Example

# define the linear map parameters:
# $X\lambda + y_0$
loc_xA = np.array([0.6,0.7])
loc_xB = np.array([0.8,0.6])
locX_mat = np.stack([loc_xA,loc_xB]) # location of observations
y0 = 3 # height value


# define distribution parameters
# mean of parameter slope or true fixed value for the slope
lam_dagger = np.array([0.8,1.4])

# define the observed distribution parameters
# observed mean
obs_mean = np.dot(locX_mat,lam_dagger)+y0

# define the observed covariance
cov_rot = np.pi/6 # rotation in radians
cov_dir = np.array([[np.cos(cov_rot),-np.sin(cov_rot)],
                    [np.sin(cov_rot),np.cos(cov_rot)]])
cov_eig = 0.25*np.array([3,1])*np.eye(2)
obs_cov = np.linalg.multi_dot([cov_dir,cov_eig,cov_dir.T])

# obtain inverse lambda distribution
locX_mat_inv = np.linalg.inv(locX_mat)
lam_gen_cov = np.linalg.multi_dot([locX_mat_inv,obs_cov,locX_mat_inv.T])

# define the observed sample and generate
# number of observations
n_obs = 500
q_obs = sps.multivariate_normal(obs_mean,obs_cov).rvs(n_obs)

# define prior and initial parameters
sigma0_sphere = 16


####################################################
# Nonlinear 1D-1D Example

def Q_nonlinear_1D_to_1D(x):
    return 1/4*(x+1)*(x-4)*(x-7)

####################################################
# Nonlinear 1D Density Approx Example

# Define the distribution for 1D Approx
left_bound = -2
distA = sps.beta(2,5,loc=-2,scale=3)
distB = sps.truncnorm(loc=2.5,scale=1.5,a=(left_bound-2.5)/1.5,b=np.inf)
distC = sps.truncnorm(loc=10,scale=2.5,a=(left_bound-10)/2.5,b=np.inf)

dist_weights = [2/10,5/10,3/10]

# Tri-peak Mixture
tri_peak_mixture = mixture_dist([distA,distB,distC],weights=dist_weights)

# Total Sample Size and the General Sample
tri_peak_n = 1500
tri_peak_sample = tri_peak_mixture.rvs(tri_peak_n)

# plotting domain for consistency of pointwise evaluations
tri_peak_qx = np.linspace(-3,20,350)

# sample sets over which we can compute terms
tri_peak_N_list = [20,50,100,250,500,1000,1500]

# this is the factor R such that hmise=(R/n)^1/5
KDE_MISE_factor = 2.09122 # computed in Review of Density Estimation notebook

# name of MISE *.npz file
tri_peak_MISE_name = 'tri_MISE'

# name of CI *.npz file
tri_peak_CI_name = 'tri_CI'

# confidence interval chosen estimate
tri_peak_CI_sample_size = 250


# List General Parameters for the Mixture Model
tri_peak_BGMM_arg_dict = {'tol': 1e-3, 
            'init_params': 'random', # parameters are initialized randomly from prior 
            'n_init': 5, # since we are initializing randomly, doing a few initializations is important
            'max_iter': 500,
            'verbose': False, 'verbose_interval': 100,
#             'random_state': 875714, # note that we are FIXING the random state here!
            } 

# BGM L2 err name
tri_peak_BGM_name = 'tri_BGM'

# for the update info
tri_peak_lamx = np.linspace(0,10,500)
tri_peak_update_MISE_name = 'tri_up_MISE'

# name for DPMM err calcs
tri_peak_DPMM_name = 'tri_DPMM_err'
