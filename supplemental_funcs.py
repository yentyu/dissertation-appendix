'''
This module contains a set of supplemental functions that are useful for a variety of tasks that I repeat over and over again.
'''
import numpy as np
import scipy.stats as sps
from scipy.integrate import quad
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from distr_tools import mixture_dist

# the following set of functions is used so that I can evaluate probability densities
# from different packages with dealing with tedious format issues
def eval_pdf(vals, pdf):
    if type(pdf) == sps.gaussian_kde:
        out = eval_gkde_model(vals,pdf)
    elif type(pdf) == BayesianGaussianMixture or type(pdf) == GaussianMixture:
        out = eval_skl_mixture(vals,pdf)
    else:
        out = eval_sps_dist(vals,pdf)
    return out


# create wrapper functions to compute sps probabilities
def eval_sps_dist(vals,sps_dist):
    '''
    vals: values to get the observed probabilities (sample x ndim)
    sps_dist: scipy.stats distribution
    '''
    if type(vals)!=np.ndarray:
        format_vals = np.atleast_1d(np.array(vals))
    else:
        format_vals = vals
    
    # comput the pdf values
    try:
        # breaks sometimes for multivariate distributions
        out = np.squeeze(sps_dist.pdf(format_vals))
    except ValueError as err:
        try:
            # a quick fix, try transposing the distribution
            out = np.squeeze(sps_dist.pdf(format_vals.T))
        except ValueError:
            raise err
    
    return out


# create wrapper functions to kde probabilities
def eval_gkde_model(vals,sps_kde_model):
    '''
    vals: values to get the observed probabilities (sample x ndim)
    sps_kde_model: scipy.stats gaussian kde model
    '''
    if type(vals)!=np.ndarray:
        format_vals = np.atleast_1d(np.array(vals))
    else:
        format_vals = vals
    
    # reshape vals to appropriate shape
    if len(format_vals.shape) > 1:
        if format_vals.shape[0] != sps_kde_model.d:
            format_vals = format_vals.T
            
    return np.squeeze(sps_kde_model(format_vals))

# create wrapper functions to compute skl probabilities
def eval_skl_mixture(vals,skl_model):
    '''
    vals: values to get the observed probabilities (sample x ndim)
    skl_model: scikit-learn mixture model
    '''
    if type(vals)!=np.ndarray:
        format_vals = np.atleast_1d(np.array(vals))
    else:
        format_vals = vals
    
    # reshape vals to appropriate shape
    if len(format_vals.shape) < 2:
        if skl_model.n_features_in_==1:
            format_vals = format_vals.reshape(-1,1)
        else:
            format_vals = format_vals.reshape(1,-1)
    else:
        if format_vals.shape[1] != skl_model.n_features_in_:
            format_vals = format_vals.T
    
    return np.squeeze(np.exp(skl_model.score_samples(format_vals)))


# function for determining L2 error in 1D
def L2_err_1D(densityA,densityB, a,b, quad_kwargs={}):
    # make sure we're using same functions    
    M2_func = lambda x: (eval_pdf(x,densityA)-eval_pdf(x,densityB))**2
    
    return quad(M2_func,a,b,**quad_kwargs)

# function for determining L2 error in 1D
def L1_err_1D(densityA,densityB, a,b, quad_kwargs={}):
    # make sure we're using same functions    
    M1_func = lambda x: np.abs(eval_pdf(x,densityA)-eval_pdf(x,densityB))
    
    return quad(M1_func,a,b,**quad_kwargs)


## Dirichlet Process Stick Breaking PDF
class trunc_stick_breaking:
    '''
    Creates a truncated stick-breaking process that can be used for sampling
    and evaluating pdfs like other scipy.stats functions
    
    K_trunc: number of components before truncation
    alpha: vector of "a" parameters for Beta("a","b"), one for each k in K_trunc
    beta: vector of "b" parameters for Beta("a","b"), one for each k in K_trunc
    
    '''
    
    def __init__(self, alpha, beta, K_trunc):
        self.K = K_trunc # truncating of stick-breaking process
        
        # make sure there are enough parameters for the beta distributions
        if np.shape(alpha)[0] != self.K or np.shape(beta)[0] != self.K:
            raise ValueError('Must have K_trunc={} parameters (alpha,beta)'.format(self.K))
        
        # save parameter values
        self.a = alpha
        self.b = beta
        
    def pdf(self, vals):
        '''
        vals: points to evaluate size (eval N x truncation K)
        returns: pdf values
        '''
        w_eval = np.atleast_2d(vals)
        eval_N, K = w_eval.shape
        if K != self.K:
            raise ValueError('Points are not correct dimension.'+
                             ' Must be (eval N x truncation K)')
        
        these_beta = self.get_beta_from_weights(w_eval)
        
        # the likelihood is the product of independent beta draws
        pdf_out = np.ones(eval_N)
        for k in np.arange(K-1):
            pdf_out *= sps.beta.pdf(these_beta[:,k],a=self.a[k],b=self.b[k])
        
        return np.squeeze(pdf_out)
        
        
    def rvs(self, N):
        sample = np.zeros([N,self.K])
        
        # get sample of beta sticks for each seperate beta
        for k in np.arange(self.K):
            sample[:,k] = sps.beta.rvs(size=N,a=self.a[k],b=self.b[k])
        # make sure last value is 1
        sample[:,-1] = 1    
        
        weight_sample = self.weights_from_beta(sample)
        
        return np.squeeze(weight_sample)
    
    def get_beta_from_weights(self,w):
        # w should be at least 2D, eval N x trunc K
        N_eval, K  = w.shape
        
        # beta is same size as w
        beta = np.copy(w)
        
        for k in np.arange(K):
            if k == 0:
                continue
            else:
                beta[:,k] = w[:,k]/np.prod(1-beta[:,0:k],axis=1)
        # returns beta at least 2D
        return beta
        
        
    def weights_from_beta(self,beta_sample):
        # make sure last sampled value is equal to 1
        if np.any(beta_sample[:,-1]!=1):
            beta_sample[:,-1] = 1 
        
        # get the leftover stick sizes:
        # produces an array 1, (1-b1), (1-b1)(1-b2), etc.
        stick_leftover = np.roll(beta_sample,1,axis=1)
        stick_leftover[:,1::] = np.cumproduct(1-stick_leftover[:,1::],axis=1)
        
        # for each k, beta_k * leftover stick lengths
        these_weights = beta_sample*stick_leftover
        
        return these_weights
    
    def mean(self):
        expect_beta = np.atleast_2d(self.a/(self.a+self.b))

        # compute the weights from expected beta values
        mean_weights = self.weights_from_beta(expect_beta)
        return np.squeeze(mean_weights)
    

# sklearn Bayesian Gaussian Mixture Model Forward Sampler
class Forward_BGM_Model():
    '''
    Takes a scikit-learn fitted BGM model and extracts the parameters
    to create a forward model for sampling the potential density estimations.
    
    Note that the BGM model must be fitted, even if the prior is desired.
    This is because the BGM model only calculates certain prior parameters
    after skl_BGM.fit(...) is called. If the prior is desired, a quick
    workaround is to simply fit to a small set of arbitrary data.
    '''
    def __init__(self,skl_BGM, prior=False):
        # check to see if the sklearn model has been fitted or not
        try:
            skl_BGM.converged_
        except AttributeError:
            raise ValueError('Model not fitted. Model must be fitted to data'+
                         ' even if prior desired.')
        
        # use the prior or the posterior distribution
        self.prior = prior
        
        # get the max truncation number of components and dimension
        self.K = skl_BGM.n_components
        self.dim = skl_BGM.n_features_in_
        
        # get the covariance type (full,tied,diag,spherical)
        # this influences how BGM returns covariances
        this_cov_type = skl_BGM.covariance_type
        
        # get the type of BGMM: either finite (dirichlet_distribution) or 
        # "infinite" DP model (dirichlet_process)
        this_model_type = skl_BGM.weight_concentration_prior_type
        self.model_type = this_model_type
        
        # get the prior hyper parameters
        if self.prior == True:
            # hyperparams for the weights 
            # will be different for finite Bayes and DP models
            # use dictionary-type case handling
            this_alpha_float = skl_BGM.weight_concentration_prior_
            this_alpha = {'dirichlet_distribution': 
                              np.ones(self.K)*this_alpha_float ,
                          'dirichlet_process': 
                          # needs to be a tuple for betas(1,alpha) in stickbreaking
                              ( np.ones(self.K), np.ones(self.K)*this_alpha_float)
                             }[this_model_type]
            
            # inverse-wishart hyper params
            these_nu = skl_BGM.degrees_of_freedom_prior_*np.ones(self.K)
            
            # use dictionary-style to handle different cov-type cases
            # consistent with sklearn code
            these_Psi = {'full': skl_BGM.covariance_prior_,
                         'tied': skl_BGM.covariance_prior_,
                         'diag': np.diag(skl_BGM.covariance_prior_),
                         'spherical': skl_BGM.covariance_prior_*np.eye(self.dim)
                        }[this_cov_type]
            
            # the hyperparameters for the covariances (K, dim x dim)
            these_Psi = np.repeat(these_Psi[np.newaxis,:],self.K,axis=0) # each component has same covariance
            
            
            # get the means hyper-parameters
            # note that the means are dependent on the covariance
            # for the prior, we use the mode of the IW as the initial parameter
            these_cov = these_Psi/(these_nu[0]+self.K+1) # (K, dim x dim)
            this_kappa = skl_BGM.mean_precision_prior_*np.ones([self.K,1,1])
            these_mean = np.repeat(skl_BGM.mean_prior_[np.newaxis,:],
                                   self.K,axis=0)
            
        else:
            # get the params for fitted model
            # For the weights will be different for finite Bayes and DP models
            # however the output is the correct format:
            # it will be a vector in distribution case and a tuple in DP case.
            this_alpha = skl_BGM.weight_concentration_
            
            # inverse-wishart hyper params
            these_nu = skl_BGM.degrees_of_freedom_
            
            # treat covariances differently depending on output type
            if this_cov_type == 'full':
                these_cov = skl_BGM.covariances_
            elif this_cov_type == 'tied':
                # tied uses same covariance for each component k
                these_cov = np.repeat(skl_BGM.covariances_,self.K,axis=0)
            elif this_cov_type == 'diag': 
                # diag uses k pairs of diagonal covariances
                these_cov = np.array([np.diag(cov) for cov in skl_BGM.covariances_])
            elif this_cov_type == 'spherical': 
                these_cov = np.array([cov*np.eye(self.K) for cov in skl_BGM.covariances_])
            else:
                raise ValueError('Unknown model covariance type: {}'.format(this_cov_type))
            
            # the hyperparameters for the covariances (K, dim x dim)
            these_Psi = these_nu[:,np.newaxis,np.newaxis]*these_cov
            
            
            # get the means hyper-parameters
            # note that the means are dependent on the covariance
            # for the posterior, we use the posterior covariances
            this_kappa = skl_BGM.mean_precision_[:,np.newaxis,np.newaxis]
            these_mean = skl_BGM.means_
        
        # use the hyperparameters above to define the 
        # appropriate sampling distributions
        
        # save the weight distributions: different for finite vs. infinite cases
        if skl_BGM.weight_concentration_prior_type == 'dirichlet_distribution':
            # this is the finite case
            self.weight_dist = sps.dirichlet(this_alpha)
        else:
            # use custom stick-breaking class: a set beta(a,b), one for each K
            self.weight_dist = trunc_stick_breaking(this_alpha[0],
                                                    this_alpha[1],self.K)
        
        # create the sampling distribution for the variances
        self.var_dists = [sps.invwishart(df=df,scale=sc) for df,sc in \
                          zip(these_nu,these_Psi)]
        
        # save the precisions
        self.kappa = this_kappa
        
        # create the sampling distribution for the means
        self.mean_dists = [sps.multivariate_normal(mu,sigma2) for mu, sigma2 \
                           in zip(these_mean,1/this_kappa*these_cov)]
    
    def pdf(self,x,parameter=['expect_dist']):
        '''
        Evaluates the pdf of given parameter
        x: array-like, shape compatible with parameter distribution
        parameter: list, which parameter of hierarchical model to evaluate must be a list
                    ['expect_dist'],['weight'],['cov',k],['mean',k]
                    where k is the component pdf to be evaluated
        
        '''
        if parameter[0]=='weight':
            output = self.weight_dist.pdf(x)
            
        elif parameter[0]=='cov':
            k = parameter[1]
            output = self.var_dists[k].pdf(x)
            
        elif parameter[0]=='mean':
            k = parameter[1]
            output = self.mean_dists[k].pdf(x)
        
        elif parameter[0]=='expect_dist':
            # computes the expected distribution for the forward model
            eval_dict = {} # dictionary of values to be computed
            
            # expected covariance
            cov_list = []
            for dist in self.var_dists:
                if dist.df > dist.dim +1:
                    cov_list.append(dist.mean())
                else:
                    # the mean doesn't exist for df <= dim + 1
                    cov_list.append(dist.mode()) # so use the mode
            eval_dict['cov'] = np.atleast_3d(np.array(cov_list))
            
            # mean values are just mean of conditionals
            these_means = np.array([dist.mean for dist in self.mean_dists])
            eval_dict['mean'] = these_means[np.newaxis,:]
            
            # weights: different if BMM vs. DPMM
            if self.model_type == 'dirichlet_distribution':
                expected_weights = self.weight_dist.alpha/np.sum(self.weight_dist.alpha)
                eval_dict['weight'] = np.atleast_2d(expected_weights)
            else:
                # compute the expected beta values from each distribution
                this_a = self.weight_dist.a
                this_b = self.weight_dist.b
                expect_beta = np.atleast_2d(this_a/(this_a+this_b))
                
                # compute the weights from expected beta values
                eval_dict['weight'] = self.weight_dist.weights_from_beta(expect_beta)
            
            # evaluate the dictionary of parameters using batch_GMM_pdf
            # print(eval_dict)
            output = np.squeeze(batch_GMM_pdf(x,eval_dict))
            
        else:
            raise ValueError("Must be a list: ['expect_dist'],['weight'],['cov',k],['mean',k], where k is the component pdf to be evaluated")
        
        return output
    
    def rvs(self, N=1):
        '''
        Returns dictionary of sampled parameters from posterior distributions
        '''
        out_dict = {}
        
        # sample the weight distribution
        out_dict['weight'] = self.weight_dist.rvs(N)
        
        # sample the covariances
        cov_list = []
        for dist in self.var_dists:
            cov_list.append(dist.rvs(N))
        out_dict['cov'] = np.stack(cov_list,axis=1)
        
        # note that the means are dependent on the covariances
        mean_list = []
        for i in np.arange(N):
            k_means = np.empty([self.K,self.dim])
            for k,dist in enumerate(self.mean_dists):
                # this covariance is the ith sample of component k
                this_cov = 1/self.kappa[k]*cov_list[k][i]
                k_means[k,:] = sps.multivariate_normal.rvs(mean=dist.mean,cov=this_cov,size=1)
            mean_list.append(k_means)
        out_dict['mean'] = np.stack(mean_list,axis=0)
        
        return out_dict
    
def batch_GMM_pdf(x,param_sample):
    '''
    Evaluates a batch of GMM at x
    x: array-like, shape (eval_points x dim of GMM)
    param_sample: dictionary of weights, means, and covariances, each an array shape (N_sample x components x ... )
    '''
    # get the batch sample size and number of components from the parameter sample
    n_samp, n_comp = param_sample['weight'].shape
    
    # specify output vals as a list
    output_vals = []
    
    # for each sampled GMM pdf
    for i in range(n_samp):
        
        this_pdf_vals = np.zeros(x.shape[0])
        
        # for each component, add the mv pdf contribution at x
        for k in range(n_comp):
            this_mvnormal = sps.multivariate_normal(mean=param_sample['mean'][i][k],
                                                    cov=param_sample['cov'][i][k])
            this_pdf_vals += param_sample['weight'][i][k]*this_mvnormal.pdf(x)
        
        # record the values
        output_vals.append(this_pdf_vals)
    
    return np.stack(output_vals,axis=0)


# the following section of code allows us to use the DC update class
def check_nd_shape(lam,dim=1):
    if len(lam.shape)==0:
        vals = lam.reshape(1,1)
    elif len(lam.shape)==1:
        if dim==1:
            vals = lam.reshape(lam.shape[0],1)
        else:
            vals = lam.reshape(1,lam.shape[0])
    else:
        vals = lam
    return vals

class dci_update:
    def __init__(self,init,pred,obs,Qmap):
        self.init_dist = init
        self.pred_dist = pred
        self.obs_dist = obs
        self.Q = Qmap
        
    def pdf(self,lam):
        # specify vals to evaluate and get Q(lam)
        lam_vals = check_nd_shape(lam)
        lam_update = check_nd_shape(np.zeros(lam_vals.shape[0]))
        q = self.Q(lam_vals)
        
        # compute pdfs
        init_vals = check_nd_shape(eval_pdf(lam_vals,self.init_dist))
        pred_vals = check_nd_shape(eval_pdf(q,self.pred_dist))
        obs_vals = check_nd_shape(eval_pdf(q,self.obs_dist))
        
        # predictability assumption
        nonzeros = init_vals != 0
        nonzero_denom = pred_vals > 1e-6*obs_vals 
        up_ind = np.logical_and(nonzeros,nonzero_denom)
        
        # return updated values
        lam_update[up_ind] = init_vals[up_ind]*obs_vals[up_ind]/pred_vals[up_ind]
        lam_update = np.squeeze(lam_update)
        
        return lam_update
        
    

# # our model map for wobbly plate
# def Y_func(beta,x, height=3, 
#       additive_noise=False, add_noise_pdf=sps.norm(0,0.15), 
#       location_noise=False, loc_noise_pdf=sps.norm(0,0.15)):
    
#     '''
#     beta: matrix of coefficients, each row a separate observation
#     x: location vector of measurements OR matrix. If matrix, must be the same size as beta
#     height: fixed height of the center of the wobbly plate
#     addiive_noise: is there additive noise?
#     add_noise_pdf: scipy stats class specificying the additive noise model
#     location_noise: is there location noise?
#     loc_noise_pdf: scipy stats class specifying location noise model
#     '''
    
#     y0 = height # fixed height
    
#     if beta.shape != x.shape:
#         # reshape x-array to be same size as beta
#         columns = np.shape(beta)[0]
#         x_values = np.repeat(x,columns).reshape(np.shape(x)[0],columns).transpose()
#     else:
#         x_values = x
    
#     if location_noise:
#         x_values = x_values+loc_noise_pdf.rvs(x_values.shape)
    
    
#     y_output = y0+np.sum(beta*x_values,axis=1)
    
#     # add additive noise
#     if additive_noise:
#         y_output = y_output+add_noise_pdf.rvs(y_output.shape)
    
#     return y_output, x_values

# # observed distribution for Dirichlet example
# def example_obs_distr():
#     # standard deviation parameter
#     sigma = 3

#     # pdfs
#     pdf_lists = [sps.norm(2,sigma/6),sps.norm(-2.4,sigma/6),sps.norm(0,sigma/5)]

#     # weight of each gaussian mode
#     weights = [0.2,0.4,0.4]

#     return mixture_dist(pdf_lists,weights)

# # model map for Dirichlet example
# def Q(x,y):
#     return x**2+y**3-2*y**2+x*y

# # function to acquire dirichlet mixture pdfs
# def get_mixture_pdfs(x,weights,mu,tau):
#     '''
#     x: the vector you want to evaluate the pdfs at (shape M)
#     mu: the sampled mixture means from the dirichlet process (N samples x K components)
#     tau: the sampled mixture precisions from the dirichlet process (NxK)
    
#     returns: numpy array of evaluations of each mixture component pdf shape Nx(M)xK
#     '''
    
#     post_pdf_pieces = sps.norm.pdf(np.atleast_3d(x),
#                                      mu[:,np.newaxis,:],
#                                  1/np.sqrt(tau[:,np.newaxis,:]))
    
#     weighted_pieces = weights[:,np.newaxis,:] * post_pdf_pieces
    
#     return weighted_pieces

# # function to acquire dirichlet posterior pdfs
# def get_posterior_pdfs(x,weights,mu,tau):
#     '''
#     x: the vector you want to evaluate the pdfs at (shape M)
#     mu: the sampled mixture means from the dirichlet process (N samples x K components)
#     tau: the sampled mixture precisions from the dirichlet process (NxK)
    
#     returns: numpy array of evaluations of posterior pdf shape Nx(M)
#     '''
#     pieces = get_mixture_pdfs(x,weights,mu,tau)
#     return pieces.sum(axis=-1)

# ### plotting tools for interactive widgets
# # bar selector
# def sample_bar_selector(graph_list,turn_on=None):
#     '''
#     graph_list: list of bar_container objects from BarContainer class
#     turn_on: None turns off all bar graphs in graph list. 
#                 If 'all', turns on all bar graphs.
#                 Else should be an index indicating which bar graph in the graph list to turn on.
#     '''
    
#     # TURN OFF: for each bar graph in the graph list...
#     for i,bar_container in enumerate(graph_list):
#         # turn  off all bars
#         for bar in bar_container.get_children():
#             if bar.get_visible()==True:
#                 bar.set_visible(False)
    
#     # TURN ON: this bar graph
#     if turn_on==None:
#         return
#     elif turn_on=='all':
#         for i,bar_container in enumerate(graph_list):
#             # turn  off all bars
#             for bar in bar_container.get_children():
#                 if bar.get_visible()==False:
#                     bar.set_visible(True)
#     else:
#         for bar in graph_list[turn_on].get_children():
#             bar.set_visible(True)

# # plot selector
# def sample_plot_selector(graph_list,turn_on=None):
#     '''
#     graph_list: list of curve objects from Line2D class
#     turn_on: None turns off all curves graphs in graph list. 
#                 If 'all', turns on all plots graphs.
#                 Else should be an index indicating which curves in the graph list to turn on.
#     '''
    
#     # TURN OFF: for each bar graph in the graph list...
#     for i,curve in enumerate(graph_list):
#         # turn  off all curves
#         if curve.get_visible()==True:
#             curve.set_visible(False)
    
#     # TURN ON: this bar graph
#     if turn_on==None:
#         return
#     elif turn_on=='all':
#         for i,curve in enumerate(graph_list):
#             # turn  off all bars
#             if curve.get_visible()==False:
#                 curve.set_visible(True)
#     else:
#         graph_list[turn_on].set_visible(True)

# # legend changer
# def legend_changer(axis,constant_handles,changing_handles):
#     '''
#     This function changes the legend for a given axis
#     axis: axis where the legend is to be changed
#     constant_handles: list of the handles which will not change
#     changing_handles: list of handles which are changing
#     '''
#     current_handles = constant_handles+changing_handles
#     axis.legend(handles=current_handles)
    
