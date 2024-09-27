## define the GP model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
import warnings


class GP_world():

    def __init__(self, N, params=None):
        
        ## initialise the GP grid
        self.N = N
        x = np.arange(N)
        y = np.arange(N)
        X,Y = np.meshgrid(x,y)
        self.locations = np.column_stack([X.ravel(), Y.ravel()])

        ## set the kernel parameters
        if params is None:
            self.c = 0
            self.scale = 1.0
            self.theta = 0
            self.sigma_f = 1.0
            self.length_scale = 2
            self.periodic_length_scale = 4
            self.period = 8
            self.periodic_theta = 0
        else:
            self.c = params[0]
            self.scale = params[1]
            self.theta = params[2]
            self.sigma_f = params[3]
            self.length_scale = params[4]
            self.periodic_length_scale = params[5]
            self.period = params[6]
            self.periodic_theta = params[7]



        ## initialise the kernels
        self.K_lin = self.linear()
        self.K_lin_x = self.linear_1D(0)
        self.K_lin_y = self.linear_1D(1)
        self.K_rbf = self.rbf()
        self.K_rbf_x = self.rbf_1D(0)
        self.K_rbf_y = self.rbf_1D(1)
        self.K_periodic_x = self.periodic(0)
        self.K_periodic_y = self.periodic(1)


    #### define the kernels


    ## linear kernels

    
    # linear kernel over x,y, i.e. similarity as a function of the distance from the origin (0,0)
    def linear(self):
        dists = np.sqrt(self.locations[:, 0]**2 + self.locations[:, 1]**2)
        K = np.outer(dists, dists) + self.c
        return K
    
    # linear kernel over x-distance (0) or y-distance (1), i.e. similarity as a function of the distance from (0,:) or (:,0), where the basis vectors are determined by the angle theta (in radians)    
    def linear_1D(self, dim = 0):

        # define basis vectors
        rotation = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        rotated_locations = np.dot(self.locations, rotation)
        # dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])

        # calculate distances
        dists = rotated_locations[:,dim] * self.scale**2
        K = np.outer(dists, dists) + self.c
        return K

    

    ## RBFs 

    # RBF kernel over x,y (i.e. Euclidean distance)
    def rbf(self):
        dists = cdist(self.locations, self.locations, metric='euclidean')
        K = self.sigma_f**2 * np.exp(-0.5 * (dists / self.length_scale)**2)
        return K

    # RBF kernel over just x-distance or y-distance
    def rbf_1D(self, dim=0):

        # define basis vectors
        rotation = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        rotated_locations = np.dot(self.locations, rotation)

        # calculate distances
        dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])
        K = self.sigma_f**2 * np.exp(-0.5 * (dists / self.length_scale)**2)
        return K
    

    ## periodic kernel
    def periodic(self, dim=0):
        rotation = np.array([[np.cos(self.periodic_theta), -np.sin(self.periodic_theta)], [np.sin(self.periodic_theta), np.cos(self.periodic_theta)]])
        rotated_locations = np.dot(self.locations, rotation)
        dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])
        K = self.sigma_f**2 * np.exp(-2 * np.sin(np.pi * dists / self.period)**2 / self.periodic_length_scale**2)
        return K
    

    ## sample from the GP
    def sample(self, K, **kwargs):

        ## check kernel is valid
        self.k_check(K)

        # sample
        mean = np.zeros(self.N**2)
        samples = np.random.multivariate_normal(mean, K).reshape(self.N, self.N)

        #normalise
        samples = (samples - np.min(samples))/(np.max(samples)-np.min(samples))
        return samples
    

    ## generate observations from current true GP kernel
    def gen_obs(self, samples, n_obs):
        obs_idx = np.random.randint(0, self.N**2, size=n_obs)

        ## map these observations to the grid and get the reward values
        obs_coords = np.column_stack(np.unravel_index(obs_idx, (self.N, self.N)))
        obs_rewards = samples[obs_coords[:, 0], obs_coords[:, 1]]
        obs = np.column_stack([obs_idx, obs_coords, obs_rewards])

        return obs
    
    ## generate points to choose between
    def gen_preds(self, samples, n_pred=1):
        preds = []
        for i in range(2):
            pred_idx = np.random.randint(0, self.N**2, size=n_pred)

            ## map these observations to the grid and get the reward values
            pred_coords = np.column_stack(np.unravel_index(pred_idx, (self.N, self.N)))
            pred_rewards = samples[pred_coords[:, 0], pred_coords[:, 1]] #i.e. the true rewards
            preds.append(np.column_stack([pred_idx, pred_coords, pred_rewards]))

        ## convert into nx4x2 array
        preds = np.array(preds)

        return preds

    ## use GP regression to predict posterior distribution of rewards, given these observations,based on the currently inferred kernel
    def post_pred(self, K_inf, obs, pred=None, sigma=0.01):
        if pred is None:
            pred_idx = np.arange(self.N**2)
        else:
            pred_idx = pred

        obs_idx = obs[:, 0].astype(int)
        obs_rewards = obs[:, 3]
        
        # Covariance matrix of the already observed points
        K_obs = K_inf[obs_idx][:, obs_idx]
        
        # Covariance matrix between input points (i.e. points to be predicted) and observed points
        K_pred = K_inf[pred_idx][:, obs_idx]
        
        # inversion covariance matrix
        inv_K = np.linalg.inv(K_obs + sigma**2 * np.eye(len(obs_idx)))
        
        # Posterior mean calculation
        post_mean = K_pred @ inv_K @ obs_rewards
        post_cov = K_inf[pred_idx][:, pred_idx] - K_pred @ inv_K @ K_pred.T
        
        return post_mean, post_cov
    

    ## model chooses between two points
    def sigmoid(self, x, tau=1):
        p1 = 1/(1 + np.exp(-x)/tau)
        p2 = 1-p1
        return np.array([p1, p2])

    

    ## compute log marginal likelihood of set of observations, given the inference kernel
    def likelihood(self, K_inf, obs, sigma=0.01):
        n_obs = len(obs)
        obs_idx = obs[:, 0].astype(int) #i.e. x
        obs_rewards = obs[:, 3] #i.e. y
        K_tmp = K_inf[obs_idx][:,obs_idx] 
        K_tmp = K_tmp + ((sigma**2) * np.eye(n_obs))
        self.k_check(K_tmp)

        ### naive method
        # log_det = 0.5* np.log(np.linalg.det(K_tmp))
        # quad_form = 0.5* obs[:,3].dot(np.linalg.inv(K_tmp).dot(obs[:,3]))
        # norm_term = 0.5 * n_obs * np.log(2*np.pi)
        # ll = -log_det - quad_form - norm_term
        
        ### cholesky method
        
        # ## cho decomposition
        # L = np.linalg.cholesky(K_tmp, lower = True)
        L = scipy.linalg.cholesky(K_tmp, lower=True, check_finite=False)

        ## solve for S1 and S2 using triangular matrices
        # S1 = scipy.linalg.solve_triangular(L, obs_rewards, lower=True)
        # S2 = scipy.linalg.solve_triangular(L.T, S1, lower=False)
        alpha = scipy.linalg.cho_solve((L, True), obs_rewards, check_finite=False)


        ## calculate log likelihood terms
        log_det = np.sum(np.log(np.diag(L))) 
        # quad_form = 0.5 * (obs_rewards.dot(S2))
        quad_form = 0.5 * (obs_rewards@alpha)
        norm_term = 0.5 * n_obs * np.log(2*np.pi)
        ll = -quad_form - log_det - norm_term

        ## raise error if ll is positive
        # if ll > 0:
        #     # print(np.sum(np.log(np.diagonal(L))))
        #     print(log_det, quad_form, norm_term, ll)
        #     raise ValueError("Log likelihood is positive")

        return ll

    ## check that kernel is PSD and symmetric
    def k_check(self, K):
        symm = np.allclose(K,K.T)
        if not symm:
            warnings.warn("Kernel matrix is not symmetric.", UserWarning)
        
        eigenvalues = np.linalg.eigvals(K)
        psd = np.all(eigenvalues >= -1e-10)
        if not psd:
            warnings.warn("Kernel matrix is not positive semi-definite.", UserWarning)

        return np.any([not symm, not psd])
