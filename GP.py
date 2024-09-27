## define the GP model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
import warnings
import heapq


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
        lower, upper = 0, 1
        samples = lower + (upper - lower) * (samples - np.min(samples)) / (np.max(samples) - np.min(samples))

        # make all samples non-negative
        samples += np.abs(np.min(samples))
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
    

    ## calculate shortest direct trajectory between two points, and the reward/cost along this trajectory
    def trajectory(self, points, samples, metric = 'chebyshev', manhattan_order = 'x'):

        ## convert start and end points to 2D coordinates
        start, goal = points
        x1, y1 = points[0].astype(int)
        x2, y2 = points[1].astype(int)
        traj = [(x1, y1)]
        
        ## allow agent to move diagonally
        if metric == 'chebyshev':
            # while trajectory[-1] != end:
            while (x1, y1) != (x2, y2):
                
                ## determine direction of movement
                dx = np.sign(x2 - x1)  # -1, 0, or 1 for x direction
                dy = np.sign(y2 - y1)  # -1, 0, or 1 for y direction
                
                # Move in the direction of the target (i.e. diag if both dx and dy are non-zero)
                x1, y1 = (x1 + dx, y1 + dy)
                traj.append((int(x1), int(y1)))

        ## only allow agent to move in cardinal directions
        elif metric == 'manhattan':

            # first move in x direction, then y direction
            if manhattan_order == 'x':
                while x1 != x2:
                    if x2 > x1:
                        x1 += 1  # Move right
                    else:
                        x1 -= 1  # Move left
                    traj.append((x1, y1))
                while y1 != y2:
                    if y2 > y1:
                        y1 += 1  # Move up
                    else:
                        y1 -= 1  # Move down
                    traj.append((x1, y1))
            
            # first move in y direction, then x direction
            elif manhattan_order == 'y':
                while y1 != y2:
                    if y2 > y1:
                        y1 += 1  # Move up
                    else:
                        y1 -= 1  # Move down
                    traj.append((x1, y1))
                while x1 != x2:
                    if x2 > x1:
                        x1 += 1  # Move right
                    else:
                        x1 -= 1  # Move left
                    traj.append((x1, y1))


        ## calculate the reward along this trajectory
        route_cost = [samples[x, y] for x, y in traj]
        
        return traj, route_cost
    

    ## calculate the optimal trajectory between the two points (i.e. the trajectory with the lowest cumulative cost)
    def optimal_trajectory(self, points, samples, metric = 'chebyshev', h_w = 0):

        # Initialize the open list (priority queue) and closed list (visited nodes)
        start, goal = points
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))
        open_list = []

        ## weighted combination of g(n) (actual cost) and h(n) (heuristic for step count)
        heapq.heappush(open_list, (0 + h_w* self.heuristic(start, goal, metric), 0, start, []))
        closed_list = set()

        # Pop the node with the lowest total cost from the priority queue
        while open_list:
            estimated_total_cost, current_cost, current_point, path = heapq.heappop(open_list)
            if current_point in closed_list:
                continue
            
            # Add the current point to the path
            path = path + [current_point]
            
            # If we reached the goal, return the path and the accumulated reward
            if current_point == goal:
                route_cost = [samples[x, y] for x, y in path]
                return path, route_cost
            
            # Mark this point as visited
            closed_list.add(current_point)
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current_point, samples.shape, metric):
                if neighbor not in closed_list:
                    # Calculate new cost to reach this neighbor
                    new_cost = current_cost + samples[neighbor]
                    
                    # Add the neighbor to the open list with its weighted total cost
                    weighted_total_cost = (1-h_w) * new_cost + h_w*self.heuristic(neighbor, goal, metric)
                    heapq.heappush(open_list, (estimated_total_cost, new_cost, neighbor, path))

        
        # If there's no path found, return empty
        return [], []
    
    def heuristic(self, current, goal, metric = 'chebyshev'):
        x1, y1 = current
        x2, y2 = goal
        if metric == 'chebyshev':
            return max(abs(x2 - x1), abs(y2 - y1))
        elif metric == 'manhattan':
            return abs(x2 - x1) + abs(y2 - y1)
        
    ## get all possible neighbours for a given point in the grid
    def get_neighbors(self, point, grid_shape, metric = 'chebyshev'):
        x, y = point
        neighbors = []
        if metric == 'chebyshev':
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the current point itself
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < grid_shape[0] and 0 <= new_y < grid_shape[1]:
                        neighbors.append((new_x, new_y))
        elif metric == 'manhattan':
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < grid_shape[0] and 0 <= new_y < grid_shape[1]:
                    neighbors.append((new_x, new_y))
        return neighbors

    

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

