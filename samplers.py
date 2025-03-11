from numba import njit
import seaborn as sns
import numpy as np
import random
from math import log, exp, gamma
from functools import lru_cache
from matplotlib import pyplot as plt
from scipy.special import beta, logsumexp

@njit
def compute_log_likelihood(sampled_i, sampled_j, rel_obs, proposed_row_p, proposed_col_q, current_row_p, current_col_q, high_cost, low_cost):
    log_likelihood = 0.0
    for i, j, cost in rel_obs:

        ## determine whether to use current or proposed p and q
        # if (i == sampled_i) and (j != sampled_j):
        #     rel_p = proposed_row_p
        #     rel_q = current_col_q
        # elif (j == sampled_j) and (i != sampled_i):
        #     rel_p = current_row_p
        #     rel_q = proposed_col_q
        # elif (i == sampled_i) and (j == sampled_j):
        #     rel_p = proposed_row_p
        #     rel_q = proposed_col_q

        ## more efficient way to do this?
        if i == sampled_i:
            rel_p = proposed_row_p
        else:
            rel_p = current_row_p

        if j == sampled_j:
            rel_q = proposed_col_q
        else:
            rel_q = current_col_q

        proposed_prob = rel_p * rel_q
        current_prob = current_row_p * current_col_q

        ## NB this is assuming prob = p(low cost)
        if cost ==low_cost:
            log_likelihood += log(proposed_prob)
            log_likelihood -= log(current_prob)
        else:
        # elif cost == high_cost:
            log_likelihood += log(1 - proposed_prob)
            log_likelihood -= log(1 - current_prob)

    return log_likelihood

@njit
def compute_log_likelihood_global(obs, row_probs, col_probs, high_cost, low_cost):
    """Compute the total log-likelihood for all observations."""
    log_likelihood = 0.0
    for i, j, cost in obs:
        prob = row_probs[int(i)] * col_probs[int(j)]
        if cost == low_cost:
            log_likelihood += np.log(prob)
        # if cost == high_cost:
        else:
            log_likelihood += np.log(1 - prob)
    return log_likelihood


@njit
def propose(alpha, beta):
    return np.random.beta(alpha, beta)

# @njit
# def beta(a,b):
#     return gamma(a) * gamma(b) / gamma(a+b)

# @njit
@lru_cache(maxsize=None)
def proposal_params(alpha_prior, beta_prior, other_alpha_prior, other_beta_prior, low_counts, high_counts):

    ## calculate prior mean failure
    prior_mean_failure = 1-(
        other_beta_prior / (2*(other_alpha_prior + other_beta_prior))
    )

    ### Count occurrences of each cost

    ## standard case (i.e. pq = p(high cost))
    #... 

    ## alternative case (i.e. pq = p(low cost))
    m = low_counts
    n = prior_mean_failure * high_counts

    ## normalise counts to cap their magnitude
    # cap = 10
    # total_count = m + n
    # if total_count > cap:
    #     m = cap * m / total_count
    #     n = cap * n / total_count

    ## Update Beta parameters based on observed data
    alpha_prop = alpha_prior + m
    beta_prop = beta_prior + n

    return alpha_prop, beta_prop, m, n


@njit
def random_idx(arr_len):
    return np.random.randint(arr_len)

@njit
def acceptance_priors(ps, qs, m1,m2,n1,n2):
    return np.sum(m1 * np.log(ps) + n1 * np.log(1 - ps) + m2 * np.log(qs) + n2 * np.log(1 - qs))

# @lru_cache(maxsize=None)
# def get_rel_obs(obs, sampled_i, sampled_j):
#     rel_obs = np.array([(i_, j_, cost_) for (i_, j_, cost_) in obs if (i_ == sampled_i) or (j_ == sampled_j)], dtype=np.float64)
#     return rel_obs

class GridSampler:
    def __init__(self, alpha_row, beta_row, alpha_col, beta_col, low_cost, high_cost, obs, N=10, CE=False):
        self.alpha_row = alpha_row
        self.beta_row = beta_row
        self.alpha_col = alpha_col
        self.beta_col = beta_col
        self.obs = obs
        self.CE = CE
        if self.obs is None:
            self.obs = np.array([])
        # else:
            # self.obs = np.array([(int(i), int(j), float(cost)) for (i, j, cost) in self.obs])
        self.N = N
        self.low_cost = low_cost
        self.high_cost = high_cost

        ## cache obs groups for lazy sampling
        self.row_to_obs = {i: [(i, j, cost) for (i_, j, cost) in self.obs if i_ == i] for i in range(self.N)}
        self.col_to_obs = {j: [(i, j, cost) for (i, j_, cost) in self.obs if j_ == j] for j in range(self.N)}
        self.cached_obs = {}

        ## precompute the number of high and low costs for each row and column
        self.low_counts_rows = np.array([np.sum([cost == self.low_cost for (_, _, cost) in self.row_to_obs[i]]) for i in range(self.N)])
        self.low_counts_cols = np.array([np.sum([cost == self.low_cost for (_, _, cost) in self.col_to_obs[j]]) for j in range(self.N)])
        # self.high_counts_rows = np.array([np.sum([cost == self.high_cost for (_, _, cost) in self.row_to_obs[i]]) for i in range(self.N)])
        # self.high_counts_cols = np.array([np.sum([cost == self.high_cost for (_, _, cost) in self.col_to_obs[j]]) for j in range(self.N)])
        self.high_counts_rows = np.array([np.sum([cost != self.low_cost for (_, _, cost) in self.row_to_obs[i]]) for i in range(self.N)])
        self.high_counts_cols = np.array([np.sum([cost != self.low_cost for (_, _, cost) in self.col_to_obs[j]]) for j in range(self.N)])


    def init_pqs(self):
        if self.CE:
            self.row_probs = np.full(self.N, self.alpha_row / (self.alpha_row + self.beta_row))
            self.col_probs = np.full(self.N, self.alpha_col / (self.alpha_col + self.beta_col))
            for o in self.obs:
                # if o[0] < self.N:
                self.row_probs[int(o[0])] = np.random.beta(self.alpha_row, self.beta_row)
                # if o[1] < self.N:
                self.col_probs[int(o[1])] = np.random.beta(self.alpha_col, self.beta_col)
        else:
            self.row_probs = np.random.beta(self.alpha_row, self.beta_row, size=self.N)
            self.col_probs = np.random.beta(self.alpha_col, self.beta_col, size=self.N)


    def update(self, it):

        ## get a random observation and the current p and q associated with it
        # sampled_obs = self.obs[random_idx(len(self.obs))]
        sampled_obs = self.obs[self.random_idxs[it]]
        # sampled_i, sampled_j = sampled_obs[0], sampled_obs[1]
        sampled_i, sampled_j = int(sampled_obs[0]), int(sampled_obs[1])
        current_p = self.row_probs[sampled_i]
        current_q = self.col_probs[sampled_j]


        ### get params for proposal distribution

        ## first for the row
        low_counts_row = self.low_counts_rows[sampled_i]
        high_counts_row = self.high_counts_rows[sampled_i]
        alpha_p, beta_p, m1, n1 = proposal_params(self.alpha_row, self.beta_row, 
                                                    self.alpha_col, self.beta_col,
                                                  low_counts_row, high_counts_row)

        ## then for the column
        low_counts_col = self.low_counts_cols[sampled_j]
        high_counts_col = self.high_counts_cols[sampled_j]
        alpha_q, beta_q, m2, n2 = proposal_params(self.alpha_col, self.beta_col, 
                                                    self.alpha_row, self.beta_row,
                                                  low_counts_col, high_counts_col)

        ## draw from proposal distribution
        proposed_p = propose(alpha_p, beta_p)
        proposed_q = propose(alpha_q, beta_q)

        ## subselect relevant obs (i.e. those containing i or j)
        # rel_row_obs = self.row_to_obs[sampled_i]
        # rel_col_obs = [obs for obs in self.col_to_obs[sampled_j] if obs[0] != sampled_i] #need to ensure that this array doesn't also include the observation for row=i and col=j, since this is already included in rel_row_obs
        # rel_obs = rel_row_obs + rel_col_obs
        # rel_obs = np.array([(int(i), int(j), cost) for i, j, cost in self.obs if i == sampled_i or j == sampled_j], dtype=np.float64)
        # rel_obs = np.array([(i_, j_, cost_) for (i_, j_, cost_) in self.obs if (i_ == sampled_i) or (j_ == sampled_j)], dtype=np.float64)
        rel_obs = self.get_rel_obs(sampled_i, sampled_j)

        ## calculate likelihoods
        log_likelihood = compute_log_likelihood(sampled_i, sampled_j, rel_obs, proposed_p, proposed_q,
                                                self.row_probs[sampled_i], self.col_probs[sampled_j], self.high_cost, self.low_cost)

        ## calculate prior * transition prob terms
        proposal_distr_num = (m1 * log(current_p) + n1 * log(1 - current_p) + 
                        m2 * log(current_q) + n2 * log(1 - current_q))
        proposal_distr_den = (m1 * log(proposed_p) + n1 * log(1 - proposed_p) +
                        m2 * log(proposed_q) + n2 * log(1 - proposed_q))        
        
        ## acceptance ratio
        log_acceptance_ratio = log_likelihood + proposal_distr_num - proposal_distr_den
        acceptance_ratio = np.exp(log_acceptance_ratio)

        # if np.random.random() < min(1, acceptance_ratio):
        if self.acceptance_thresholds[it] < min(1, acceptance_ratio):
            self.row_probs[sampled_i] = proposed_p
            self.col_probs[sampled_j] = proposed_q
            # self.n_accepts += 1


    ## Perform a full MH sampling update for all rows and columns
    def update_full(self):
        
        # Backup current parameters
        current_ps = self.row_probs.copy()
        current_qs = self.col_probs.copy()

        ## Propose new probabilities for all rows and columns
        proposed_ps = np.zeros(self.N)
        # proposed_qs = proposed_ps.copy()
        proposed_qs = np.zeros(self.N)
        for i in range(self.N):
            low_counts_row = self.low_counts_rows[i]
            high_counts_row = self.high_counts_rows[i]
            alpha_p, beta_p, m1, n1 = proposal_params(self.alpha_row, self.beta_row, 
                                                    self.alpha_col, self.beta_col,
                                                      low_counts_row, high_counts_row)
            proposed_ps[i] = propose(alpha_p, beta_p)
        for j in range(self.N):
            low_counts_col = self.low_counts_cols[j]
            high_counts_col = self.high_counts_cols[j]
            alpha_q, beta_q, m2, n2 = proposal_params(self.alpha_col, self.beta_col, 
                                                    self.alpha_row, self.beta_row,
                                                      low_counts_col, high_counts_col)
            proposed_qs[j] = propose(alpha_q, beta_q)

        # Compute likelihood for proposed and current probabilities using all observations
        proposed_likelihood = compute_log_likelihood_global(self.obs, proposed_ps, proposed_qs, self.high_cost, self.low_cost)
        current_likelihood = compute_log_likelihood_global(self.obs, current_ps, current_qs, self.high_cost, self.low_cost)

        ## Compute prior terms for all rows and columns
        # prior_proposed = np.sum(m1 * np.log(proposed_ps) + n1 * np.log(1 - proposed_ps) + 
        #                 m2 * np.log(proposed_qs) + n2 * np.log(1 - proposed_qs))
        # prior_current = np.sum(m1 * np.log(current_ps) + n1 * np.log(1 - current_ps) + 
        #                     m2 * np.log(current_qs) + n2 * np.log(1 - current_qs))
        prior_proposed = acceptance_priors(proposed_ps, proposed_qs, m1, m2, n1, n2)
        prior_current = acceptance_priors(current_ps, current_qs, m1, m2, n1, n2)


        # Calculate acceptance ratio
        log_acceptance_ratio = proposed_likelihood - current_likelihood +  prior_current - prior_proposed
        acceptance_ratio = np.exp(log_acceptance_ratio)

        # Accept or reject
        if np.random.random() < min(1, acceptance_ratio):
            self.row_probs = proposed_ps
            self.col_probs = proposed_qs
            # self.n_accepts += 1


    def lazy_sample(self, n_iter=100):
        self.init_pqs()
        
        ## precompute some misc things
        self.random_idxs = np.random.randint(len(self.obs), size=n_iter)
        self.acceptance_thresholds = np.random.random(n_iter)

        ## save progress of samples to check convergence
        # self.row_iters = np.zeros((n_iter, self.N))
        # self.col_iters = np.zeros((n_iter, self.N))
        # self.n_accepts = 0

        ## iterate
        for it in range(n_iter):
            self.update(it)

            ## save samples to plot progress
        #     self.row_iters[it] = self.row_probs
        #     self.col_iters[it] = self.col_probs
        # fig, axs = plt.subplots(1,2, figsize=(10,5))
        # axs[0].plot(self.row_iters)
        # axs[0].set_title('Row probs')
        # axs[1].plot(self.col_iters)
        # axs[1].set_title('Col probs')
        # plt.show()
        # ## same, but a kdeplot of samples
        # fig, axs = plt.subplots(1,2, figsize=(10,5))
        # for i in range(self.N):
        #     sns.kdeplot(self.row_iters[:,i], ax=axs[0])
        #     sns.kdeplot(self.col_iters[:,i], ax=axs[1])
        # axs[0].set_title('Row probs')
        # axs[1].set_title('Col probs')
        # plt.show()
        # print('Acceptance rate:', self.n_accepts / n_iter)

        return self.row_probs, self.col_probs
    
    def full_sample(self, n_iter=100):
        self.init_pqs()

        ## save progress of samples to check convergence
        # self.row_iters = np.zeros((n_iter, self.N))
        # self.col_iters = np.zeros((n_iter, self.N))

        for it in range(n_iter):
            self.update_full()

            ## save samples to plot progress
        #     self.row_iters[it] = self.row_probs
        #     self.col_iters[it] = self.col_probs
        # fig, axs = plt.subplots(1,2, figsize=(10,5))
        # axs[0].plot(self.row_iters)
        # axs[0].set_title('Row probs')
        # axs[1].plot(self.col_iters)
        # axs[1].set_title('Col probs')
        # plt.show()

        return self.row_probs, self.col_probs
    
    ## no combinatorial structure - just use the counts for each row and column
    def simple_sample(self, col_context=True):
        self.col_probs = np.ones(self.N) 
        self.row_probs = np.ones(self.N)
        
        ## if BAMCP, then parameters are *sampled* from beta distribution
        if not self.CE:
            if col_context:
                # self.row_probs = np.ones(self.N)
                for j in range(self.N):
                    low_counts_col = self.low_counts_cols[j]
                    high_counts_col = self.high_counts_cols[j]
                    alpha = self.alpha_col + low_counts_col
                    beta = self.beta_col + high_counts_col
                    self.col_probs[j] = propose(alpha, beta)
            else:
                # self.col_probs = np.ones(self.N)
                for i in range(self.N):
                    low_counts_row = self.low_counts_rows[i]
                    high_counts_row = self.high_counts_rows[i]
                    alpha = self.alpha_row + low_counts_row
                    beta = self.beta_row + high_counts_row
                    self.row_probs[i] = propose(alpha, beta)

        ## if CE, then parameters are *fixed* at the mean of the beta distribution, whose parameters are determined by the counts
        elif self.CE:
            if col_context:
                for j in range(self.N):
                    low_counts_col = self.low_counts_cols[j]
                    high_counts_col = self.high_counts_cols[j]
                    alpha = self.alpha_col + low_counts_col
                    beta = self.beta_col + high_counts_col
                    self.col_probs[j] = alpha / (alpha + beta)
            else:
                for i in range(self.N):
                    low_counts_row = self.low_counts_rows[i]
                    high_counts_row = self.high_counts_rows[i]
                    alpha = self.alpha_row + low_counts_row
                    beta = self.beta_row + high_counts_row
                    self.row_probs[i] = alpha / (alpha + beta)
        return self.row_probs, self.col_probs
    

    def context_posterior(self, context_prior=0.5):
        
        # Compute log-likelihood for columns
        log_col_likelihoods = []
        for j in range(self.N):
            low_counts_col = self.low_counts_cols[j]
            high_counts_col = self.high_counts_cols[j]
            log_lik_j = np.log(beta(self.alpha_col + low_counts_col, self.beta_col + high_counts_col)) - np.log(beta(self.alpha_col, self.beta_col))
            log_col_likelihoods.append(log_lik_j)
        log_col_likelihood = np.sum(log_col_likelihoods)
        
        # Compute log-likelihood for rows
        log_row_likelihoods = []
        for i in range(self.N):
            low_counts_row = self.low_counts_rows[i]
            high_counts_row = self.high_counts_rows[i]
            log_lik_i = np.log(beta(self.alpha_row + low_counts_row, self.beta_row + high_counts_row)) - np.log(beta(self.alpha_row, self.beta_row))
            log_row_likelihoods.append(log_lik_i)
        log_row_likelihood = np.sum(log_row_likelihoods)
        
        # Combine with the log priors (where context_prior is the prior for the context being the column world)
        log_prior_col = np.log(context_prior)
        log_prior_row = np.log(1 - context_prior)
        
        # Compute the log joint probabilities:
        log_joint_col = log_col_likelihood + log_prior_col
        log_joint_row = log_row_likelihood + log_prior_row
        
        # Use logsumexp for the denominator:
        log_denominator = logsumexp([log_joint_col, log_joint_row])
        
        # Compute the posterior probability for context being the column world
        posterior_col = np.exp(log_joint_col - log_denominator)

        # print('Posterior probability of context being the column world:', posterior_col)
        
        return posterior_col

    
    def get_rel_obs(self, sampled_i, sampled_j):
        if (sampled_i, sampled_j) in self.cached_obs:
            return self.cached_obs[(sampled_i, sampled_j)]
        rel_obs = np.array([(i_, j_, cost_) for (i_, j_, cost_) in self.obs if (i_ == sampled_i) or (j_ == sampled_j)], dtype=np.float64)
        self.cached_obs[(sampled_i, sampled_j)] = rel_obs
        return rel_obs

