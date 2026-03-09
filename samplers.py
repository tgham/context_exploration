from numba import njit
import seaborn as sns
import numpy as np
import random
from math import log, exp, gamma
from functools import lru_cache
from matplotlib import pyplot as plt
from scipy.special import beta, logsumexp, digamma

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
def propose(alpha, beta, size=1):
    return np.random.beta(alpha, beta, size=size)

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

        ## hacky fix: should be no duplicates in obs!
        self.obs = np.unique(self.obs, axis=0)

        ## cache obs groups 
        self.row_to_obs = {i: [(i, j, cost) for (i_, j, cost) in self.obs if i_ == i] for i in range(self.N)}
        self.col_to_obs = {j: [(i, j, cost) for (i, j_, cost) in self.obs if j_ == j] for j in range(self.N)}
        self.cached_obs = {}

        ## precompute the number of high and low costs for each row and column
        self.low_counts_rows = np.array([np.sum([cost == self.low_cost for (_, _, cost) in self.row_to_obs[i]]) for i in range(self.N)])
        self.low_counts_cols = np.array([np.sum([cost == self.low_cost for (_, _, cost) in self.col_to_obs[j]]) for j in range(self.N)])
        self.high_counts_rows = np.array([np.sum([cost == self.high_cost for (_, _, cost) in self.row_to_obs[i]]) for i in range(self.N)])
        self.high_counts_cols = np.array([np.sum([cost == self.high_cost for (_, _, cost) in self.col_to_obs[j]]) for j in range(self.N)])


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


    
    ## sampling using counts for each row and column
    def sample(self, col_context=True, n_samples=1):
        self.col_probs = np.ones((n_samples, self.N)) 
        self.row_probs = np.ones((n_samples, self.N))
        
        ## if BAMCP, then parameters are *sampled* from beta distribution
        if not self.CE:
            if col_context:
                # self.row_probs = np.ones(self.N)
                for j in range(self.N):
                    low_counts_col = self.low_counts_cols[j]
                    high_counts_col = self.high_counts_cols[j]
                    alpha = self.alpha_col + low_counts_col
                    beta = self.beta_col + high_counts_col
                    self.col_probs[:,j] = propose(alpha, beta, size=n_samples)
            else:
                # self.col_probs = np.ones(self.N)
                for i in range(self.N):
                    low_counts_row = self.low_counts_rows[i]
                    high_counts_row = self.high_counts_rows[i]
                    alpha = self.alpha_row + low_counts_row
                    beta = self.beta_row + high_counts_row
                    self.row_probs[:,i] = propose(alpha, beta, size=n_samples)

        ## if CE, then parameters are *fixed* at the mean of the beta distribution, whose parameters are determined by the counts
        elif self.CE:
            if col_context:
                for j in range(self.N):
                    low_counts_col = self.low_counts_cols[j]
                    high_counts_col = self.high_counts_cols[j]
                    alpha = self.alpha_col + low_counts_col
                    beta = self.beta_col + high_counts_col
                    self.col_probs[:,j] = alpha / (alpha + beta)
            else:
                for i in range(self.N):
                    low_counts_row = self.low_counts_rows[i]
                    high_counts_row = self.high_counts_rows[i]
                    alpha = self.alpha_row + low_counts_row
                    beta = self.beta_row + high_counts_row
                    self.row_probs[:,i] = alpha / (alpha + beta)
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

