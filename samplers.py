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

class GridSampler:
    def __init__(self, env):
        self.alpha_row = env.alpha_row
        self.beta_row = env.beta_row
        self.alpha_col = env.alpha_col
        self.beta_col = env.beta_col
        self.N = env.N
        self.low_cost = env.low_cost
        self.high_cost = env.high_cost
        self.set_obs(env.obs)
        self.env = env

    def set_obs(self, obs):
        self.obs = obs

        ## nothing observed yet - set up empty structures
        if len(self.obs) == 0:
            self.low_counts_rows = np.zeros(self.N, dtype=int)
            self.low_counts_cols = np.zeros(self.N, dtype=int)
            self.high_counts_rows = np.zeros(self.N, dtype=int)
            self.high_counts_cols = np.zeros(self.N, dtype=int)
        
        else:
            ## hacky fix: should be no duplicates in obs!
            self.obs = np.array([(int(i), int(j), float(cost)) for (i, j, cost) in self.obs])
            self.obs = np.unique(self.obs, axis=0)

            ## precompute row and column counts directly from obs
            obs_rows = self.obs[:, 0].astype(int)
            obs_cols = self.obs[:, 1].astype(int)
            obs_costs = self.obs[:, 2]

            self.low_counts_rows = np.bincount(obs_rows[obs_costs == self.low_cost], minlength=self.N).astype(int)
            self.high_counts_rows = np.bincount(obs_rows[obs_costs == self.high_cost], minlength=self.N).astype(int)
            self.low_counts_cols = np.bincount(obs_cols[obs_costs == self.low_cost], minlength=self.N).astype(int)
            self.high_counts_cols = np.bincount(obs_cols[obs_costs == self.high_cost], minlength=self.N).astype(int)

    
    ## sampling row and column probabilities from posterior
    def sample_probs(self, col_context=True, n_samples=1):
        self.col_probs = np.ones((n_samples, self.N)) 
        self.row_probs = np.ones((n_samples, self.N))
        
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

        return self.row_probs, self.col_probs
    
    ## get posterior mean row and column probabilities
    def mean_probs(self, col_context=True):
        self.col_probs = np.ones(self.N) 
        self.row_probs = np.ones(self.N)
    
        ## parameters are *fixed* at the mean of the beta distribution, whose parameters are determined by the counts
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
    

    ## update the context posterior based on the observed data and the current context prior
    def update_context_posterior(self, context_prior=0.5):

        ## known context - no need for inference
        if context_prior == 0.0 or context_prior == 1.0:
            return context_prior
        
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
        
        return posterior_col

    
    def sample_mdps(self, n_samples, context_prior):
        """
        Sample n_samples MDP parameterisations from the posterior.
        
        Args:
            n_samples: Number of posterior samples to draw.
            context_prior: Scalar probability that the context is column-world.
        
        Returns:
            List of shallow-copied GridEnv objects.
        """
        
        ## update context posterior and determine how many samples to draw under each context
        context_indicators = np.random.binomial(1, context_prior, size=n_samples).astype(bool)
        n_col_samples = int(np.sum(context_indicators))
        n_row_samples = n_samples - n_col_samples

        ## sample row and column probabilities under each context
        posterior_ps_col, posterior_qs_col = self.sample_probs(col_context=True, n_samples=n_col_samples)
        posterior_ps_row, posterior_qs_row = self.sample_probs(col_context=False, n_samples=n_row_samples)

        ## assemble into (n_samples, N) arrays
        all_ps = np.zeros((n_samples, self.N))
        all_qs = np.zeros((n_samples, self.N))
        all_ps[:n_col_samples, :] = posterior_ps_col
        all_ps[n_col_samples:, :] = posterior_ps_row
        all_qs[:n_col_samples, :] = posterior_qs_col
        all_qs[n_col_samples:, :] = posterior_qs_row

        ## shuffle consistently
        idx = np.random.permutation(n_samples)
        all_ps = all_ps[idx]
        all_qs = all_qs[idx]

        ## compute sampled MDP grids via outer product
        sample_probs = np.einsum('si,sj->sij', all_ps, all_qs)

        ## pin observed cells to their known values
        for i, j, c in self.obs:
            i = int(i)
            j = int(j)
            prob = 1 if c == self.low_cost else 0
            sample_probs[:, i, j] = prob
        
        ## fill in with binary values, i.e. p = p(low cost)
        sample_grids = np.where(
            np.random.rand(*sample_probs.shape) < sample_probs, self.low_cost, self.high_cost
        )


        return [self.env.sim_clone(sample_grids[s]) for s in range(n_samples)]
    


    def mean_mdp(self, context_prior):
        """
        Compute the posterior mean MDP grid (no sampling).
        
        Args:
            context_prior: Scalar probability that the context is column-world.
        
        Returns:
            posterior_mean_mdp: Array of shape (N, N) — expected p(low cost) grid.
            extra: Dict of any additional sampler-specific outputs.
        """
        posterior_ps_col, posterior_qs_col = self.mean_probs(col_context=True)
        posterior_ps_row, posterior_qs_row = self.mean_probs(col_context=False)

        posterior_mean_mdp = (
            context_prior * np.outer(posterior_ps_col, posterior_qs_col)
            + (1 - context_prior) * np.outer(posterior_ps_row, posterior_qs_row)
        )

        ## pin observed cells
        for i, j, c in self.obs:
            i = int(i)
            j = int(j)
            prob = 1 if c == self.low_cost else 0
            posterior_mean_mdp[i, j] = prob

        return posterior_mean_mdp
