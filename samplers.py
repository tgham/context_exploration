from numba import njit
import numpy as np
import random

@njit
def compute_log_likelihood(sampled_i, sampled_j, rel_obs, proposed_row_p, proposed_col_q, current_row_p, current_col_q, high_cost, low_cost):
    log_likelihood = 0.0
    for o in rel_obs:
        i, j, cost = int(o[0]), int(o[1]), o[2]
        if (i == sampled_i) and (j != sampled_j):
            rel_p = proposed_row_p
            rel_q = current_col_q
        elif (j == sampled_j) and (i != sampled_i):
            rel_p = current_row_p
            rel_q = proposed_col_q
        elif (i == sampled_i) and (j == sampled_j):
            rel_p = proposed_row_p
            rel_q = proposed_col_q
        else:
            raise ValueError("Observation does not match row or column.")

        proposed_prob = rel_p * rel_q
        current_prob = current_row_p * current_col_q

        ## NB this is assuming prob = p(low cost)
        if cost == high_cost:
            log_likelihood += np.log(1 - proposed_prob)
            log_likelihood -= np.log(1 - current_prob)
        elif cost == low_cost:
            log_likelihood += np.log(proposed_prob)
            log_likelihood -= np.log(current_prob)

    return log_likelihood

@njit
def propose(alpha, beta):
    return np.random.beta(alpha, beta)

@njit
def random_idx(arr_len):
    return np.random.randint(arr_len)

class GridSampler:
    def __init__(self, alpha_row, beta_row, alpha_col, beta_col, obs, N=10, CE=False):
        self.alpha_row = alpha_row
        self.beta_row = beta_row
        self.alpha_col = alpha_col
        self.beta_col = beta_col
        self.obs = obs
        self.CE = CE
        if self.obs is None:
            self.obs = np.array([])
        else:
            self.obs = np.array([(int(i), int(j), float(cost)) for (i, j, cost) in self.obs])
        self.N = N
        self.high_cost = -0.9
        self.low_cost = -0.1

        ## cache obs groups for lazy sampling
        self.row_to_obs = {i: [(i, j, cost) for (i_, j, cost) in self.obs if i_ == i] for i in range(self.N)}
        self.col_to_obs = {j: [(i, j, cost) for (i, j_, cost) in self.obs if j_ == j] for j in range(self.N)}
        

    def init_pqs(self):
        if self.CE:
            self.row_probs = np.full(self.N, self.alpha_row / (self.alpha_row + self.beta_row))
            self.col_probs = np.full(self.N, self.alpha_col / (self.alpha_col + self.beta_col))
            for o in self.obs:
                if o[0] < self.N:
                    self.row_probs[int(o[0])] = np.random.beta(self.alpha_row, self.beta_row)
                if o[1] < self.N:
                    self.col_probs[int(o[1])] = np.random.beta(self.alpha_col, self.beta_col)
        else:
            self.row_probs = np.random.beta(self.alpha_row, self.beta_row, size=self.N)
            self.col_probs = np.random.beta(self.alpha_col, self.beta_col, size=self.N)

    

    
    ## generate beta parameters for the proposal distribution
    def proposal_params(self, index, is_row):

        # Get relevant observations for this row/column
        if is_row:
            rel_obs = self.row_to_obs[index]
            prior_mean_failure = 1-(
                self.beta_col / (2*(self.alpha_col + self.beta_col))
            )
        else:
            rel_obs = self.col_to_obs[index]
            prior_mean_failure = 1-(
                self.beta_row / (2*(self.alpha_row + self.beta_row))
            )

        ### Count occurrences of each cost

        ## standard case (i.e. pq = p(high cost))
        # m = np.sum([cost == self.high_cost for (_, _, cost) in rel_obs])
        # n = prior_mean_failure * np.sum([cost == self.low_cost for (_, _, cost) in rel_obs])

        ## alternative case (i.e. pq = p(low cost))
        m = np.sum([cost == self.low_cost for (_, _, cost) in rel_obs])
        n = prior_mean_failure * np.sum([cost == self.high_cost for (_, _, cost) in rel_obs])

        ## normalise counts to cap their magnitude
        cap = np.max([5, self.alpha_row + self.beta_row, self.alpha_col + self.beta_col])
        total_count = m + n
        if total_count > cap:
            m = cap * m / total_count
            n = cap * n / total_count

        # Update Beta parameters based on observed data
        alpha_prop = self.alpha_row + m if is_row else self.alpha_col + m
        beta_prop = self.beta_row + n if is_row else self.beta_col + n

        return alpha_prop, beta_prop, m, n

    def update(self):
        sampled_obs = self.obs[random_idx(len(self.obs))]
        sampled_i, sampled_j = sampled_obs[0], sampled_obs[1]
        sampled_i, sampled_j = int(sampled_i), int(sampled_j)
        current_p = self.row_probs[sampled_i]
        current_q = self.col_probs[sampled_j]


        ## generate proposals
        alpha_p, beta_p, m1, n1 = self.proposal_params(sampled_i, is_row=True)
        alpha_q, beta_q, m2, n2 = self.proposal_params(sampled_j, is_row=False)
        proposed_p = propose(alpha_p, beta_p)
        proposed_q = propose(alpha_q, beta_q)

        ## subselect relevant obs (i.e. those containing i or j)
        rel_obs = np.array([(int(i), int(j), cost) for i, j, cost in self.obs if i == sampled_i or j == sampled_j])
        # rel_obs = [(i_, j_, cost_) for (i_, j_, cost_) in self.obs if (i_ == sampled_i) or (j_ == sampled_j)]

        ## calculate likelihoods
        log_likelihood = compute_log_likelihood(sampled_i, sampled_j, rel_obs, proposed_p, proposed_q,
                                                self.row_probs[sampled_i], self.col_probs[sampled_j], self.high_cost, self.low_cost)

        ## calculate prior * transition prob terms
        proposal_distr_num = (m1 * np.log(current_p) + n1 * np.log(1 - current_p) + 
                        m2 * np.log(current_q) + n2 * np.log(1 - current_q))
        proposal_distr_den = (m1 * np.log(proposed_p) + n1 * np.log(1 - proposed_p) +
                        m2 * np.log(proposed_q) + n2 * np.log(1 - proposed_q))        
        
        ## acceptance ratio
        log_acceptance_ratio = log_likelihood + proposal_distr_num - proposal_distr_den
        acceptance_ratio = np.exp(log_acceptance_ratio)
        
        ## GPT rubbish
        # alpha_p, beta_p = proposal_params(rel_obs, self.high_cost, self.low_cost, self.alpha_row, self.beta_row, current_p)
        # alpha_q, beta_q = proposal_params(rel_obs, self.high_cost, self.low_cost, self.alpha_col, self.beta_col, current_q)

        # proposed_p = propose(alpha_p, beta_p)
        # proposed_q = propose(alpha_q, beta_q)

        # log_likelihood = compute_log_likelihood(sampled_i, sampled_j, rel_obs, proposed_p, proposed_q,
        #                                         self.row_probs, self.col_probs, self.high_cost, self.low_cost)

        # acceptance_ratio = np.exp(log_likelihood)

        if np.random.random() < min(1, acceptance_ratio):
            self.row_probs[sampled_i] = proposed_p
            self.col_probs[sampled_j] = proposed_q

    def lazy_sample(self, n_iter=100):
        self.init_pqs()
        for _ in range(n_iter):
            self.update()
        return self.row_probs, self.col_probs