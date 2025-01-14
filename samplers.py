from numba import njit
import numpy as np

@njit
def compute_log_likelihood(sampled_i, sampled_j, rel_obs, row_p, col_q, row_probs, col_probs, high_cost, low_cost):
    log_likelihood = 0.0
    for o in rel_obs:
        i, j, cost = int(o[0]), int(o[1]), o[2]
        if (i == sampled_i) and (j != sampled_j):
            rel_p = row_p
            rel_q = col_probs[j]
        elif (j == sampled_j) and (i != sampled_i):
            rel_p = row_probs[i]
            rel_q = col_q
        elif (i == sampled_i) and (j == sampled_j):
            rel_p = row_p
            rel_q = col_q
        else:
            raise ValueError("Observation does not match row or column.")

        prob_tmp = rel_p * rel_q

        if cost == high_cost:
            log_likelihood += np.log(1 - prob_tmp)
            log_likelihood -= np.log(1 - row_probs[sampled_i] * col_probs[sampled_j])
        elif cost == low_cost:
            log_likelihood += np.log(prob_tmp)
            log_likelihood -= np.log(row_probs[sampled_i] * col_probs[sampled_j])

    return log_likelihood

@njit
def propose(alpha, beta):
    return np.random.beta(alpha, beta)

@njit
def proposal_params(obs, high_cost, low_cost, alpha, beta, sampled_prob):
    alpha_new, beta_new = alpha, beta
    for o in obs:
        cost = o[2]
        if cost == high_cost:
            beta_new += 1
        elif cost == low_cost:
            alpha_new += 1
        else:
            raise ValueError("Invalid cost encountered in observations.")
    return alpha_new, beta_new

import numpy as np
from numba import njit

class GridSampler:
    def __init__(self, alpha_row, beta_row, alpha_col, beta_col, obs, N=10, CE=False):
        self.alpha_row = alpha_row
        self.beta_row = beta_row
        self.alpha_col = alpha_col
        self.beta_col = beta_col
        self.obs = obs
        if self.obs is None:
            self.obs = np.array([])
        else:
            self.obs = np.array([(int(i), int(j), float(cost)) for (i, j, cost) in self.obs])
        self.N = N
        self.high_cost = -0.9
        self.low_cost = -0.1

        if CE:
            self.row_probs = np.full(self.N, self.alpha_row / (self.alpha_row + self.beta_row))
            self.col_probs = np.full(self.N, self.alpha_col / (self.alpha_col + self.beta_col))
            for o in self.obs:
                if o[0] < self.N:
                    self.row_probs[o[0]] = np.random.beta(self.alpha_row, self.beta_row)
                if o[1] < self.N:
                    self.col_probs[o[1]] = np.random.beta(self.alpha_col, self.beta_col)
        else:
            self.row_probs = np.random.beta(self.alpha_row, self.beta_row, size=self.N)
            self.col_probs = np.random.beta(self.alpha_col, self.beta_col, size=self.N)

    def update(self):
        sampled_i, sampled_j, _ = self.obs[np.random.randint(len(self.obs))]
        sampled_i, sampled_j = int(sampled_i), int(sampled_j)
        current_p = self.row_probs[sampled_i]
        current_q = self.col_probs[sampled_j]

        rel_obs = np.array([(int(i), int(j), cost) for i, j, cost in self.obs if i == sampled_i or j == sampled_j])

        alpha_p, beta_p = proposal_params(rel_obs, self.high_cost, self.low_cost, self.alpha_row, self.beta_row, current_p)
        alpha_q, beta_q = proposal_params(rel_obs, self.high_cost, self.low_cost, self.alpha_col, self.beta_col, current_q)

        proposed_p = propose(alpha_p, beta_p)
        proposed_q = propose(alpha_q, beta_q)

        log_likelihood = compute_log_likelihood(sampled_i, sampled_j, rel_obs, proposed_p, proposed_q,
                                                self.row_probs, self.col_probs, self.high_cost, self.low_cost)

        acceptance_ratio = np.exp(log_likelihood)

        if np.random.random() < min(1, acceptance_ratio):
            self.row_probs[sampled_i] = proposed_p
            self.col_probs[sampled_j] = proposed_q

    def lazy_sample(self, n_iter=100):
        for _ in range(n_iter):
            self.update()
        return self.row_probs, self.col_probs