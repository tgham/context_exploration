import numpy as np
import scipy

class GridSampler:
    def __init__(self, alpha_row, beta_row, alpha_col, beta_col, obs, N=10):
        """
        Initialize the sampler with prior parameters and observed data.

        Parameters:
        - alpha_row, beta_row: Beta distribution priors for rows.
        - alpha_col, beta_col: Beta distribution priors for columns.
        - obs: A list of tuples (_, i, j,cost) where
          i is the row index, j is the column index, and cost is the observed value (e.g., -0.9 or -0.1).
        - N: The size of the grid (default is 10x10).
        """
        self.alpha_row = alpha_row
        self.beta_row = beta_row
        self.alpha_col = alpha_col
        self.beta_col = beta_col
        self.obs = obs
        if self.obs is None:
            self.obs = np.array([])
        else:
            self.obs = [(int(i), int(j), float(cost)) for (i, j, cost) in self.obs]
        self.N = N
        self.high_cost = -0.9
        self.low_cost = -0.1

        # Initialize row and column probabilities for the entire grid
        self.row_probs = np.random.beta(self.alpha_row, self.beta_row, size=self.N)
        self.col_probs = np.random.beta(self.alpha_col, self.beta_col, size=self.N)

        ## cache obs groups for lazy sampling
        self.row_to_obs = {i: [(i, j, cost) for (i_, j, cost) in self.obs if i_ == i] for i in range(self.N)}
        self.col_to_obs = {j: [(i, j, cost) for (i, j_, cost) in self.obs if j_ == j] for j in range(self.N)}

        # Identify rows and columns with observations
        self.observed_rows = [i for i in range(self.N) if len(self.row_to_obs[i]) > 0]
        self.observed_cols = [j for j in range(self.N) if len(self.col_to_obs[j]) > 0]


    def sample_obs(self):
        """Select a random observation from the data."""
        return self.obs[np.random.randint(len(self.obs))]
    
    def proposal_params(self, index, is_row):
        """
        Calculate the Beta parameters for the proposal distribution.
        """
        # Get relevant observations for this row/column
        rel_obs = self.row_to_obs[index] if is_row else self.col_to_obs[index]
        prior_mean_failure = 1-(
            self.beta_col / (2*(self.alpha_col + self.beta_col)) if is_row else self.beta_row / (2*(self.alpha_row + self.beta_row))
        )

        # Count occurrences of each cost
        m = np.sum([cost == self.high_cost for (_, _, cost) in rel_obs])
        n = prior_mean_failure * np.sum([cost == self.low_cost for (_, _, cost) in rel_obs])
    
        # Update Beta parameters based on observed data
        alpha_prop = self.alpha_row + m if is_row else self.alpha_col + m
        beta_prop = self.beta_row + n if is_row else self.beta_col + n

        return alpha_prop, beta_prop, m, n

    def propose(self, alpha, beta):
        """
        Sample from a Beta distribution with given alpha and beta parameters.
        """
        return np.random.beta(alpha, beta)    

    def calculate_likelihood(self, sampled_i, sampled_j, rel_obs, row_p, col_q):
        likelihood = 1.0
        for o in rel_obs:
            if o[0] == sampled_i:
                rel_p = row_p
                rel_q = self.col_probs[o[1]]
            elif o[1] == sampled_j:
                rel_p = self.row_probs[o[0]]
                rel_q = col_q
            else:
                raise ValueError("Observation does not match row or column.")
            rel_cost = o[2]
            prob_tmp = rel_p * rel_q
            likelihood *= prob_tmp**(rel_cost == self.high_cost) * (1 - prob_tmp)**(rel_cost == self.low_cost)
            # likelihood *= prob_tmp**(rel_cost == self.low_cost) * (1 - prob_tmp)**(rel_cost == self.high_cost)
        return likelihood

    def update(self):
        """
        Perform a Metropolis-Hastings update for rows and columns with observations.
        """

        ## sample an observation at random
        sampled_i, sampled_j, cost = self.sample_obs()
        current_p = self.row_probs[sampled_i]
        current_q = self.col_probs[sampled_j]

        ## generate proposals
        alpha_p, beta_p, m1, n1 = self.proposal_params(sampled_i, is_row=True)
        alpha_q, beta_q, m2, n2 = self.proposal_params(sampled_j, is_row=False)
        proposed_p = self.propose(alpha_p, beta_p)
        proposed_q = self.propose(alpha_q, beta_q)

        ## subselect relevant obs (i.e. those containing i or j)
        rel_obs = [(i_, j_, cost_) for (i_, j_, cost_) in self.obs if (i_ == sampled_i) or (j_ == sampled_j)]

        ## calculate likelihoods
        likelihood_num = self.calculate_likelihood(sampled_i, sampled_j, rel_obs, proposed_p, proposed_q)
        likelihood_den = self.calculate_likelihood(sampled_i, sampled_j, rel_obs, current_p, current_q)

        ## calculate prior * transition prob terms
        prior_num = proposed_p**m1 * (1 - proposed_p)**n1 * proposed_q**m2 * (1 - proposed_q)**n2
        prior_den = current_p**m1 * (1 - current_p)**n1 * current_q**m2 * (1 - current_q)**n2
        # prior_num = current_p**m1 * (1 - current_p)**n1 * current_q**m2 * (1 - current_q)**n2
        # prior_den = proposed_p**m1 * (1 - proposed_p)**n1 * proposed_q**m2 * (1 - proposed_q)**n2

        ## calculate acceptance ratio
        epsilon = 1e-10
        acceptance_ratio = (likelihood_num * prior_num) / (likelihood_den * prior_den + epsilon)
        if np.random.random() < min(1, acceptance_ratio):
            self.row_probs[sampled_i] = proposed_p
            self.col_probs[sampled_j] = proposed_q


    def sample(self, n_iter=10000):
        """Run the sampler for a specified number of iterations."""
        for _ in range(n_iter):
            self.update()
        return self.row_probs, self.col_probs