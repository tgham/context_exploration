import numpy as np
from scipy.stats import beta

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
        self.N = N
        self.min_cost = -0.9
        self.max_cost = -0.1

        # Initialize row and column probabilities for the entire grid
        self.row_probs = np.random.beta(self.alpha_row, self.beta_row, size=self.N)
        self.col_probs = np.random.beta(self.alpha_col, self.beta_col, size=self.N)

    def proposal_params(self, index, is_row):
        """
        Calculate the Beta parameters for the proposal distribution.

        Parameters:
        - index: The row or column index.
        - is_row: Boolean indicating whether to compute for a row (True) or column (False).

        Returns:
        - Tuple (alpha_proposal, beta_proposal).
        """
        # Separate observed costs for the row or column
        rel_obs = [
            cost for (_, i, j,cost) in self.obs if (i == index and is_row) or (j == index and not is_row)
        ]

        # Count occurrences of each cost
        m = sum(-cost for cost in rel_obs if cost == self.min_cost)  # Weighted count for high-cost observations
        n = sum(-cost for cost in rel_obs if cost == self.max_cost)  # Weighted count for low-cost observations

        # Update Beta parameters based on observed data
        if is_row:
            alpha_prop = self.alpha_row + m
            beta_prop = self.beta_row + n
        else:
            alpha_prop = self.alpha_col + m
            beta_prop = self.beta_col + n

        return alpha_prop, beta_prop

    def propose(self, alpha, beta):
        """
        Sample from a Beta distribution with given alpha and beta parameters.
        """
        return np.random.beta(alpha, beta)

    def likelihood_ratio(self, proposed_prob, current_prob, obs):
        """
        Calculate the likelihood ratio of the proposed and current probabilities.

        Parameters:
        - proposed_prob: The proposed probability for the row or column.
        - current_prob: The current probability for the row or column.
        - obs: The relevant observations for this row or column.

        Returns:
        - Likelihood ratio (float).
        """
        likelihood = 1.0
        for (i, j,cost) in obs:
            prob = proposed_prob
            likelihood *= np.exp(-cost * prob)  # Costs (-0.9 or -0.1) weighted by probability
        return likelihood / max(1e-8, np.exp(-sum(cost * current_prob for (_, _, cost) in obs)))

    def update(self):
        """
        Perform a single Metropolis-Hastings update for all rows and columns.
        """
        # Update all row probabilities
        for i in range(self.N):
            current_prob = self.row_probs[i]
            alpha, beta = self.proposal_params(i, is_row=True)
            proposed_prob = self.propose(alpha, beta)

            # Relevant observations for this row
            row_obs = [(i, j,cost) for (_, i_, j, cost) in self.obs if i_ == i]
            likelihood_ratio = self.likelihood_ratio(proposed_prob, current_prob, row_obs)

            # Accept or reject the proposal
            if np.random.rand() < min(1, likelihood_ratio):
                self.row_probs[i] = proposed_prob

        # Update all column probabilities
        for j in range(self.N):
            current_prob = self.col_probs[j]
            alpha, beta = self.proposal_params(j, is_row=False)
            proposed_prob = self.propose(alpha, beta)

            # Relevant observations for this column
            col_obs = [(i, j,cost) for (_, i, j_, cost) in self.obs if j_ == j]
            likelihood_ratio = self.likelihood_ratio(proposed_prob, current_prob, col_obs)

            # Accept or reject the proposal
            if np.random.rand() < min(1, likelihood_ratio):
                self.col_probs[j] = proposed_prob

    def sample(self, n_iter=1000):
        """
        Run the sampler for a specified number of iterations.
        """
        for _ in range(n_iter):
            self.update()
        return self.row_probs, self.col_probs
