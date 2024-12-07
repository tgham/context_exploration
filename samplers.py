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


    def proposal_params(self, index, is_row):
        """
        Calculate the Beta parameters for the proposal distribution.
        """
        # Get relevant observations for this row/column
        rel_obs = self.row_to_obs[index] if is_row else self.col_to_obs[index]
        prior_mean_failure = (
            self.beta_col / (self.alpha_col + self.beta_col) if is_row else self.beta_row / (self.alpha_row + self.beta_row)
        )

        # Count occurrences of each cost
        m = sum(-cost for (_, _, cost) in rel_obs if cost == self.high_cost)
        n = (1 - prior_mean_failure) * sum(-cost for (_, _, cost) in rel_obs if cost == self.low_cost)


    
        # Update Beta parameters based on observed data
        alpha_prop = self.alpha_row + m if is_row else self.alpha_col + m
        beta_prop = self.beta_row + n if is_row else self.beta_col + n

        return alpha_prop, beta_prop

    def propose(self, alpha, beta):
        """
        Sample from a Beta distribution with given alpha and beta parameters.
        """
        return np.random.beta(alpha, beta)

    def compute_likelihood(self, row_prob, col_prob, obs):
        """Compute likelihood for a set of observations"""
        likelihood = 1.0
        for (i, j, cost) in obs:
            p = row_prob * col_prob
            likelihood *= p if cost == self.high_cost else (1 - p)
        return likelihood
    
    def compute_acceptance_ratio(self, proposed_prob, current_prob, other_probs, obs, is_row, alpha, beta):
        """Compute the full acceptance ratio including proposal distributions"""
        # Compute likelihoods
        if is_row:
            proposed_likes = [self.compute_likelihood(proposed_prob, q, obs) for q in other_probs]
            current_likes = [self.compute_likelihood(current_prob, q, obs) for q in other_probs]
        else:
            proposed_likes = [self.compute_likelihood(p, proposed_prob, obs) for p in other_probs]
            current_likes = [self.compute_likelihood(p, current_prob, obs) for p in other_probs]
        
        # Proposal densities
        forward_density = scipy.stats.beta.pdf(proposed_prob, alpha, beta)
        backward_density = scipy.stats.beta.pdf(current_prob, alpha, beta)
        
        # Full ratio
        return (np.prod(proposed_likes) * backward_density) / (np.prod(current_likes) * forward_density)

    def update(self):
        """
        Perform a Metropolis-Hastings update for rows and columns with observations.
        """
        # Update observed row probabilities
        for i in self.observed_rows:
            current_prob = self.row_probs[i]
            alpha, beta = self.proposal_params(i, is_row=True)
            proposed_prob = self.propose(alpha, beta)

            ratio = self.compute_acceptance_ratio(
                proposed_prob, current_prob, self.col_probs, self.row_to_obs[i],
                is_row=True, alpha=alpha, beta=beta
            )
            if np.random.random() < min(1, ratio):
                self.row_probs[i] = proposed_prob

        # Update observed column probabilities
        for j in self.observed_cols:
            current_prob = self.col_probs[j]
            alpha, beta = self.proposal_params(j, is_row=False)
            proposed_prob = self.propose(alpha, beta)

            ratio = self.compute_acceptance_ratio(
                proposed_prob, current_prob, self.row_probs, self.col_to_obs[j],
                is_row=False, alpha=alpha, beta=beta
            )
            if np.random.random() < min(1, ratio):
                self.col_probs[j] = proposed_prob

    def sample(self, n_iter=1000):
        """Run the sampler for a specified number of iterations."""
        for _ in range(n_iter):
            self.update()
        return self.row_probs, self.col_probs