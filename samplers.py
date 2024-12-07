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

    def proposal_params(self, index, is_row):
        """
        Calculate the Beta parameters for the proposal distribution.
        """
        # Separate observed costs for the row or column
        if is_row:
            rel_obs = [(i, j, cost) for (_, i, j, cost) in self.obs if i == index]
            prior_mean_failure = self.beta_col / (self.alpha_col + self.beta_col)
        else:
            rel_obs = [(i, j, cost) for (_, i, j, cost) in self.obs if j == index]
            prior_mean_failure = self.beta_row / (self.alpha_row + self.beta_row)

        # Count occurrences of each cost
        m = sum(-cost for cost in rel_obs if cost == self.high_cost)  # Weighted count for high-cost observations
        # n = sum(-cost for cost in rel_obs if cost == self.low_cost)  # Weighted count for low-cost observations
        n = (1- prior_mean_failure) * sum(-cost for cost in rel_obs if cost == self.low_cost)  # Weighted count for low-cost observations

    
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

    def compute_likelihood(self, row_prob, col_prob, obs):
        """Compute likelihood for a set of observations"""
        likelihood = 1.0
        for (i, j, cost) in obs:
            p = row_prob * col_prob
            if cost == self.high_cost:
                likelihood *= p
            else:  # low cost
                likelihood *= (1 - p)
            # likelihood *= p**(-np.round(cost)) * (1-p)**(np.round(cost))
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
        Perform a single Metropolis-Hastings update for all rows and columns.
        """
        # Update all row probabilities
        for i in range(self.N):
            current_prob = self.row_probs[i]
            alpha, beta = self.proposal_params(i, is_row=True)
            proposed_prob = self.propose(alpha, beta)

            # Relevant observations for this row
            row_obs = [(i, j,cost) for (_, i_, j, cost) in self.obs if i_ == i]

            # Compute acceptance ratio
            ratio = self.compute_acceptance_ratio(
                proposed_prob, current_prob, self.col_probs, row_obs,
                is_row=True, alpha=alpha, beta=beta
            )
            
            # Accept or reject
            if np.random.random() < min(1, ratio):
                self.row_probs[i] = proposed_prob

        # Update column probabilities
        for j in range(self.N):
            current_prob = self.col_probs[j]
            alpha, beta = self.proposal_params(j, is_row=False)
            proposed_prob = np.random.beta(alpha, beta)
            
            # Get relevant observations
            col_obs = [(i, j, cost) for (_, i, j_, cost) in self.obs if j_ == j]
            
            # Compute acceptance ratio
            ratio = self.compute_acceptance_ratio(
                proposed_prob, current_prob, self.row_probs, col_obs,
                is_row=False, alpha=alpha, beta=beta
            )
            
            # Accept or reject
            if np.random.random() < min(1, ratio):
                self.col_probs[j] = proposed_prob

    def sample(self, n_iter=1000):
        """
        Run the sampler for a specified number of iterations.
        """
        for _ in range(n_iter):
            self.update()
        return self.row_probs, self.col_probs
