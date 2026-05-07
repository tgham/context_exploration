import numpy as np
import random as _random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from abc import ABC, abstractmethod
import copy

from samplers import BanditSampler, EmpSampler


class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    info:
        Info about the environment that the agents is not supposed to know. For instance,
        info can releal the index of the optimal arm, or the value of prior parameter.
        Can be useful to evaluate the agent's perfomance
    """
    def __init__(self, p_dist, r_dist, ):
        if len(p_dist) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.sim = False
        self.info = {} ## dummy info for compatibility with gym API

        self._trial = 0
        self.n_trials = 100

        self.n_afc = len(p_dist)
        self.action_space = spaces.Discrete(self.n_afc)
        self.observation_space = spaces.box.Box(-1.0, 1.0, (1,)) #
        #self.observation_space = spaces.Discrete(1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    ## set the horizon trial in the env
    def set_trunc_trial(self, trunc_trial):
        self.trunc_trial = trunc_trial

    def step(self, action):
        reward = 0

        ## random payout
        if _random.random() < self.p_dist[action]:

            ## binary reward
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]

            ## reward drawn from normal distribution with mean and sd specified in r_dist
            else:
                reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        ## obs as lightweight tuple
        trial_obs = (action, reward)

        ## only track obs history when not in sim mode
        if not self.sim:
            self.obs = np.vstack((self.obs, trial_obs))

            ## Q-learning
            self.Q[action] = self.Q[action] + self.LR * (reward + np.max(self.Q) - self.Q[action])

        ## termination condition determined by n_trials
        terminated = self._trial >= self.n_trials

        ## truncated if we have reached horizon
        truncated = self._trial >= self.trunc_trial

        ## update trial counter
        self._trial += 1

        return trial_obs, reward, terminated, truncated, self.info

    def _reset(self):
        self.set_trunc_trial(self.n_trials-1) ## default is no truncation

        ## init obs
        if self._trial==0:
            self.obs = np.empty((0,2)) ## empty array to store action and reward history
        

    def _render(self, mode='human', close=False):
        pass

    def sim_clone(self, probs):
        
        """
        Returns a lightweight shallow copy of the environment, injecting
        the newly sampled MDP components (e.g. reward probs).
        """
        sim_env = copy.copy(self)

        # Inject the new underlying sampled MDP dynamics
        sim_env.p_dist = probs
        sim_env.sim = True

        return sim_env
    
    ## construct a task-specific sampler for this environment
    def make_sampler(self):
        """Create a GridSampler from this environment's parameters and the given observations."""
        return BanditSampler(self)

############### Bandits written by Tom Graham #################

class BanditNArmedIndependentBeta(BanditEnv):
    """
    n-armed bandit giving a reward of 1 with inDependent probabilities p_1,...,p_n drawn from a Beta distribution
    """
    def __init__(self, bandits=10, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        # p_dist = np.random.beta(alpha, beta, size=bandits)
        
        ## hacky: 
        p_dist = np.zeros(bandits)
        p_dist[0] = 0.9
        p_dist[1:] = 0.6
        
        r_dist = np.full(bandits, 1)

        ## define success and failure 
        self.success = 1
        self.failure = 0

        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

# ---------------------------------------------------------------------------
# Bandit environment wrapper
# ---------------------------------------------------------------------------
class BanditEnvWrapper(BanditNArmedIndependentBeta):
    """Adapts BanditNArmedIndependentBeta to the interface expected by MCTS."""

    def __init__(self, n_arms=5, alpha=1, beta=1, n_trials=20):
        super().__init__(bandits=n_arms, alpha=alpha, beta=beta)
        self.n_trials = n_trials
        self.sim = False

    @property
    def trial(self):
        return self._trial

    @property
    def current(self):
        return self._trial          # no spatial state; trial index suffices

    def reset(self):
        self._reset()

        ## let's also store some Q values...
        self.Q = np.zeros(self.n_afc)
        self.LR = 0.1

    def set_sim(self, sim):
        self.sim = sim

    def step(self, action):
        return super().step(action)

    def receive_task_params(self, task_params):
        pass  # no task-specific params for the bandit


# ---------------------------------------------------------------------------
# Gittins-style bandit with a retirement (safe) arm
# ---------------------------------------------------------------------------
class GittinsBandit(BanditNArmedIndependentBeta):
    """
    N-armed bandit with an additional *retirement* arm (the last arm).

    The first (bandits) arms behave identically to BanditNArmedIndependentBeta.
    The final arm pays out ``lam / (1 - gam)`` with probability 1 and
    terminates the episode immediately when pulled.

    Parameters
    ----------
    bandits : int
        Number of regular (risky) arms.
    alpha, beta : float
        Beta-distribution parameters for the risky arms.
    gam : float
        Discount factor used to define the retirement value.
    lam : float
        Per-period safe reward used to define the retirement value.
    """

    def __init__(self, bandits=10, alpha=1, beta=1, gam=0.9, lam=0.7029):
        # Initialise the risky arms via the parent
        super().__init__(bandits=bandits, alpha=alpha, beta=beta)

        self.gam = gam
        self.lam = lam

        # Append the retirement arm
        retirement_value = lam / (1 - gam)
        self.p_dist = np.append(self.p_dist, 1.0)
        self.r_dist = np.append(self.r_dist, retirement_value)

        # Update action space to include the new arm
        self.n_afc = len(self.p_dist)
        self.action_space = spaces.Discrete(self.n_afc)

    @property
    def retirement_arm(self):
        return self.n_afc - 1


# ---------------------------------------------------------------------------
# Gittins bandit wrapper for MCTS
# ---------------------------------------------------------------------------
class GittinsBanditWrapper(GittinsBandit):
    """Adapts GittinsBandit to the interface expected by MCTS."""

    def __init__(self, n_arms=5, alpha=1, beta=1, gam=0.9, lam=0.7029,
                 n_trials=20):
        super().__init__(bandits=n_arms, alpha=alpha, beta=beta,
                         gam=gam, lam=lam)
        self.n_trials = n_trials
        self.sim = False

    @property
    def trial(self):
        return self._trial

    @property
    def current(self):
        return self._trial

    def reset(self):
        self._reset()
        self.Q = np.zeros(self.n_afc)
        self.LR = 0.1

    def set_sim(self, sim):
        self.sim = sim

    def step(self, action):

        # Pulling the retirement arm: deterministic payout and episode ends
        if action == self.retirement_arm:
            reward = self.r_dist[action]
            trial_obs = (action, reward)

            if not self.sim:
                self.obs = np.vstack((self.obs, trial_obs))

            self._trial += 1
            terminated = True
            truncated = True
            return trial_obs, reward, terminated, truncated, self.info

        # Regular arms: delegate to BanditEnv.step via super()
        return BanditEnv.step(self, action)

    def receive_task_params(self, task_params):
        pass






# ---------------------------------------------------------------------------
# Empowerment bandit: each arm induces a categorical distribution over m outcomes
# ---------------------------------------------------------------------------
class EmpBandit(BanditEnv):
    """
    n-armed bandit where each arm a induces a categorical distribution over
    m possible outcomes. Pulling arm a samples an outcome o ~ Cat(P[a, :]).

    Parameters
    ----------
    n_arms : int
        Number of actions.
    n_outcomes : int
        Number of possible outcomes per arm.
    p_matrix : np.ndarray, optional
        (n_arms, n_outcomes) row-stochastic matrix. If None, rows are drawn
        from a symmetric Dirichlet(alpha).
    alpha : float
        Dirichlet concentration used when p_matrix is None.
    """

    def __init__(self, n_arms=5, n_outcomes=3, p_matrix=None, alpha=1.0, ell=1.0,
                 termination_arm=False, seed=None):
        if p_matrix is None:
            p_matrix = np.random.dirichlet(np.full(n_outcomes, alpha), size=n_arms)
            p_matrix = p_matrix.reshape((n_arms, n_outcomes))

        p_matrix = np.asarray(p_matrix, dtype=float)

        if p_matrix.shape != (n_arms, n_outcomes):
            raise ValueError("p_matrix must have shape (n_arms, n_outcomes)")
        if not np.allclose(p_matrix.sum(axis=1), 1.0):
            raise ValueError("Each row of p_matrix must sum to 1")

        self.n_outcomes = n_outcomes
        self.p_matrix = p_matrix
        self.alpha = alpha
        self.alphas = np.full_like(p_matrix, alpha)  # Dirichlet parameters for each arm
        self.posterior_p_matrix = self.alphas/self.alphas.sum(axis=1, keepdims=True)
        self.ell = ell

        self.sim = False
        self.info = {}

        self._trial = 0
        self.n_trials = 10

        self.n_arms = n_arms
        self.termination_arm = bool(termination_arm)
        self.terminate_action = n_arms if self.termination_arm else None
        self.n_afc = n_arms + int(self.termination_arm)
        self.action_space = spaces.Discrete(self.n_afc)
        self.observation_space = spaces.Discrete(n_outcomes)

        self._seed(seed)

    @staticmethod
    def empowerment(p_matrix, ell):
        """Emp_ell(theta) = sum_{s'} [max_a p(s'|a)]^ell."""
        return float(np.sum(np.max(p_matrix, axis=0) ** ell))
    

    def posterior_update(self, action, outcome):
        self.alphas[action, outcome] += 1
        self.posterior_p_matrix[action, :] = self.alphas[action, :] / self.alphas[action, :].sum()

    def step(self, action):
        ## voluntary termination: collect current empowerment, end episode, no posterior update
        if self.termination_arm and action == self.terminate_action:
            trial_obs = (action, -1)
            reward = self.empowerment(self.posterior_p_matrix, self.ell)
            if not self.sim:
                self.obs = np.vstack((self.obs, trial_obs))
            self._trial += 1
            return trial_obs, reward, True, False, self.info

        outcome = int(self.np_random.choice(self.n_outcomes, p=self.p_matrix[action]))
        trial_obs = (action, outcome)

        if not self.sim:
            self.obs = np.vstack((self.obs, trial_obs))
            self.posterior_update(action, outcome)

        terminated = self._trial >= self.n_trials
        truncated = self._trial >= self.trunc_trial
        self._trial += 1
        if terminated or truncated or not self.sim:
            reward = self.empowerment(self.posterior_p_matrix, self.ell)
        else:
            reward = 0.0

        return trial_obs, reward, terminated, truncated, self.info

    def sim_clone(self, p_matrix):
        sim_env = copy.copy(self)
        sim_env.p_matrix = np.asarray(p_matrix, dtype=float)
        sim_env.posterior_p_matrix = sim_env.p_matrix
        sim_env.sim = True
        return sim_env

    def make_sampler(self):
        return EmpSampler(self)


# ---------------------------------------------------------------------------
# Empowerment bandit wrapper for MCTS
# ---------------------------------------------------------------------------
class EmpBanditWrapper(EmpBandit):
    """Adapts EmpBandit to the interface expected by MCTS."""

    def __init__(self, n_arms=5, n_outcomes=3, alpha=1.0, ell=1.0, n_trials=20,
                 termination_arm=False, seed=None):
        super().__init__(n_arms=n_arms, n_outcomes=n_outcomes, alpha=alpha, ell=ell,
                         termination_arm=termination_arm, seed=seed)
        self.n_trials = n_trials

    @property
    def trial(self):
        return self._trial

    @property
    def current(self):
        return self._trial

    def reset(self):
        self._reset()
        self.Q = np.zeros(self.n_afc)
        self.LR = 0.1

    def set_sim(self, sim):
        self.sim = sim

    def receive_task_params(self, task_params):
        pass

