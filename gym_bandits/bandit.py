import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from abc import ABC, abstractmethod
import copy

from samplers import BanditSampler


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
        if not self.sim:
            assert self.action_space.contains(action)

        reward = 0

        ## random payout
        if np.random.uniform() < self.p_dist[action]:

            ## binary reward
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]

            ## reward drawn from normal distribution with mean and sd specified in r_dist
            else:
                reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        ## obs defined as action and reward r?
        trial_obs = np.array([action, reward])

        ## only track obs history when not in sim mode
        if not self.sim:
            self.obs = np.vstack((self.obs, trial_obs))

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

    def set_sim(self, sim):
        self.sim = sim

    def step(self, action):
        trial_obs, reward, terminated, truncated, info = super().step(action)
        # MCTS uses sum(rewards), so wrap scalar reward in a list
        return trial_obs.reshape(1, -1), (reward,), terminated, truncated, info

    def receive_task_params(self, task_params):
        pass  # no task-specific params for the bandit





# ############### Bandits written by Thomas Lecat #################

# class BanditTwoArmedIndependentUniform(BanditEnv):
# 	"""
# 	2 armed bandit giving a reward of 1 with inDependent probabilities p_1 and p_2
# 	"""
# 	def __init__(self, bandits=2):
# 		p_dist = np.random.uniform(size=bandits)
# 		r_dist = np.full(bandits,1)

# 		BanditEnv.__init__(self, p_dist = p_dist, r_dist=r_dist)

# class BanditTwoArmedDependentUniform(BanditEnv):
#     """
#     2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0,1] and p_2 = 1 - p_1
#     """
#     def __init__(self):
#         p = np.random.uniform()
#         p_dist = [p, 1-p]
#         r_dist = [1, 1]
#         info={'parameter':p}
#         BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info = info)

# class BanditTwoArmedDependentEasy(BanditEnv):
#     """
#     2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0.1,0.9] and p_2 = 1 - p_1
#     """
#     def __init__(self):
#         p = [0.1,0.9][np.random.randint(0,2)]
#         p_dist = [p, 1-p]
#         r_dist = [1, 1]
#         optimal_arm = np.abs(1-int(round(p)))
#         info = {'optimal_arm':optimal_arm}
#         BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)

# class BanditTwoArmedDependentMedium(BanditEnv):
#     """
#     2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0.25,0.75] and p_2 = 1 - p_1
#     """
#     def __init__(self):
#         p = [0.25,0.75][np.random.randint(0,2)]
#         p_dist = [p, 1-p]
#         r_dist = [1, 1]
#         optimal_arm = np.abs(1-int(round(p)))
#         info = {'optimal_arm':optimal_arm}
#         BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)

# class BanditTwoArmedDependentHard(BanditEnv):
#     """
#     2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0.4,0.6] and p_2 = 1 - p_1
#     """
#     def __init__(self):
#         p = [0.4,0.6][np.random.randint(0,2)]
#         p_dist = [p, 1-p]
#         r_dist = [1, 1]
#         optimal_arm = np.abs(1-int(round(p)))
#         info = {'optimal_arm':optimal_arm}
#         BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)

# class BanditElevenArmedWithIndex(BanditEnv):
#     """
#     11 armed bandit:
#     1 out of the 10 first arms gives a reward of 5 (optimal arm), the 9 other arms give reward of 1.1. The 11th arm gives a reward of 0.1 * index of the optimal arm.
#     """
#     def __init__(self):
#         index = np.random.randint(0,10)
#         p_dist = np.full(11,1)
#         r_dist = np.full(11,1.1)
#         r_dist[index] = 5
#         r_dist[-1] = 0.1*(index+1) # Note: we add 1 because the arms are indexed from 1 to 10, not 0 to 9
#         info = {'optimal_arm':10*index}
#         BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)


# ###################### Bandits written by Jesse Coopper #########################

# class BanditTwoArmedDeterministicFixed(BanditEnv):
#     """Simplest case where one bandit always pays, and the other always doesn't"""
#     def __init__(self):
#         BanditEnv.__init__(self, p_dist=[1, 0], r_dist=[1, 1], info={'optimal_arm':1})


# class BanditTwoArmedHighLowFixed(BanditEnv):
#     """Stochastic version with a large difference between which bandit pays out of two choices"""
#     def __init__(self):
#         BanditEnv.__init__(self, p_dist=[0.8, 0.2], r_dist=[1, 1], info={'optimal_arm':1})


# class BanditTwoArmedHighHighFixed(BanditEnv):
#     """Stochastic version with a small difference between which bandit pays where both are good"""
#     def __init__(self):
#         BanditEnv.__init__(self, p_dist=[0.8, 0.9], r_dist=[1, 1], info={'optimal_arm':2})


# class BanditTwoArmedLowLowFixed(BanditEnv):
#     """Stochastic version with a small difference between which bandit pays where both are bad"""
#     def __init__(self):
#         BanditEnv.__init__(self, p_dist=[0.1, 0.2], r_dist=[1, 1], info={'optimal_arm':2})


# class BanditTenArmedRandomFixed(BanditEnv):
#     """10 armed bandit with random probabilities assigned to payouts"""
#     def __init__(self, bandits=10):
#         p_dist = np.random.uniform(size=bandits)
#         r_dist = np.full(bandits, 1)
#         BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


# class BanditTenArmedUniformDistributedReward(BanditEnv):
#     """10 armed bandit with that always pays out with a reward selected from a uniform distribution"""
#     def __init__(self, bandits=10):
#         p_dist = np.full(bandits, 1)
#         r_dist = np.random.uniform(size=bandits)
#         BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


# class BanditTenArmedRandomRandom(BanditEnv):
#     """10 armed bandit with random probabilities assigned to both payouts and rewards"""
#     def __init__(self, bandits=10):
#         p_dist = np.random.uniform(size=bandits)
#         r_dist = np.random.uniform(size=bandits)
#         BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


# class BanditTenArmedGaussian(BanditEnv):
#     """
#     10 armed bandit mentioned on page 30 of Sutton and Barto's
#     [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0)

#     Actions always pay out
#     Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
#     Actual reward is drawn from a normal distribution (q*(a), 1)
#     """
#     def __init__(self, bandits=10):
#         p_dist = np.full(bandits, 1)
#         r_dist = []

#         for i in range(bandits):
#             r_dist.append([np.random.normal(0, 1), 1])

#         BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)
