from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from plotter import *
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
from scipy.special import softmax
import warnings
import heapq 
from collections import defaultdict
from IPython.display import display, clear_output
from utils import *
from base_kernels import *
from scipy.special import beta, logsumexp, digamma, comb, betaln


### base farmer model
class Farmer(ABC):

    def __init__(self,
                 mcts_class=None, run_fn=None, 
                 temp=1, lapse=0, horizon=3,
                 exploration_constant=None, discount_factor=None, n_samples=None,
                 **task_params):

        ## behavioural parameters
        self.temp = temp
        self.lapse = lapse

        ## task-specific parameters (passed through to env via receive_task_params)
        self.task_params = task_params

        ## MCTS parameters
        self.horizon = horizon
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.n_samples = n_samples
        self._mcts_class = mcts_class

        ## run function
        self.run_fn = run_fn
    
    
    ### general methods for sampling, choice, fitting etc.

    ## create a sampler object from the environment's current state
    def init_sampler(self, env):
        """Delegate to the environment's task-specific sampler factory."""

        ## pass task-specific parameters to env (not all envs support this)
        if hasattr(env, 'receive_task_params'):
            env.receive_task_params(self.task_params)

        ## make sampler if there is not one, otherwise we just need to update the sampler's observations
        # if not hasattr(self, 'sampler') or self.sampler is None:
        #     self.sampler = env.make_sampler()
        # else:
        #     self.sampler.set_obs(env.obs)
        self.sampler = env.make_sampler()

        ## if dealing with context priors, give this to the sampler
        if hasattr(self, 'context_prior'):
            self.sampler.context_prior = self.context_prior

            
    ## agent-specific calculation of Q values based on posterior
    @abstractmethod
    def compute_Q(self, env_copy, tree_reset=True):
        pass

    ## CE Q values: act based on posterior mean
    def compute_CE_Q(self, env_copy):

        ## get task-specific sampler
        self.init_sampler(env_copy)

        ## get posterior mean grid
        CE_Q = self.sampler.posterior_mean_val()
        return CE_Q

    ## choice function
    def softmax(self, Q):
        CPs = (1-self.lapse) * softmax(Q/self.temp) + self.lapse/len(Q)
        return CPs


    ## loss function
    def loss_func(self, df_trials):

        ## flatten + other init
        self.p_choice_flat = self.p_choice[:,:,:,1].flatten() ## i.e. p(choose path B)
        if len(self.p_choice_flat) != len(df_trials):
            # warnings.warn('p_choice_flat length does not match df_trials length. Check your data!')
            print('p_choice_flat length does not match df_trials length for participant {}. Truncating p_choice_flat to match df_trials length.'.format(df_trials['pid'].values[0]))
            self.p_choice_flat = self.p_choice_flat[:len(df_trials)] ## i.e. truncate to match df_trials length
        # self.p_choice_flat = self.p_choice_flat[~np.isnan(self.p_choice_flat)]
        self.ppt_choices = (df_trials['path_chosen']=='b').values


        ## numerical stability
        self.p_choice_flat[(self.p_choice_flat==0) & (self.ppt_choices)] = 0 + np.finfo(float).tiny
        self.p_choice_flat[(self.p_choice_flat==1) & (~self.ppt_choices)] = 1 - np.finfo(float).eps

        ## negative log likelihood
        self.trial_loss[self.ppt_choices] = np.log(self.p_choice_flat[self.ppt_choices])
        self.trial_loss[~self.ppt_choices] = np.log((1-self.p_choice_flat[~self.ppt_choices]))
        self.loss = -np.nansum(self.trial_loss)


    ## pseudo r^2
    def pseudo_r2(self, df_trials):
        
        ## calculate loss under null (random choice)
        n_trials = len(df_trials)
        p_choice_null = np.ones(n_trials) * 0.5
        loss_null = -np.sum(np.log(p_choice_null))

        ## pseudo r^2
        pseudo_r2 = 1 - (self.loss / loss_null)

        ## LLR test?
        llr = 2 * (loss_null - self.loss)
        df = 2
        p_value = scipy.stats.chi2.sf(llr, df)

        return pseudo_r2, p_value
    

    ## run agent — delegates to the injected run function
    def run(self, *args, **kwargs):
        return self.run_fn(self, *args, **kwargs)



## define subclasses
class BAMCP(Farmer):

    def __init__(self,
                 mcts_class, run_fn,
                 temp=1, lapse=0, horizon=3,
                 exploration_constant=None, discount_factor=None, n_samples=None,
                 **task_params):
        super().__init__(
            mcts_class, run_fn,
            temp, lapse, horizon, exploration_constant, discount_factor, n_samples,
            **task_params)
        self.arm_weight = task_params.get('arm_weight', 0)


    ## initialise MCTS object for tree search
    def init_mcts(self, env, reset=True):
        """
        Initialise or update the MCTS object for tree search.

        Args:
            env: The environment to use for MCTS.
            reset: If True, create a new MCTS object. If False, update the existing one.
        """
        if reset:
            tree = Tree()
            self.mcts = self._mcts_class(
                env=env,
                tree=tree,
                exploration_constant=self.exploration_constant,
                discount_factor=self.discount_factor,
                horizon=self.horizon,
            )
        else:
            self.mcts.refresh_env(env)


    ## tree search using this agent's internal MCTS object
    def search(self):
        """
        Perform MCTS search using this agent's internal MCTS object.
        
        Args:
            n_samples: Number of MCTS samples to use for the search.
            
        Returns:
            action: The selected action.
            MCTS_estimates: The Q-value estimates for each action.
        """
        if self.mcts is None:
            raise ValueError("MCTS object has not been initialized. Call run() with agent='BAMCP' first.")

        ## check root
        assert self.mcts.root_trial == self.mcts.env.trial, 'trial mismatch between env and tree at start of search\n env trial: {} \n tree trial: {}'.format(self.mcts.env.trial, self.mcts.root_trial)

        ## generate new set of root samples
        self.all_posterior_MDPs = self.sampler.sample_mdps(self.n_samples)

        ## debugging Q-vals
        self.mcts.Q_tracker = []
        self.mcts.return_tracker = []
        self.mcts.first_node_updates = []
        self.mcts.first_node_updates_by_depth = []
        self.mcts.tree_cost_tracker = []
        self.mcts.conditional_tree_cost_tracker = [[] for _ in range(self.mcts.n_afc)]
        for t in range(self.mcts.env.n_trials):
            self.mcts.first_node_updates_by_depth.append([])
            self.mcts.tree_cost_tracker.append([])
            for a in range(self.mcts.n_afc):
                self.mcts.conditional_tree_cost_tracker[a].append([])
        
        ## loop through simulations
        for s in range(self.n_samples):
            
            ## root sampling of new posterior
            # posterior_MDP = self.all_posterior_MDPs[s]
            # self.mcts.env.receive_predictions(posterior_MDP)
            self.mcts.env = self.all_posterior_MDPs[s]

            ## selection, expansion, simulation
            action_leaf = self.mcts.traverse_tree()
            self.mcts.rollout(action_leaf)
            
            ##backup
            self.mcts.backup()

            ## update Q tracker
            try:
                Qs = [self.mcts.tree.root.action_leaves[a].performance for a in self.mcts.tree.root.action_leaves.keys()]
                self.mcts.Q_tracker.append(Qs)
            except:
                pass

        ## return final Q estimates
        MCTS_estimates = np.full(self.mcts.n_afc, np.nan)
        for action, leaf in self.mcts.tree.root.action_leaves.items():
            MCTS_estimates[action] = leaf.performance

        return MCTS_estimates


    ## get MCTS Q estimates
    def compute_Q(self, env_copy, tree_reset=True):

        ## get task-specific sampler
        self.init_sampler(env_copy)

        ## reset tree (or reuse it)
        self.init_mcts(env=env_copy, reset=tree_reset)
        assert self.mcts.env.trial == self.mcts.root_trial, 'trial mismatch between env and MCTS\n env: {} \n MCTS: {}'.format(env_copy.trial, self.mcts.root_trial)
        assert self.mcts.env.sim == True, 'env not in sim mode'


        ## search
        MCTS_Q = self.search()

        # ## debugging plot
        # # toplot_mean = np.mean([MDP.costs == self.sampler.low_cost for MDP in self.all_posterior_MDPs], axis=0) ## mean of binary grids
        # toplot_mean = self.sampler.mean_mdp() ## or actual posterior mean
        # fig, axs = plt.subplots(1,1, figsize=(5,5))
        # plot_r(toplot_mean, axs, title = 'Posterior reward distribution\nmean of all root samples\nMCTS Q: {}'.format(np.round(MCTS_Q,2)))
        # plot_traj([env_copy.path_states[self.mcts.root_trial][c] for c in range(self.mcts.n_afc)], ax = axs)
        # plot_obs(env_copy.obs, ax = axs, text=True)
        # plt.show()

        return MCTS_Q
    

    ## update the MCTS tree
    def update_tree(self, env_copy, action):

        ## prune tree (not always successful due to high branching factor, or if participant made no choice in which case reset the tree)
        init_info_state = self.mcts.tree.root.node_id
        trial_obs = env_copy.trial_obs.copy()
        t = self.mcts.root_trial
        next_node_id = self.mcts.init_node_id(trial_obs, init_info_state)

        if next_node_id in self.mcts.tree.root.action_leaves[action].children:
            self.mcts.tree.prune(action, next_node_id)
            # assert np.array_equal(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs), 'error in root update\n root state: {} \n costs: {}'.format(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs)
            # assert np.array_equal(self.mcts.tree.root.belief_state[1:], costs), 'error in root update\n root state: {} \n costs: {}'.format(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs)
            assert self.mcts.tree.root.trial == t+1, 'trial mismatch after pruning\n env trial: {}, MCTS trial: {}'.format(t, self.mcts.tree.root.trial)
            tree_reset = False
        else:
            tree_reset = True

        ## hacky: unless full BAMCP with real future paths and full horizon, reset the tree
        # if (not self.real_future_paths) or (self.horizon < (env_copy.n_trials-t-1)):
        if self.horizon < (env_copy.n_trials-t-1):
            tree_reset = True
    
        return tree_reset
        



class CE(BAMCP):
    def __init__(self,
                 mcts_class, run_fn,
                 temp=1, lapse=0, horizon=None,
                 exploration_constant=None, discount_factor=None, n_samples=None,
                 **task_params):
        super().__init__(mcts_class, run_fn,
                         temp=temp, lapse=lapse, horizon=horizon,
                         exploration_constant=exploration_constant, discount_factor=discount_factor, n_samples=n_samples,
                         **task_params)
        self.arm_weight = task_params.get('arm_weight', 0)


    ## act based on posterior mean grid
    def compute_Q(self, env_copy, tree_reset=None):
        CE_Q = self.compute_CE_Q(env_copy)
        return CE_Q


    ## trivially need to do this
    def update_tree(self, env_copy, action):
        return True  # default: always reset
    




