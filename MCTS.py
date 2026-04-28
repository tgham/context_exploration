from abc import ABC, abstractmethod
import random
from math import sqrt, log
import numpy as np

## base class 
class MonteCarloTreeSearch():

    def __init__(self, env, tree, exploration_constant=2, discount_factor=0.99, horizon=None):
        
        ## initialize tree and and MCTS params
        self.tree = tree
        self.discount_factor = discount_factor
        self.exploration_constant = exploration_constant
        if horizon is None:
            self.horizon = env.n_trials-1 #default to n_trials-1 (i.e. look to the end of the day, which is up to n_trials-1 from the root)
        else:
            self.horizon = horizon
        
        ## init root info
        self.refresh_env(env)
        self.n_afc = self.env.n_afc

        ## create id for root node
        node_id = self.init_node_id(self.env.obs, None)

        ## add state node to the tree
        self.tree.add_state_node(state = env.current, node_id=node_id, reward=None, terminated=False, trial = self.root_trial, n_afc=self.n_afc, parent=None, 
                                 )

        ## some lists for debugging
        self.first_node_updates = [] ## the updates applied to the first node in the tree during
        self.first_node_updates_by_depth = {d: [] for d in range(self.horizon+1)} ## the updates applied to the first node, conditional on the depth at which the first action leaf is located
        self.tree_reward_tracker = {d: [] for d in range(self.horizon+1)} ## the rewards of each step in the tree, indexed by depth
        self.conditional_tree_reward_tracker = {a: {d: [] for d in range(self.horizon+1)} for a in range(self.n_afc)} ## the rewards of each
    
    ### abstract methods

    ## create node id, which represents the agent's current state of knowledge
    @abstractmethod
    def init_node_id(self, obs=None, init_info_state=None):
        pass

    ## update MCTS object with misc info from the env
    @abstractmethod
    def refresh_env(self, env=None):
        pass
    
    ## rollout policy
    @abstractmethod
    def rollout_policy(self, action_leaf):
        pass
    
    ## debugging method for checking if node's belief state matches the env state
    @abstractmethod
    def check_node(self, node):
        pass
    

    
    ### general MCTS methods
    
    ## expand the action space of a node
    def expand(self, node):
        assert self.env.sim, 'env is not in sim mode'

        ### take action (or path) and get new state
        action = node.untried_action()
            
        ## create new action leaf and attach to node
        node.action_leaves[action] = Action_Node(action=action, trial=node.trial, parent_id=node.node_id)
        node.action_leaves[action].performance = 0
        
        return node.action_leaves[action]
    

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def traverse_tree(self):

        ## initialise the tree
        node = self.tree.root
        node_trial = node.trial

        ## create a record of the nodes/leaves visited in the tree
        self.tree_rewards = [] ## i.e. the reward associated with each traversal of the tree *under the tree policy*. Hence, this does not include the reward of the current state, which is the starting point of the tree policy, nor does it include the reward of expansion.
        self.tree_actions = [] ## i.e. the states and actions visited in the tree. This *does* include the root, because it is from the root that we move to the next leaf (and then next node). 
        self.node_id_path = []
        
        ## loop until you reach a leaf node or terminal state
        assert self.env.sim, 'env is not in sim mode'
        terminated=False
        truncated = False
        while not terminated and not truncated:
            self.check_node(node)

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                return action_leaf
                
            ## selection step
            else:

                ## get the best child
                action_leaf = self.best_child(node)
                assert self.env.trial == action_leaf.trial, 'trial mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(self.env.trial, action_leaf.trial, self.root_trial)

                ## update the tree path
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                ## move in env
                step_obs, reward, terminated, truncated, _ = self.env.step(action_leaf.action)
                self.tree_rewards.append(reward)

                ## continue down the tree if not terminated (NB: WE MAY ACTUALLY WANT TO STILL CREATE THE NODE FOR THE TERMINAL STATE)
                if not terminated and not truncated:

                    ## get the next node id, i.e. the informational state after taking this path
                    next_node_id = self.init_node_id(step_obs, action_leaf.parent_id)
                    node_trial += 1
                    assert node_trial == action_leaf.trial+1, 'trial mismatch between env and tree after step\n env: {} \n tree: {}'.format(node_trial, action_leaf.trial+1)

                    ## see if the next state node already exists as a child of this action leaf
                    if next_node_id in action_leaf.children:
                        node = action_leaf.children[next_node_id]
                    else:

                        ## create new node
                        node = self.tree.add_state_node(state = self.env.current, node_id=next_node_id, reward = reward, terminated=terminated, trial = node_trial, n_afc=self.n_afc, parent=action_leaf,
                                                        )


        ## if terminal node, there are no more action leaves to choose from
        if terminated or truncated:
            action_leaf = None

        return action_leaf
    
    ## rollout policy
    def rollout(self, action_leaf):

        ## if no action leaf because tree policy has reached a terminal node, return None
        if action_leaf is None:
            return None

        ## first need to get the starting reward r, which is essentially the reward of choice that corresponds to the action leaf
        _, reward, terminated, truncated, _ = self.env.step(action_leaf.action)
        total_reward = reward

        ## if final trial, just stop here
        if terminated or truncated:
            self.tree_rewards.append(total_reward)
            return total_reward
        
        ## in bandit problem, we can take an analytical rollout substitute
        # best_arm_p = max(self.env.p_dist)
        # delayed_reward = best_arm_p / (1 - self.discount_factor)
        # total_reward += (self.discount_factor * delayed_reward)
        # self.tree_rewards.append(total_reward)
        # return total_reward 
        
        ## else, loop through remaining trials
        discount_power = 1.0
        remaining_ro_rewards = []
        while not terminated and not truncated:
            discount_power *= self.discount_factor
            ro_action = self.rollout_policy()
            _, reward, terminated, truncated, _ = self.env.step(ro_action)
            total_reward += reward * discount_power
            remaining_ro_rewards.append(total_reward)

        self.tree_rewards.append(total_reward)
        # assert len(remaining_ro_rewards)+first_trial+1 == self.horizon_trial + 1, 'remaining RO rewards do not match number of trials\n n remaining RO rewards: {}, n trials: {}'.format(len(remaining_ro_rewards), self.horizon_trial + 1)
        return total_reward 


    ## backup rewards until you reach the root
    def backup(self):
        tree_len = len(self.tree_rewards)
        assert tree_len == len(self.tree_actions), 'tree rewards and path lengths do not match\n n tree rewards: {} \n n tree path: {}\ntree rewards: {}\n tree path: {}'.format(len(self.tree_rewards), len(self.tree_actions), self.tree_rewards, self.tree_actions)

        ## efficiently precompute discounted returns via backward pass
        discounted_returns = [0.0] * tree_len
        discounted_returns[-1] = self.tree_rewards[-1]
        for i in range(tree_len - 2, -1, -1):
            discounted_returns[i] = self.tree_rewards[i] + self.discount_factor * discounted_returns[i + 1]

        ## Loop through the tree path
        node = self.tree.root
        for depth, action in enumerate(self.tree_actions):

            ## Get the corresponding action leaf
            action_leaf = node.action_leaves[action]

            ## Discounted reward from the current node to the terminal node
            discounted_reward = discounted_returns[depth]

            ## update visit counts and performance estimates
            action_leaf.n_action_visits += 1
            node.n_state_visits += 1

            ## Incremental average update for performance
            action_leaf.performance += (
                (discounted_reward - action_leaf.performance) / action_leaf.n_action_visits
            )


            ## save per-node max and min Q values to normalise Qs in UCT calculation
            node.max_Q = max(node.max_Q, action_leaf.performance)
            node.min_Q = min(node.min_Q, action_leaf.performance)

            
            ## Move to the next node in the path if not at the end
            if depth < tree_len - 1:
                next_node_id = self.node_id_path[depth+1]
                node = action_leaf.children[next_node_id]

            
            ### some lists for debugging 

            ## debugging: save updates applied to the first node
            if depth == 0:
                to_append = [np.nan] * self.n_afc
                to_append[action] = discounted_reward
                self.first_node_updates.append(to_append)
                self.first_node_updates_by_depth[tree_len-1].append(to_append)
            
            ## save rewards of each step in the tree - i.e. the reward of making each move in the tree
            to_append = [np.nan] * self.n_afc
            to_append[action] = self.tree_rewards[depth]
            self.tree_reward_tracker[depth].append(to_append)

            ## updates, conditional on first action
            first_action = self.tree_actions[0]
            to_append = [np.nan] * self.n_afc
            to_append[first_action] = self.tree_rewards[depth]
            self.conditional_tree_reward_tracker[action][depth].append(to_append)




    ## calculate E-E value
    def compute_UCT(self, node, action_leaf, min_Q=None, max_Q=None): 
        assert action_leaf.n_action_visits > 0 or action_leaf.terminated, 'action leaf has not been visited: {}'.format(action_leaf)
        
        ## standard case
        # exploitation_term = action_leaf.performance
        # exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        
        ### or, min-max normalisation based on min and max for that node
        norm_term = max_Q - min_Q + 1e-8 ## add small constant to avoid divide by zero
        exploitation_term = (action_leaf.performance - min_Q) / norm_term
        exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values?
    def best_child(self, node):

        ## normalise by per-node Q
        min_Q, max_Q = node.min_Q, node.max_Q
        norm_term = max_Q - min_Q + 1e-8
        log_N = log(node.n_state_visits)
        c = self.exploration_constant

        ## or, scale c by recursive sum of discounted rewwards from current node to end of horizon -e.g. for horizon 3, self.env.N + self.env.N*discount + self.env.N*discount^2 + self.env.N*discount^3
        # c = self.exploration_constant * (self.env.N/2 * (1 - self.discount_factor**(self.horizon - node.trial + 1)) / (1 - self.discount_factor)) ## AFC
        # c = self.exploration_constant * (1/2 * (1 - self.discount_factor**(self.horizon - node.trial + 1)) / (1 - self.discount_factor)) ## bandit
        # norm_term = 1
        # log_N = log(node.n_state_visits) 
        # min_Q = 0

        ## or, no normalisation
        # c = self.exploration_constant
        # log_N = log(node.n_state_visits)
        # norm_term = 1
        # min_Q = 0

        ## or, normalise by the max reward of the env (which is known in these cases)
        # c = self.exploration_constant
        # norm_term = (1-self.discount_factor**(self.horizon-node.trial))/(1-self.discount_factor) ## bandit
        # # norm_term = self.env.N * norm_term ## AFC
        # min_Q = 0
        # log_N = log(node.n_state_visits)

        best_leaf = None
        best_uct = -float('inf')
        n_ties = 0
        
        # if node.trial==0:
        #     print()

        # ## debugging: force first action if trial==0
        # if node.trial==self.root_trial:
        #     return list(node.action_leaves.values())[0]

        for leaf in node.action_leaves.values():
            uct = (leaf.performance - min_Q) / norm_term + c * sqrt(log_N / leaf.n_action_visits)
            # if node.trial==0:
            #     print('exploitation: {}, exploration: {}, uct: {}'.format((leaf.performance - min_Q) / norm_term, c * sqrt(log_N / leaf.n_action_visits), uct))
            if uct > best_uct:
                best_uct = uct
                best_leaf = leaf
                n_ties = 1
            elif uct == best_uct:
                n_ties += 1
                if random.randrange(n_ties) == 0:
                    best_leaf = leaf

        return best_leaf


class MonteCarloTreeSearch_AFC(MonteCarloTreeSearch):

    def __init__(self, env, tree, exploration_constant=2, discount_factor=0.99, horizon=None):
        super().__init__(env, tree, exploration_constant, discount_factor, horizon)

    ## node_ids are defined by the informational state, i.e. the counts of low and high cost states in each cell
    def init_node_id(self, obs=None, parent_node_id=None):
        
        ## uses sparse representation: tuple of ((i, j), (low_count, high_count)) for observed cells only

        # Initialize counts from parent node_id (sparse tuple) if provided
        if parent_node_id is not None:
            counts = dict(parent_node_id)
        else:
            counts = {}
        
        # Add new observations - optimized loop
        high_cost = self.env.high_cost
        for i, j, c in obs:
            key = (int(i), int(j))
            prev = counts.get(key)
            if prev is None:
                counts[key] = (0, 1) if c == high_cost else (1, 0)
            elif c == high_cost:
                counts[key] = (prev[0], prev[1] + 1)
            else:
                counts[key] = (prev[0] + 1, prev[1])
        
        return tuple(sorted(counts.items()))
    
    ## in AFC, there is no meaningful state of the MDP, so belief state just contains the trial number
    def check_node(self, node):
        # assert np.array_equal(node.belief_state[:2*self.n_afc].reshape(self.n_afc,2), self.env.current), 'mismatch between node and env state\n node: {} \n env: {}'.format(node.belief_state[:2*self.n_afc].reshape(self.n_afc,2), self.env.current)
        # assert node.belief_state[0] == self.env.trial, 'mismatch between node and env trial\n node: {} \n env: {}'.format(node.belief_state[0], self.env.trial)
        assert node.trial == self.env.trial, 'mismatch between node and env trial\n node: {} \n env: {}'.format(node.trial, self.env.trial)


    ## when the external environment has changed, we need to update the MCTS object's version to reflect this
    def refresh_env(self, env=None):

        if env is not None:
            self.env = env.unwrapped

        ## i.e. the trial that the agent is current faced with in the real env
        self.root_trial = self.env.trial 

        ## set the horizon_trial - i.e. the trial at which search terminates
        self.horizon_trial = min(self.root_trial + self.horizon, self.env.n_trials-1)
        self.env.set_trunc_trial(self.horizon_trial)


    ## rollout policy (greedy or random)
    def rollout_policy(self):

        ### greedy:

        ## get the total reward of the paths
        t = self.env.trial
        path_rewards = []
        for action in range(self.n_afc):
            path = self.env.path_states[t][action]
            path_weight_idx = self.env.path_weights[t][action]
            weighted_rewards = [float(self.env.costs[x, y]) * self.env.sim_weight_map[path_weight_idx[k]] for k, (x, y) in enumerate(path)]
            path_rewards.append(sum(weighted_rewards))
        
        ## take greedy action
        best_ro_reward = max(path_rewards)
        ro_action = path_rewards.index(best_ro_reward)

        
        ### RANDOM: randomly choose between the paths
        # ro_action = random.choice(range(self.n_afc))

        return ro_action
    

# 2. Bandit-specific MCTS
class MonteCarloTreeSearch_Bandit(MonteCarloTreeSearch):
    """MCTS subclass with bandit-appropriate node IDs and rollout policy."""

    def __init__(self, env, tree, exploration_constant=2,
                 discount_factor=0.99, horizon=None):
        super().__init__(env, tree, exploration_constant,
                         discount_factor, horizon)

    
    ## node_id is the sufficient statistic for the belief state in the bandit: the counts of successes and failures for each arm
    ## stored as a flat tuple of length n_arms*2: (s0, f0, s1, f1, ...)
    def init_node_id(self, obs=None, parent_node_id=None):
        """
        Node ID = sufficient statistic for Beta-Bernoulli bandit:
        per-arm (n_successes, n_failures) counts, stored as a flat tuple.
        """
        if parent_node_id is not None:
            counts = list(parent_node_id)
        else:
            counts = [0] * (self.n_afc * 2)

        if len(obs) == 0:
            return tuple(counts)

        ## fast path for single observation tuple (common case during tree traversal)
        if isinstance(obs, tuple):
            arm = int(obs[0])
            if obs[1] == 1:
                counts[arm * 2] += 1
            else:
                counts[arm * 2 + 1] += 1
            return tuple(counts)

        ## numpy array path (used for initial obs from real env)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        for action, reward in obs:
            arm = int(action)
            if reward == 1:
                counts[arm * 2] += 1
            else:
                counts[arm * 2 + 1] += 1

        return tuple(counts)


    def refresh_env(self, env=None):
        if env is not None:
            self.env = env.unwrapped if hasattr(env, 'unwrapped') else env

        self.root_trial = self.env.trial
        self.horizon_trial = min(self.root_trial + self.horizon,
                                 self.env.n_trials - 1)
        self.env.set_trunc_trial(self.horizon_trial)

    def rollout_policy(self):

        ## random
        ro_action = int(random.random() * self.n_afc)

        ## greedy wrt/ current MDP?
        # ro_action = np.argmax(self.env.p_dist)

        ## learned Q values - e-greedy
        # Q = self.env.Q
        # best_a = np.argmax(Q)
        # if random.random() < 0.5:  # epsilon = 0.5
        #     ro_action = random.randrange(self.n_afc)
        # else:
        #     ro_action = best_a


        return ro_action

    def check_node(self, node):
        assert node.trial == self.env.trial, \
            f'mismatch: node trial={node.trial}, env trial={self.env.trial}'


# 3. Empowerment-specific MCTS
class MonteCarloTreeSearch_Emp(MonteCarloTreeSearch):
    """MCTS subclass with bandit-appropriate node IDs and rollout policy."""

    def __init__(self, env, tree, exploration_constant=2,
                 discount_factor=0.99, horizon=None):
        super().__init__(env, tree, exploration_constant,
                         discount_factor, horizon)

    
    ## node_id is the sufficient statistic for the multinomial distirbution in the empowerment problem: the counts of each outcome for each arm
    def init_node_id(self, obs=None, parent_node_id=None):
        """
        Node ID = sufficient statistic for multinomial bandit:
        per-arm counts for each outcome, stored as a flat tuple of length
        n_afc * n_outcomes: (arm0_o0, arm0_o1, ..., arm1_o0, ...).
        """
        n_outcomes = self.env.n_outcomes

        if parent_node_id is not None:
            counts = list(parent_node_id)
        else:
            counts = [0] * (self.n_afc * n_outcomes)

        if len(obs) == 0:
            return tuple(counts)

        ## fast path for single observation tuple (common during tree traversal)
        if isinstance(obs, tuple):
            arm = int(obs[0])
            outcome = int(obs[1])
            counts[arm * n_outcomes + outcome] += 1
            return tuple(counts)

        ## numpy array path (used for initial obs from real env)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        for action, outcome in obs:
            arm = int(action)
            counts[arm * n_outcomes + int(outcome)] += 1

        return tuple(counts)


    def refresh_env(self, env=None):
        if env is not None:
            self.env = env.unwrapped if hasattr(env, 'unwrapped') else env

        self.root_trial = self.env.trial
        self.horizon_trial = min(self.root_trial + self.horizon,
                                 self.env.n_trials - 1)
        self.env.set_trunc_trial(self.horizon_trial)

    def rollout_policy(self):

        ## random
        ro_action = int(random.random() * self.n_afc)

        return ro_action

    def check_node(self, node):
        assert node.trial == self.env.trial, \
            f'mismatch: node trial={node.trial}, env trial={self.env.trial}'




### MCTS components
class Node:

    def __init__(self, state, node_id, reward, terminated, trial,
                    n_afc=2,
                 ):

        ## state info
        self.state = state
        self.n_state_visits = 0
        self.reward = reward
        self.trial = trial
        self.terminated = terminated
        self.node_id = node_id

        ## save the max and min Q values observed among the children of this action node, for normalization purposes in the UCB formula
        ## (pure-Python inf avoids cpyext numpy overhead under PyPy; numerically identical to np.inf)
        self.max_Q = float('-inf')
        self.min_Q = float('inf')

        ## define valid actions
        self.untried_actions = list(range(n_afc))

        ## action leaves
        self.action_leaves = {a: None for a in self.untried_actions}


    def __str__(self):
        action_leaves_msg = {action: np.round(leaf.performance,3) if leaf is not None else None for action, leaf in self.action_leaves.items()}
        return "state {}: (trial={}, visits={}, terminated={})\n{})".format(
                                                    self.state,
                                                    self.trial,
                                                  self.n_state_visits,
                                                  self.terminated,
                                                  action_leaves_msg
                                                  )

    ## select a random untried action
    def untried_action(self):
        idx = random.randint(0, len(self.untried_actions) - 1)

        # Swap with last element and pop for removal
        self.untried_actions[idx], self.untried_actions[-1] = self.untried_actions[-1], self.untried_actions[idx]
        return self.untried_actions.pop()


class Action_Node:

    def __init__(self, action, trial, parent_id):
        self.action = action
        self.performance = None
        self.n_action_visits = 0
        self.trial = trial
        self.parent_id = parent_id
        self.children={}

    def __str__(self):
        return "(action={}, n_children={}, visits={}, performance={:0.3f})".format(
                                                  self.action,
                                                  len(self.children.keys()),
                                                  self.n_action_visits,
                                                  self.performance,
                                                  )


class Tree:

    def __init__(self):
        self.root = None

    ## check if node is expandable
    def is_expandable(self, node):
        return node.untried_actions and not node.terminated

    ## attach action leaf to child state
    def add_state_node(self, state, node_id, reward, terminated, trial, n_afc=2, parent=None,
                       ):

        ## create a new node
        node = Node(state=state, node_id=node_id, reward=reward, terminated=terminated, trial = trial,
                    n_afc=n_afc
                    )


        ### store parent-child relationships

        ## if no action led to this node, then this is the root
        if parent is None:
            self.root = node
        else:

            ## add this state node to the children of the previous action leaf
            parent.children[node.node_id] = node

        return node

    def get_children(self, node, dummy=False):
        children = []
        for a, leaf in node.action_leaves.items():
            if leaf is not None:
                for child_key in leaf.children.keys():
                    child = leaf.children[child_key]
                    children.append(tuple((a, leaf, child_key, child)))

                ## if there are no children (i.e. the S-A leaf has been made, but doesn't have any S nodes), add a dummy child
                if dummy:
                    if len(leaf.children) == 0:
                        children.append(tuple((a, leaf, None, None)))
        return children


    def print_tree(self, node, indent="", is_last=True, dummy=False, depth=0, max_depth=None):
        """
        Recursively print the tree structure with markers, visit counts, and values.
        """
        # Stop printing if max depth is reached
        if max_depth is not None and depth > max_depth:
            return

        # Get the current node
        if dummy:
            if node is None:
                return
            else:
                node_label = f"{node.reward}"
        else:
            node_label = f"{node.reward}"
        trial_label = f"{node.trial}"

        # Add branch marker
        branch = "└── " if is_last else "├── "
        print(f"{indent}{branch}Node: {node_label}, Trial: {trial_label}, Visits: {node.n_state_visits}")

        # Update indentation for children
        child_indent = indent + ("    " if is_last else "│   ")

        # Group children by action
        children_by_action = {}
        for action, leaf, child_id, child_node in self.get_children(node, dummy):
            if action not in children_by_action:
                children_by_action[action] = []
            children_by_action[action].append((leaf, child_id, child_node))

        # Find the best action based on performance
        best_action = max(
            children_by_action.items(),
            key=lambda item: item[1][0][0].performance,
            default=(None, [])
        )[0]

        # Iterate through actions and their corresponding children
        num_actions = len(children_by_action)
        for i, (action, children) in enumerate(children_by_action.items()):
            is_action_last = i == num_actions - 1

            leaf = children[0][0]
            action_label = f"Action {action}, (n_v: {leaf.n_action_visits},  branch factor: {len(children)}, perf: {leaf.performance:.2f})"

            if action == best_action:
                action_label = f"\033[1m{action_label}\033[0m"

            action_branch = "└── " if is_action_last else "├── "
            print(f"{child_indent}{action_branch}{action_label}")

            sub_child_indent = child_indent + ("    " if is_action_last else "│   ")

            for j, (leaf, child_id, child_node) in enumerate(children):
                is_child_last = j == len(children) - 1

                self.print_tree(
                    child_node,
                    indent=sub_child_indent,
                    is_last=is_child_last,
                    dummy=dummy,
                    depth=depth + 1,
                    max_depth=max_depth
                )


    def max_depth(self, node):
        """Recursively calculate the maximum depth of the tree starting from the given node."""
        if not self.get_children(node):
            return 1

        child_depths = []
        for _, _, _, child_node in self.get_children(node):
            child_depths.append(self.max_depth(child_node))

        return 1 + max(child_depths)


    ## function for getting the max and min Q-values at a given depth of the tree
    def min_max_Q(self, node, depth, current_depth=0):
        """Recursively calculate the maximum and minimum Q-values at a given depth of the tree."""
        if current_depth == depth:
            Qs = []
            for a in node.action_leaves.keys():
                if node.action_leaves[a] is not None:
                    Qs.append(node.action_leaves[a].performance)
            if len(Qs) == 0:
                return np.inf, -np.inf
            return min(Qs), max(Qs)

        max_Q = -np.inf
        min_Q = np.inf
        for _, _, _, child_node in self.get_children(node):
            child_min_Q, child_max_Q = self.min_max_Q(child_node, depth, current_depth + 1)
            max_Q = max(max_Q, child_max_Q)
            min_Q = min(min_Q, child_min_Q)

        return min_Q, max_Q


    ## prune, i.e. after taking a step, keep only that subtree
    def prune(self, action, next_node_id):

        ## delete actions not taken
        actions_to_delete = [a for a in self.root.action_leaves.keys() if (a != action) and (self.root.action_leaves[a] is not None)]
        for a in actions_to_delete:
            del self.root.action_leaves[a]

        ## delete subtree for the other state children reachable from the root-action pair
        self.root.action_leaves[action].children = {next_node_id: self.root.action_leaves[action].children[next_node_id]}

        ## update the root
        self.root = self.root.action_leaves[action].children[next_node_id]
