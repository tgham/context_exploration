import random
from math import sqrt, log
from utils import Node, Tree
import numpy as np

class MonteCarloTreeSearch():

    def __init__(self, env, tree):
        self.env = env
        self.tree = tree
        self.action_space = self.env.action_space.n
        observation, info = self.env.reset()
        state = observation['agent']
        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False))

    def expand(self, node):
        action = node.untried_action()
        observation, reward, terminated, _, _ = self.env.step(action)
        state=observation['agent'] ## this may not be right
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=terminated) 
        self.tree.add_node(new_node, node)
        return new_node

    def rollout_policy(self, node):
        if node.terminal:
            return node.reward

        ## surely this needs to stop at some point, rather than rolling out until u get a terminal state?
        while True:
            action = random.randint(0, self.action_space-1)
            observation, reward, terminated, _, _ = self.env.step(action)
            if terminated:
                return reward
            

    ## calculate E-E value
    def compute_UCT(self, parent, child, exploration_constant): # could turn this exploration constant into a param defined at init
        exploitation_term = child.total_simulation_reward / child.num_visits
        exploration_term = exploration_constant * sqrt(2 * log(parent.num_visits) / child.num_visits)
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values?
    def best_child(self, node, exploration_constant):
        best_child = self.tree.children(node)[0]
        best_value = self.compute_UCT(node, best_child, exploration_constant)
        iter_children = iter(self.tree.children(node))
        next(iter_children)
        for child in iter_children:
            value = self.compute_UCT(node, child, exploration_constant)
            if value > best_value:
                best_child = child
                best_value = value
        return best_child

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self):
        node = self.tree.root
        while not node.terminal:
            if self.tree.is_expandable(node):
                return self.expand(node)
            else:
                node = self.best_child(node, exploration_constant=1.0/sqrt(2.0))
                observation, reward, terminated, _, _ = self.env.step(node.action)
                state = observation['agent']
                # assert node.state == state
                # print(state, node.state)
                if not np.array_equal(node.state, state):
                    # print(node.state, node.action, state)
                    print(node.state, node.action, self.env.step(node.action))

                assert np.array_equal(node.state, state)
        return node

    
    ## backup rewards until you reach the root
    def backward(self, node, value):
        while node:
            node.num_visits += 1
            node.total_simulation_reward += value
            node.performance = node.total_simulation_reward/node.num_visits
            node = self.tree.parent(node)

    def forward(self):
        self._forward(self.tree.root)

    def _forward(self,node):
        best_child = self.best_child(node, exploration_constant=0)

        print("****** {} ******".format(best_child.state))

        for child in self.tree.children(best_child):
            print("{}: {:0.4f}".format(child.state, child.performance))

        if len(self.tree.children(best_child)) > 0:
            self._forward(best_child)
