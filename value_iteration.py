import numpy as np
from numba import jit, njit
from utils import argm

## value iteration
@njit
def value_iteration(dp_costs, goal, max_iters = 1000, theta = 0.0001, discount = 0.99):

    N = len(dp_costs)
    n_actions = 4

    ## init actions
    action_directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    
    ## init tables
    V = np.zeros((N, N))
    A = np.zeros((N, N))
    Q = np.zeros((N, N, n_actions))

    ## determine whether to use true costs or inferred costs
    # if self.known_costs:
    #     dp_costs = self.costs.copy()
    # else:
    #     dp_costs = self.posterior_mean.reshape(self.N, self.N).copy()

    ## set cost of goal to 0
    # dp_costs[goal[0], goal[1]] = 0

    # assert np.all(dp_costs <= 0), 'costs are not all negative: {}'.format(dp_costs)

    ## loop through states
    for i in range(max_iters):
        delta = 0
        for x in range(N):
            for y in range(N):
                
                ## skip if at goal state to ensure that v(goal) = 0
                if x == goal[0] and y == goal[1]:
                    # V[x, y] = 0
                    continue

                v = V[x, y]
                # q = np.zeros(n_actions)
                q = np.full(n_actions, np.nan)

                ## loop through actions and get the discounted value of each of the next states
                for a in range(n_actions):

                    ## allow wall moves
                    # next_state = np.clip([x, y] + self._action_to_direction[a], 0, self.N-1)
                    # q[a] = dp_costs[next_state[0], next_state[1]] + discount*V[next_state[0], next_state[1]]

                    ## or, don't allow wall moves
                    next_x = x + action_directions[a][0]
                    next_y = y + action_directions[a][1]
                    if 0 <= next_x < N and 0 <= next_y < N:
                        q[a] = dp_costs[next_x, next_y] + discount * V[next_x, next_y]

                ## update the Q-table
                Q[x, y] = q

                ## use the best action to update the value of the current state
                V[x, y] = np.nanmax(q)

                idx = np.where(q == np.nanmax(q))[0]
                A[x, y] = np.random.choice(idx)

                ## check if converged
                delta = max(delta, np.abs(v - V[x, y]))
        
        if delta < theta:
            # print('DP converged after {} iterations'.format(i))
            break

        if i == max_iters-1:
            print('DP did not converge after '+str(i)+ ' iterations')

    ## need to check if this has lead to a valid policy

    return V, Q, A

