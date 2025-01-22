import numpy as np
from numba import jit, njit
from utils import argm

## value iteration
@njit
def value_iteration(dp_costs, goal, max_iters=1000, theta=0.0001, discount=0.99):
    N = len(dp_costs)
    n_actions = 4

    # Action directions
    action_directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    # Initialize tables
    V = np.zeros((N, N))
    A = np.zeros((N, N), dtype=np.int32)
    Q = np.full((N, N, n_actions), np.nan)

    # Set cost of the goal to 0
    goal_x, goal_y = goal

    for i in range(max_iters):
        delta = 0

        for x in range(N):
            for y in range(N):
                if x == goal_x and y == goal_y:
                    continue

                v = V[x, y]

                # Compute Q-values for all valid actions in one loop
                for a in range(n_actions):
                    next_x = x + action_directions[a][0]
                    next_y = y + action_directions[a][1]

                    if 0 <= next_x < N and 0 <= next_y < N:
                        Q[x, y, a] = dp_costs[next_x, next_y] + discount * V[next_x, next_y]
                    else:
                        Q[x, y, a] = np.nan

                # Update value and action tables
                max_q = np.nanmax(Q[x, y])
                V[x, y] = max_q
                best_actions = np.where(Q[x, y] == max_q)[0]
                A[x, y] = np.random.choice(best_actions)

                # Update convergence threshold
                delta = max(delta, abs(v - max_q))

        if delta < theta:
            break

    return V, Q, A


