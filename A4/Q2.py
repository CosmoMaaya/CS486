# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

# This script uses multithreading. 
# Matplotlib sometimes will raise errors "main thread is not in main loop"
# This should not prevent the script from running. But the following line will disable the graphing GUI to aviud the error.
# If you want to see the graphing GUI, comment out the next line
matplotlib.use('Agg')
# %%
# Four moving directions, used for numpy array arithmetic
UP = np.array([-1, 0])
DOWN = np.array([1, 0])
LEFT = np.array([0, -1])
RIGHT = np.array([0, 1])

# Add them to array, and match the unicode for printing
MOVING_ACTIONS = np.array([LEFT, UP, RIGHT, DOWN])
ACTION_PRINT = ['\u2190', '\u2191', '\u2192', '\u2193']

GRID_EMPTY = 0
GRID_WALL = 1
GRID_PRIZE = 8

# wall positions for different grid types
FOUR_ROOM = ([0, 2, 2, 2, 2, 4], [2, 0, 1, 3, 4, 2])
# ((0, 2), (2, 0), (2, 1), (2, 3), (2, 4), (4, 2))
MAZE = ([1, 1, 1, 2, 3, 4, 4, 4, 4], [1, 2, 3, 3, 1, 1, 2, 3, 4,])
# ((1, 1), (1, 2), (1, 3), (2, 3), (3, 1), (4, 1), (4, 2), (4, 3), (4, 4))
EMPTY = None

class Grid():
    def __init__(self, maze_type):
        self.grid = np.zeros((5,5))
        self.position = np.array([2,2])
        # set up walls if not empty
        if maze_type != EMPTY:
            self.grid[maze_type] = GRID_WALL
        # set up prize
        self.prize_location = self.__random_init_prize()    # Tuple

    def __random_init_prize(self):
        # all possible prize locations (not walls and starting position)
        prize_location_choices = [(i, j) for i in range(5) for j in range(5) if self.grid[i, j] == GRID_EMPTY and (i, j) != (2, 2)]
        prize_idx = np.random.randint(len(prize_location_choices))
        self.grid[prize_location_choices[prize_idx]] = GRID_PRIZE
        return tuple(prize_location_choices[prize_idx])

    def move(self, direction):
        self.position += direction
        # if out of bounds or hit a wall
        if (self.position >= 5).any() or (self.position < 0).any() or self.grid[tuple(self.position)] == GRID_WALL:
            # revert moving, no award
            self.position -= direction
            return 0, self.position

        # if hit the prize
        if self.grid[tuple(self.position)] == GRID_PRIZE:
            return 1, self.position
        
        # if not hit anything
        return 0, self.position
    
    def print_grid(self):
        for i in range(5):
            for j in range(5):
                if self.grid[i, j] == GRID_WALL:
                    print('#', end=' ')
                elif self.grid[i, j] == GRID_PRIZE:
                    print('P', end=' ')
                elif i == self.position[0] and j == self.position[1]:
                    print('*', end=' ')
                else:
                    print('O', end=' ')
            print()
    
    def restart(self):
        # set back to start position and regeneralize prize
        self.position = np.array([2,2])
        self.grid[self.prize_location] = GRID_EMPTY
        self.prize_location = self.__random_init_prize()

# %%
DISCOUNT = 0.95
TERMINATION_PROB = 0.05

class Receiver():
    def __init__(self, grid, num_symbols):
        self.grid = grid
        self.Q = np.zeros((5, 5, num_symbols, len(MOVING_ACTIONS)))
        self.current_state = (2, 2, -1)
        self.num_symbols = num_symbols
        
    def start_learning(self, message, learning_rate, eps):
        # When start learning, the current state is the starting position of the grid. We let Sender to reset the grid.
        self.current_state = (self.grid.position[0], self.grid.position[1], message)
        while True:
            action_idx = self.select_action(eps)
            reward, new_position = self.grid.move(MOVING_ACTIONS[action_idx])
            new_state = (new_position[0], new_position[1], message)
            if reward == 1:
                # No future reward in this case as we terminate the episode
                self.Q[self.current_state][action_idx] = (1 - learning_rate) * self.Q[self.current_state][action_idx] + learning_rate * reward
            else:
                self.Q[self.current_state][action_idx] = (1 - learning_rate) * self.Q[self.current_state][action_idx] + learning_rate * DISCOUNT * np.max(self.Q[new_state])
            self.current_state = new_state
            # termination condition
            if np.random.uniform(0, 1) < TERMINATION_PROB or reward == 1:
                return reward
    
    def test(self, message):
        self.current_state = (self.grid.position[0], self.grid.position[1], message)
        count = 0
        while True:
            # we only exploit the best action
            action_idx = self.select_action(0)
            reward, new_position = self.grid.move(MOVING_ACTIONS[action_idx])
            self.current_state = (new_position[0], new_position[1], message)
            count += 1
            if np.random.uniform(0, 1) < TERMINATION_PROB or reward == 1:
                return reward, np.power(DISCOUNT, count) * reward

    def select_action(self, eps):
        # Eps: probability to explore
        if np.random.uniform(0, 1) < eps:
            return np.random.randint(len(MOVING_ACTIONS))
        else:
            # random tie breaking. useful for all zero position (in initialization)
            return np.random.choice(np.flatnonzero(self.Q[self.current_state] == self.Q[self.current_state].max()))
    
    def print_action(self):
        for msg in range(self.num_symbols):
            print(f"Actions for message {msg}")
            print("\ta\tb\tc\td\te")
            q = self.Q[:,:,msg,:]
            for i, row in enumerate(q):
                print(f"{i} \t", end='')
                for j, col in enumerate(row):
                    assert col.shape == (4,)
                    if self.grid.grid[i, j] == GRID_WALL:
                        print(" \t", end='')
                    else:
                        idx = np.argmax(col)
                        print(f"{ACTION_PRINT[idx]} \t", end='')
                print()

class Sender():
    def __init__(self, grid, num_symbols):
        self.grid = grid
        self.Q = np.zeros((5, 5, num_symbols))
        self.num_symbols = num_symbols
        self.receiver = Receiver(grid, num_symbols)

    def start_learning(self, eps, N_episodes):
        learning_rate = 0.9
        for i in range(N_episodes):
            # reset the grid for each episode
            self.grid.restart()
            self.current_state = self.grid.prize_location
            action_message = self.select_action(eps)
            # Inner loop for receiver
            reward = self.receiver.start_learning(action_message, learning_rate, eps)
            # No future reward and state change because it does not make sense. The state of sender is random (prize location), nothing to do with its action. Absorbing state
            self.Q[self.current_state][action_message] = (1 - learning_rate) * self.Q[self.current_state][action_message] + learning_rate * reward
            learning_rate -= (0.9 - 0.01) / N_episodes
    
    def test(self, N_episodes):
        received_reward = 0
        for i in range(N_episodes):
            self.grid.restart()
            self.current_state = self.grid.prize_location
            action_message = self.select_action(0)
            # Inner loop for receiver, get the dicounted reward
            reward, receiver_discounted_reward = self.receiver.test(action_message)
            received_reward += receiver_discounted_reward
            
        return received_reward / N_episodes

    def select_action(self, eps):
        # Eps: probability to explore
        if np.random.uniform(0, 1) < eps:
            return np.random.randint(self.num_symbols)
        else:
            # random tie breaking. Useful for eps small
            return np.random.choice(np.flatnonzero(self.Q[self.current_state] == self.Q[self.current_state].max()))

# %%

# Helper functions for multithreading. Speed up the learning process
from multiprocessing import Pool
NEPs = [10, 100, 1000, 10000, 50000, 100000]
EPSs = [0.01, 0.1, 0.4]

# For question C
def c_one_test(eps):
    rewards = np.zeros(len(NEPs))
    senders = [None for _ in NEPs]
    for n_idx, n_episodes in enumerate(NEPs):
        g = Grid(FOUR_ROOM)
        sender = Sender(g, 4)
        sender.start_learning(eps, n_episodes)
        rewards[n_idx] = sender.test(1000)
        senders[n_idx] = sender
    return rewards, senders

# For question D
NUM_SYMBOLS_D = [2, 4, 10]
def d_one_test(num_symbols):
    rewards = np.zeros(len(NEPs))
    senders = [None for _ in NEPs]
    for n_idx, n_episodes in enumerate(NEPs):
        g = Grid(FOUR_ROOM)
        sender = Sender(g, num_symbols)
        sender.start_learning(0.1, n_episodes)
        rewards[n_idx] = sender.test(1000)
        senders[n_idx] = sender
    return rewards, senders

# For question E
NUM_SYMBOLS_E = [2, 3, 5]
def e_one_test(num_symbols):
    rewards = np.zeros(len(NEPs))
    senders = [None for _ in NEPs]
    for n_idx, n_episodes in enumerate(NEPs):
        g = Grid(MAZE)
        sender = Sender(g, num_symbols)
        sender.start_learning(0.1, n_episodes)
        rewards[n_idx] = sender.test(1000)
        senders[n_idx] = sender
    return rewards, senders

# For question F
def f_one_test(grid_type):
    rewards = np.zeros(len(NEPs))
    senders = [None for _ in NEPs]
    for n_idx, n_episodes in enumerate(NEPs):
        g = Grid(EMPTY)
        sender = Sender(g, 1)
        sender.start_learning(0.1, n_episodes)
        rewards[n_idx] = sender.test(1000)
        senders[n_idx] = sender
    return rewards, senders

# %%

if __name__ == "__main__":
    # Run the training for three epsilon values, 10 tests each simultaneously
    results_c = [None for _ in range(10 * len(EPSs))]
    parameters_c = [EPSs[0]] * 10 + [EPSs[1]] * 10 + [EPSs[2]] * 10
    with Pool() as pool:
        results_c = pool.map(c_one_test, parameters_c)
    rewards_c = np.array([np.array([results_c[i][0], results_c[i+10][0], results_c[i+20][0]]) for i in range(10)])
    senders_c = np.array([np.array([results_c[i][1], results_c[i+10][1], results_c[i+20][1]]) for i in range(10)])
    y_c = np.mean(rewards_c, axis=0)
    err_c = np.std(rewards_c, axis=0)
    for i, eps in enumerate(EPSs):
        plt.errorbar(np.log10(NEPs), y_c[i], yerr=err_c[i], fmt='o', capsize=5, label=f'Epsilon = {eps}', linestyle='--')
    plt.ylabel("Average discounted reward")
    plt.xlabel("log N_ep")
    plt.title(f"Receiver's Average Discounted Reward for Different Epsilon")
    plt.legend()
    plt.savefig(f"Receiver's Average Discounted Reward for Different Epsilon.png")
    plt.show()
    plt.close()
    
    # Show example of the sender's Q-table and receiver's policy
    print(np.argmax(senders_c[0][1][-1].Q, axis=2))

    senders_c[0][1][-1].receiver.print_action()

# %%

    # d)

    # Run the training for three N_ep values, 10 tests each simultaneously
    results_d = [None for _ in range(10 * len(NUM_SYMBOLS_D))]
    parameters_d = [NUM_SYMBOLS_D[0]] * 10 + [NUM_SYMBOLS_D[1]] * 10 + [NUM_SYMBOLS_D[2]] * 10
    with Pool() as pool:
        results_d = pool.map(d_one_test, parameters_d)

    rewards_d = np.array([np.array([results_d[i][0], results_d[i+10][0], results_d[i+20][0]]) for i in range(10)])
    senders_d = np.array([np.array([results_d[i][1], results_d[i+10][1], results_d[i+20][1]]) for i in range(10)])
    y_d = np.mean(rewards_d, axis=0)
    err_d = np.std(rewards_d, axis=0)

    for i in range(len(NUM_SYMBOLS_D)):
        plt.errorbar(np.log10(NEPs), y_d[i], yerr=err_d[i], fmt='o', capsize=5, label=f"{NUM_SYMBOLS_D[i]} symbols", linestyle='--')
    plt.ylabel("Average discounted reward")
    plt.xlabel("log N_ep")
    plt.title(f"Receiver's Average Discounted Reward for Different N in Four Room")
    plt.legend()
    plt.savefig(f"Receiver's Average Discounted Reward for Different N in Four Room.png")
    plt.show()
    plt.close()

# %%
    # e)

    # Run the training for three N_ep values, 10 tests each simultaneously
    results_e = [None for _ in range(10 * len(NUM_SYMBOLS_E))]
    parameters_e = [NUM_SYMBOLS_E[0]] * 10 + [NUM_SYMBOLS_E[1]] * 10 + [NUM_SYMBOLS_E[2]] * 10
    with Pool() as pool:
        results_e = pool.map(e_one_test, parameters_e)

    rewards_e = np.array([np.array([results_e[i][0], results_e[i+10][0], results_e[i+20][0]]) for i in range(10)])
    senders_e = np.array([np.array([results_e[i][1], results_e[i+10][1], results_e[i+20][1]]) for i in range(10)])
    y_e = np.mean(rewards_e, axis=0)
    err_e = np.std(rewards_e, axis=0)

    for i in range(len(NUM_SYMBOLS_E)):
        plt.errorbar(np.log10(NEPs), y_e[i], yerr=err_e[i], fmt='o', capsize=5, label=f"{NUM_SYMBOLS_E[i]} symbols", linestyle='--')
    plt.ylabel("Average discounted reward")
    plt.xlabel("log N_ep")
    plt.title(f"Receiver's Average Discounted Reward for Different N in Maze")
    plt.legend()
    plt.savefig(f"Receiver's Average Discounted Reward for Different N in Maze.png")
    plt.show()
    plt.close()

# %%
    # f)
    results_f = [None for _ in range(10)]
    with Pool() as pool:
        results_f = pool.map(f_one_test, [EMPTY for _ in range(10)])

    rewards_f = np.array([r[0] for r in results_f])
    senders_f = np.array([r[1] for r in results_f])
    y_f = np.mean(rewards_f, axis=0)
    err_f = np.std(rewards_f, axis=0)

    plt.errorbar(np.log10(NEPs), y_f, yerr=err_f, fmt='o', capsize=5, linestyle='--')
    plt.ylabel("Average discounted reward")
    plt.xlabel("log N_ep")
    plt.title(f"Receiver's Average Discounted Reward for Empty Grid")
    plt.savefig(f"Receiver's Average Discounted Reward for Empty Grid.png")
    plt.show()
    plt.close()