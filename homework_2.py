# %%
# %load_ext autoreload
# %autoreload 2

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iml.dungeon import Dungeon
from iml.agent import DungeonAgent
from iml.visualization import DungeonVisualization
from iml.dungeon_maker import make_treasure_room, make_underground_maze, make_underground_maze_with_meat


# %% [markdown]
# **Group members:** [your names]

# %% [markdown]
# ## Part 1: Gridworld Environment

# %% [markdown]
# ### 1.1 A drunken knight in a treasure room
# ### (a)
# > Implement a stochastic policy where you select actions uniformly at random.

# %%
def my_reward_11(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    return 0.0

def my_action_11(state, q_value):
	# <implement your action here>
	# return ... (int)

def my_Q_update_11(agent, state, action, reward, next_state, dungeon):
    return None # use None for now

treasure_room = make_treasure_room()
vis_11 = DungeonVisualization(treasure_room)

drunken_knight = DungeonAgent(n_states=treasure_room.n_states, 
                             choose_action=my_action_11,
                             reward_func=my_reward_11,
                             update_q_values=my_Q_update_11)

# Run the agent for 1 episode & collect states, actions, rewards
states, actions, rewards = drunken_knight.run_episode(treasure_room, nstep=10, gamma=1.0)

# Show the map, Q-values, and path.
vis_11.show(states, actions, drunken_knight.q_values)

# %% [markdown]
# ### (b)
# > Observe how you move and draw a Markov decision process (MDP) diagram for all four non-wall states.
#
# *(Insert your MDP diagram photo/figure here)*

# %% [markdown]
# ### (c)
# > Compute the theoretical probability distribution over a 2-step horizon analytically (without writing code) when you use the stochastic policy you implemented.
#

# %% [markdown]
# ### (d)
# > If you implement Q-learning with an episode length of 1 (`nstep = 1`) under current setup, will you manage to perform better? Why or why not? Explain your reasoning.
#

# %% [markdown]
# ### (e)
# > For a dungeon of width $N$ and height $M$, what is the minimum number of steps required to reach the goal from the starting position? Why? 
#

# %% [markdown]
# ---
# ### 1.2 A tricky knight in an underground maze
#
# ### (a)
# > (a) Using your stochastic policy $\pi$, implement the Q-learning update with a discount factor ($\gamma = 1$) and a maximum episode length of 30 steps ($n_{\text{step}} = 30$). Run consecutive episodes without resetting the Q-table. Explain your observations on how the Q-values change relative to the goal and starting locations.

# %%
def my_reward_12(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    return 0.0

def my_action_12(state, q_values):
    return np.random.choice(4) 

def my_Q_update_12(agent, state, action, reward, next_state, dungeon):
	# <implement your Q update here>
	# Hint: lecture 9, page 18 <20/29>
	# return ... (float)

underground_maze = make_underground_maze(p_slippery=0.0)
vis_12 = DungeonVisualization(underground_maze)

tricky_knight = DungeonAgent(n_states=underground_maze.n_states, 
                             choose_action=my_action_12,
                             reward_func=my_reward_12,
                             update_q_values=my_Q_update_12)

training_reward = []
# program your knight here
# compute episodic reward
# append "training_reward"

# plot "training_reward"
vis_12.plot_training_reward(training_reward)


# %% [markdown]
# ### (b)
# > By varying the discount factor $\gamma$, do you observe any differences? Summarize how the discount rate affects the results and explain why it is necessary.
#

# %% [markdown]
# ### (c)
# > Implement an $\epsilon$-greedy policy for your agent by varying $\epsilon$. Plot the sum of discounted rewards per episode over 100 training runs and explain your observations.

# %% [markdown]
# ### (d)
# > Is a single training run sufficient to evaluate and compare an agent’s performance across different hyperparameter settings? Specifically: (i) Will an agent reach the goal on every evaluation episode? Explain why or why not. (ii) Under what specific conditions would RL training and evaluation results be strictly deterministic and reproducible? And (iii) Given these factors, how should an RL agent's performance be properly evaluated and compared, both when those deterministic conditions are met and when they are not?

# %% [markdown]
# ### (e)
# > Recreate the maze using `make_underground_maze_with_meat()`, update the reward function, set $\epsilon = 0.9$, $n_{\text{step}} = 50$, and $\gamma = 0.95$, and run the agent for 100 to 200 training runs. Explain your observations regarding your strategy (Q-values) and behavior.

# %%
def my_reward_12(state, action, next_state, dungeon):
    if next_state == dungeon.end:
        return 1.0
    if dungeon.tiles[next_state] == "+":
        return 0.5
    return 0.0

def my_action_12(state, q_values):
	# <implement your action here>
	# return ... (int)

def my_Q_update_12(agent, state, action, reward, next_state, dungeon):
	# <implement your Q update here>
	# Hint: lecture 9, page 18 <20/29>
	# return ... (float)

underground_maze = make_underground_maze_with_meat(p_slippery=0.0)
vis_12 = DungeonVisualization(underground_maze)

tricky_knight = DungeonAgent(n_states=underground_maze.n_states, 
                             choose_action=my_action_12,
                             reward_func=my_reward_12,
                             update_q_values=my_Q_update_12)

training_reward = []
# program your knight here
# compute episodic reward
# append "training_reward"

# plot "training_reward"
vis_12.plot_training_reward(training_reward)

# %% [markdown]
# > (f) Compare the state values $V(s)$ when you go directly to the treasure (e), collects free meat as you observed (+), or do both. Explain your results mathematically.
#

# %% [markdown]
# > (g) How might you modify the reward function or environment design to prevent this *reward hacking*? Propose and explain a solution. (Bonus)


# %% [markdown]
# ## Part 2: Bayesian Decision-Making in Multi-Armed Bandits

# %% [markdown]
# ### 2.1 Bayesian Estimation of Arm Rewards

# %% [markdown]
# ### (a)
# > Starting from Bayes' theorem and the Bernoulli likelihood, derive the posterior distribution $p(\theta \mid D)$ and identify its updated parameters. Hence, show that the Beta distribution is a conjugate prior to the Bernoulli likelihood. 

# %% [markdown]
# ### (b)
# > Suppose the prior is $(\beta(2,2))$. After ten pulls, the arm produces seven successes and three failures. Compute the posterior distribution and posterior mean. Show that, in general, the posterior mean can be expressed as a weighted average of the prior mean and the empirical success rate. Interpret the weights and explain how they change as the number of observations increases

# %% [markdown]
# ### 2.2 Learning with Thompson Samplin

# %% [markdown]
# ### (a)
# > Implement a five-armed Bernoulli bandit environment that generates binary rewards using the given success probabilities. Then implement a Thompson sampling agent that maintains the posterior parameters for each arm, selects actions using posterior sampling, and updates the corresponding posterior after receiving each reward. 

# %% [markdown]
# ### (b)
# > Using the results from part (a), plot the evolution of the posterior mean estimates for all five arms as a function of time; visualize the posterior distributions; discuss how the posterior means and distributions evolve; and explain why some arms become more certain about their estimated success probabilities than others.

# %% [markdown]
# ### 2.3 Experimental Evaluation and Exploration

# %% [markdown]
# ### (a)
# > Perform 100 independent trials of Thompson sampling, each consisting of 2,000 decisions. Initialize a new agent with the same priors at the beginning of every trial.

# %% [markdown]
# ### (b)
# > Derive the agent's expected reward per decision and its expected instantaneous pseudo-regret. Hence, derive the expected cumulative pseudo-regret $\mathbb{E}[\mathcal{R}_T]$. How does it grow with $T$, assuming $\epsilon$ remains fixed and the greedy choice remains optimal? (Bonus)

# %% [markdown]
# ### (c)
# > Compare $\epsilon$-greedy exploration with Thompson sampling. Explain why Thompson sampling may select an arm with a lower posterior mean and how its exploration behavior changes as more observations accumulate.

# %%
