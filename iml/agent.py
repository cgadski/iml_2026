import os
import numpy as np

class DungeonAgent:
	def __init__(
		self, 
		n_states, 
		choose_action=None, 
		update_q_values=None,
		update_sarsa_values=None,
		reward_func=None
	):
		self.q_values = np.zeros((4, n_states))
		self.visited = np.zeros(n_states, dtype=np.int32)
		
		# External function holders
		self._external_choose_action = choose_action
		self._external_update_q_values = update_q_values
		self._external_update_sarsa_values = update_sarsa_values
		self._external_reward_func = reward_func

	def choose_action(self, state: int) -> int:
		return self._external_choose_action(state, self.q_values)

	def get_reward(self, state, action, next_state, dungeon):
		return self._external_reward_func(state, action, next_state, dungeon)

	def update_q_values(self, state, action, reward, next_state, dungeon):
		self._external_update_q_values(self, state, action, reward, next_state, dungeon)

	def update_sarsa_values(self, state, action, reward, next_state, next_action, dungeon):
		self._external_update_sarsa_values(self, state, action, reward, next_state, next_action, dungeon)

	def run_episode(self, dungeon, nstep=1000, gamma = 1.0):
		states = [dungeon.start]
		actions = []
		rewards = []
		
		self.visited.fill(0)
		self.visited[dungeon.start] += 1

		for step in range(nstep):
			state = states[-1]
			
			if dungeon.tiles[state] == 'e':
				break
				
			action = int(self.choose_action(state))
			actions.append(action)
			
			probs = dungeon.transitions[action, state]
			next_state = int(np.random.choice(dungeon.n_states, p=probs))
			states.append(next_state)
			
			self.visited[next_state] += 1
			
			reward = self.get_reward(state, action, next_state, dungeon)
			rewards.append(np.power(gamma,step)*reward)
			
			if self._external_update_q_values is not None:
				self.update_q_values(state, action, reward, next_state, dungeon)
			elif self._external_update_sarsa_values is not None:
				next_action = int(self.choose_action(next_state))
				self.update_sarsa_values(state, action, reward, next_state, next_action, dungeon)

		return np.array(states, dtype=np.int16), np.array(actions, dtype=np.int32), np.array(rewards, dtype=np.float32)