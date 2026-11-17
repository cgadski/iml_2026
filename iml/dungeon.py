import os
import numpy as np


class Dungeon:
	"""Environment definition and transition dynamics."""

	def __init__(self, s: str, p_slippery=0.0, hole_reset=True):

		# parameters
		self.tile_mat = self._read_dungeon(s)
		self.h, self.w = self.tile_mat.shape
		self.n_states = self.h * self.w
		self.tiles = self.tile_mat.flatten()
		self.p_slippery = p_slippery
		self.hole_reset = hole_reset

		# state
		self.start = self.tile_indices("s")[0]
		self.end = self.tile_indices("e")[0]
		self.rewards = -np.ones_like(self.tiles, dtype=float)

		# transitions
		self.transitions = self._build_transitions()

	def tile_indices(self, char: str) -> np.ndarray:
		return np.where(self.tiles == char)[0]

	def _read_dungeon(self, s: str) -> np.ndarray:
		lines = s.strip().split("\n")
		max_len = max(len(line) for line in lines)
		return np.array([list(line.ljust(max_len)) for line in lines])

	def _evaluate_action(self, action: np.ndarray, pos: np.ndarray) -> np.ndarray:
		start_pos = np.array([x[0] for x in np.where(self.tile_mat == "s")])

		# Action mapping: 0: Down, 1: Up, 2: Right, 3: Left
		directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
		pos_ = pos + directions[action] # Move
		pos_ = np.maximum(pos_, 0)  # Clip to bounds
		pos_ = np.minimum(pos_, np.array([self.h - 1, self.w - 1]))

		square_ = self.tile_mat[pos_[:, 0], pos_[:, 1], None]
		pos_ = np.where(square_ == "#", pos, pos_) # Don't move into walls
		
		return pos_


	def _build_transitions(self) -> np.ndarray:

		transitions = np.zeros((4, self.n_states, self.n_states))
		final_transitions = np.zeros_like(transitions)

		# Deterministic transitions
		for action in range(4):
			for y in range(self.h):
				for x in range(self.w):
					next_y, next_x = self._evaluate_action(
						action, np.array([[y, x]])
					).T
					state = y * self.w + x
					next_state = next_y[0] * self.w + next_x[0]
					transitions[action, state, next_state] = 1.0

		# Stocastic transitions: apply slippery action probabilities
		for action in range(4):
			for state in range(self.n_states):
				if self.p_slippery > 0:
					final_transitions[action, state] += ((1 - self.p_slippery) * transitions[action, state])
					slipped_action = (action - 1) % 4
					final_transitions[action, state] += (self.p_slippery * transitions[slipped_action, state])
				else:
					final_transitions[action, state] = transitions[action, state]

		# reset if hole/dragon
		if self.hole_reset:
			hole_indices = np.concatenate([self.tile_indices("o"), self.tile_indices("O")])
			if len(hole_indices) > 0:
				for action in range(4):
					for h_state in hole_indices:
						# Clear any existing transitions from this hole state
						final_transitions[action, h_state] = 0.0
						# Force all actions from a hole state to transition 100% to the start state
						final_transitions[action, h_state, self.start] = 1.0

		return final_transitions


