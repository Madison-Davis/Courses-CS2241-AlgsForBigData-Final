# prioritized_replay_buffer.py


# ++++++++++++ Imports and Installs ++++++++++++ #
import numpy as np
from sac.unused_files.segment_tree import SumSegmentTree, MinSegmentTree


# ++++++++++++++ Class Definition ++++++++++++++ #
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-6):
        """
        Args:
            capacity (int): maximum number of transitions.
            alpha (float): how much prioritization is used (0 - no prioritization, 1 - full prioritization).
            beta (float): importance sampling exponent for bias correction.
            epsilon (float): small constant to avoid zero priority.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.pos = 0  # points to the next index to overwrite
        self.size = 0  # current number of transitions stored
        self.storage = [None] * capacity

        # To use segment trees efficiently, we want the underlying tree capacity to be a power of 2.
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2

        self.tree_capacity = tree_capacity
        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)
        self.max_priority = 1.0  # initial maximum priority

    def add(self, obs, next_obs, action, reward, done, info):
        """
        Add a new transition to the buffer.
        """
        data = (obs, next_obs, action, reward, done, info)
        self.storage[self.pos] = data

        # Use the current maximum priority for new transitions.
        priority = self.max_priority ** self.alpha
        self.sum_tree.update(self.pos, priority)
        self.min_tree.update(self.pos, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of transitions with indices and importance-sampling weights.
        Returns:
            A tuple: (observations, next_observations, actions, rewards, dones, infos, indices, weights)
        """
        indices = []
        weights = []
        batch = []

        total_priority = self.sum_tree.sum()
        # Minimum probability for any sample (for normalization)
        min_priority = self.min_tree.query(0, self.size)
        min_prob = min_priority / total_priority if total_priority > 0 else 0
        max_weight = (min_prob * self.size) ** (-self.beta) if min_prob > 0 else 1

        for _ in range(batch_size):
            mass = np.random.random() * total_priority
            idx = self.sum_tree.find_prefixsum_idx(mass)
            # Ensure idx is within the current size of the buffer
            idx = min(idx, self.size - 1)
            indices.append(idx)
            batch.append(self.storage[idx])
            p_sample = self.sum_tree.tree[idx + self.sum_tree.capacity] / total_priority
            weight = (p_sample * self.size) ** (-self.beta) if p_sample > 0 else 1.0
            weights.append(weight / max_weight)

        # Unpack the batch of transitions
        obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch, infos_batch = zip(*batch)
        return (np.array(obs_batch), np.array(next_obs_batch), np.array(actions_batch),
                np.array(rewards_batch), np.array(dones_batch), infos_batch,
                np.array(indices), np.array(weights))

    def update_priorities(self, indices, priorities):
        """
        Update the priorities for the sampled transitions.
        Args:
            indices (array-like): indices of the transitions to update.
            priorities (array-like): new priority values (e.g., TD errors) for each transition.
        """
        for idx, priority in zip(indices, priorities):
            self.max_priority = max(self.max_priority, priority)
            updated_priority = (priority + self.epsilon) ** self.alpha
            self.sum_tree.update(idx, updated_priority)
            self.min_tree.update(idx, updated_priority)
