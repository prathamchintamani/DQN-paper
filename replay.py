import collections as col
import random

class ReplayBuffer(col.dequeue):
    def __init__(self, capacity):
        super().__init__(maxlen = capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones