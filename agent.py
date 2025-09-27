import DQN as dqn
import replay
from torch.optim import RMSprop
import random
import torch

class agent:
    def __init__(self, action_size, batch_size=32, gamma=0.99, lr=0.001, epsilon = 1.0, epsilon_decay = 0.000009, epsilon_min = 0.1):
        self.action_size = action_size
        self.policy = dqn.DQN(action_size)
        self.target = dqn.DQN(action_size)
        self.memory = replay.ReplayBuffer(10000)
        self.batch_size = batch_size
        self.gamma = gamma 
        self.lr = lr
        self.steps = 0
        self.optimizer = RMSprop(self.policy.parameters(), lr=self.lr)
        self.lossfn = torch.nn.SmoothL1Loss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        self.steps += 1
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.policy(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)
        next_states = torch.FloatTensor(next_states)

        q_pred = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next = self.target(next_states).max(1)[0]

        if dones.any():
            q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        loss = self.lossfn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target.load_state_dict(self.policy.state_dict())