import random
import numpy as np

class Replay(object):
    def __init__(self, max_size, env, n_steps=1, gamma=0.99):
        self.buffer = []
        self.capacity = max_size
        self.position = 0
        self.env = env
        self.n_steps = n_steps
        self.gamma = gamma
        self.initialize(init_length=1000)

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (np.asarray(state), action, reward,\
                                      np.asarray(next_state), done)
        self.position = (self.position+1)%self.capacity

    def initialize(self, init_length):
        state = self.env.reset()
        step_counter = 0
        episode_states = []
        episode_rewards = []
        episode_actions = []


        while True:
            action = np.random.uniform(-1.0, 1.0, size=self.env.action_space.shape)
            next_state, reward, done, _ = self.env.step(action)

            #### n-steps buffer
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            if step_counter >= self.n_steps - 1:
                cum_reward = 0.
                exp_gamma = 1
                for k in range(-self.n_steps, 0):
                    cum_reward += exp_gamma * episode_rewards[k]
                    exp_gamma *= self.gamma
                self.add(episode_states[-self.n_steps].reshape(-1), episode_actions[-1], cum_reward,
                                           next_state, done)
                # self.add(state, action, reward, next_state, done)

            if len(self.buffer) >= init_length:
                break
            if done:
                state = self.env.reset()
                step_counter = 0
                episode_states = []
                episode_rewards = []
                episode_actions = []
            else:
                state = next_state
                step_counter += 1


    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminates = []
        samples = random.sample(self.buffer, batch_size)
        for state, action, reward, next_state, done in samples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminates.append(done)

        states = np.array(states, dtype=np.float).reshape(batch_size,-1)
        actions = np.array(actions, dtype=np.float).reshape(batch_size,-1)
        rewards = np.array(rewards, dtype=np.float).reshape(batch_size,-1)
        next_states = np.array(next_states, dtype=np.float).reshape(batch_size,-1)
        terminates = np.array(terminates, dtype=np.float).reshape(batch_size,-1)
        return states, actions, rewards, next_states, terminates
