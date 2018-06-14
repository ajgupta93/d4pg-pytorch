import numpy as np
import pdb
import math
from models import actor, critic
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from random_process import OrnsteinUhlenbeckProcess, GaussianNoise
from utils import *
from her_replay_memory import Replay  # SequentialMemory as Replay
from pdb import set_trace as bp
from prioritized_replay_memory import PrioritizedReplayBuffer, LinearSchedule


class DDPG:
    replayBuffer = None  # type: Replay

    def __init__(self, obs_dim, act_dim, env=None, memory_size=50000, batch_size=64, \
                 lr_critic=1e-4, lr_actor=1e-4, gamma=0.99, tau=0.05, prioritized_replay=True, \
                 critic_dist_info=None, n_steps=1, \
                 max_transitions_per_episode=100, her=None, her_ratio=0.8, n_episode_per_worker=2):

        self.gamma = gamma
        self.n_steps = n_steps
        self.n_step_gamma = self.gamma ** self.n_steps
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.memory_size = memory_size
        self.tau = tau
        self.env = env
        self.her = her
        self.her_ratio = her_ratio if self.her else 0

        self.n_episode_per_worker = n_episode_per_worker
        self.max_transitions_per_episode = max_transitions_per_episode
        ##   critic_dist_info:
        # dictionary with information about critic output distribution.
        # parameters:
        # 1. distribution_type = 'categorical' or 'mixture_of_gaussian'
        #    if 'categorical':
        #       a.
        #    if 'mixture_of_gaussian':
        #       b.

        self.dist_type = critic_dist_info['type']
        if critic_dist_info['type'] == 'categorical':
            self.v_min = critic_dist_info['v_min']
            self.v_max = critic_dist_info['v_max']
            self.n_atoms = critic_dist_info['n_atoms']
            self.delta = (self.v_max - self.v_min) / float(self.n_atoms - 1)
            self.bin_centers = np.array([self.v_min + i * self.delta for i in range(self.n_atoms)]).reshape(-1, 1)
        elif critic_dist_info['type'] == 'mixture_of_gaussian':
            # TODO
            pass
        else:
            print("Error: Unsupported distribution type")
            # TODO
            # throw exception

        # actor
        self.actor = actor(input_size=obs_dim, output_size=act_dim)
        self.actor_target = actor(input_size=obs_dim, output_size=act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # critic
        self.critic = critic(state_size=obs_dim, action_size=act_dim, dist_info=critic_dist_info)
        self.critic_target = critic(state_size=obs_dim, action_size=act_dim, dist_info=critic_dist_info)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # critic loss
        self.critic_loss = nn.CrossEntropyLoss()

        # noise
        # self.noise = OrnsteinUhlenbeckProcess(dimension=act_dim, num_steps=5000)
        self.noise = GaussianNoise(dimension=act_dim, num_epochs=5000)

        # replay buffer
        self.prioritized_replay = prioritized_replay
        if self.prioritized_replay:
            # Open AI prioritized replay memory
            self.replayBuffer = PrioritizedReplayBuffer(self.memory_size, alpha=0.6)
            prioritized_replay_beta0 = 0.4  # type: float
            prioritized_replay_beta_iters = 100000
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters, \
                                                initial_p=prioritized_replay_beta0, \
                                                final_p=1.0)
            self.prioritized_replay_eps = 1e-6
        else:
            self.replayBuffer = Replay(self.memory_size, self.env, n_steps=self.n_steps,
                                       gamma=self.gamma)  # <- self implemented memory buffer

    def add_new_episodes(self, n_episodes=None):
        # Add n_episodes in replay buffer
        n_episodes = n_episodes if n_episodes is not None else self.n_episode_per_worker
        for i in range(n_episodes):
            episode = []
            state = self.env.reset()
            for _ in range(self.max_transitions_per_episode):
                o, a_g, d_g = state['observation'], state['achieved_goal'], state['desired_goal']
                action = to_numpy(self.actor(to_tensor(np.concatenate((o, d_g)))))
                action = np.clip(action + self.noise.sample(), -1, 1)
                next_state, reward, done, info = self.env.step(action)
                o_n, a_g2, done, = next_state['observation'], next_state['achieved_goal'], bool(info['is_success'])
                done = bool(info['is_success'])

                ##### o: current state, o_n: next_state, a_g: achieved goal(at current state), a_g2: achieved goal(next state)
                ##### d_g: the desired goal(target)
                transition = {'o': o, 'o_n': o_n, 'a_g': a_g, 'a_g2': a_g2, 'done': done, 'info': info, 'r': reward,
                              'act': action, 'd_g': d_g}

                episode.append(transition)
                state = next_state
            self.replayBuffer.add_episode(episode)

    def sample_batch(self, batch_size=None):
        weights = None
        batch_idxes = None
        states = [];
        actions = [];
        rewards = [];
        next_states = [];
        terminates = []

        batch_size = batch_size if batch_size is not None else self.batch_size
        batch_episodes = np.array(self.replayBuffer.sample(self.n_episode_per_worker))

        # Sample indexes corresponding to episode and transition from those episodes
        ep_indexes = np.random.randint(0, self.n_episode_per_worker, batch_size)
        trans_indexes = np.random.randint(0, self.max_transitions_per_episode, batch_size)

        # Randomly select transitions which will have HER goals
        her_indexes = np.random.binomial(n=1, p=self.her_ratio, size=batch_size).astype(bool)
        n_her = her_indexes.sum()
        sampled_transitions = batch_episodes[ep_indexes, trans_indexes]

        # sample episodes
        offset = (np.random.uniform(size=batch_size) * (self.max_transitions_per_episode - trans_indexes)).astype(int)
        future = (batch_episodes[ep_indexes, trans_indexes + offset])
        for i, is_her in enumerate(her_indexes):
            curr_transition = sampled_transitions[i].copy()
            if is_her:
                # Replace current goal by any future goal
                # curr_transition['r'] = self.env.compute_reward(curr_transition['a_g2'],
                #                                                future[i]['a_g2'].copy(),
                #                                                curr_transition['info'])
                curr_transition['d_g'] = future[i]['a_g2'].copy()
                # curr_transition['done'] = bool(curr_transition['r'] == 0)  # reward    -1 desired goal not reached,
                                                                           #            0 reached
            states.append(np.concatenate((curr_transition['o'], curr_transition['d_g'])))
            actions.append(curr_transition['act'])
            rewards.append(curr_transition['r'])
            next_states.append(np.concatenate((curr_transition['o_n'], curr_transition['d_g'])))
            terminates.append(curr_transition['done'])

        states = np.array(states, dtype=np.float).reshape(batch_size, -1)
        actions = np.array(actions, dtype=np.float).reshape(batch_size, -1)
        rewards = np.array(rewards, dtype=np.float).reshape(batch_size, -1)
        next_states = np.array(next_states, dtype=np.float).reshape(batch_size, -1)
        terminates = np.array(terminates, dtype=np.bool).reshape(batch_size, -1)
        return states, actions, rewards, next_states, terminates, weights, batch_idxes

    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()

    def assign_global_optimizer(self, optimizer_global_actor, optimizer_global_critic):
        self.optimizer_global_actor = optimizer_global_actor
        self.optimizer_global_critic = optimizer_global_critic

    def copy_gradients(self, model_local, model_global):
        for param_local, param_global in zip(model_local.parameters(), model_global.parameters()):
            if param_global.grad is not None:
                return
            param_global._grad = param_local.grad

    def update_target_parameters(self):
        # Soft update of actor_target
        for parameter_target, parameter_source in zip(self.actor_target.parameters(), self.actor.parameters()):
            parameter_target.data.copy_((1 - self.tau) * parameter_target.data + self.tau * parameter_source.data)
        # Soft update of critic_target
        for parameter_target, parameter_source in zip(self.critic_target.parameters(), self.critic.parameters()):
            parameter_target.data.copy_((1 - self.tau) * parameter_target.data + self.tau * parameter_source.data)

    def sync_local_global(self, global_model):
        self.actor.load_state_dict(global_model.actor.state_dict())
        self.critic.load_state_dict(global_model.critic.state_dict())

    def reproj_categorical_dist(self, target_z_dist, rewards, terminates):
        rewards = rewards.reshape((-1, 1))
        terminates = terminates.reshape((-1, 1))
        batch_size = rewards.shape[0]
        m_prob = np.zeros((batch_size, self.n_atoms))

        tz = rewards + self.n_step_gamma * (1 - terminates) * self.bin_centers.T
        tz = np.minimum(self.v_max, np.maximum(self.v_min, tz))
        bj = (tz - self.v_min) / self.delta
        m_l, m_u = np.floor(bj).astype(np.int64), np.ceil(bj).astype(np.int64)
        m_l[(m_u > 0) * (m_l == m_u)] -= 1
        m_u[(m_l < (self.n_atoms - 1)) * (m_l == m_u)] += 1

        offset = np.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).reshape((-1, 1)).repeat(self.n_atoms,
                                                                                                     axis=1).astype(
            np.int64)
        np.add.at(m_prob.reshape(-1), (m_l + offset).reshape(-1),
                  (target_z_dist * (m_u.astype(np.float64) - bj)).reshape(-1))
        np.add.at(m_prob.reshape(-1), (m_u + offset).reshape(-1),
                  (target_z_dist * (bj - m_l.astype(np.float64))).reshape(-1))

        return m_prob

    def reproject2(self, target_z_dist, rewards, terminates):
        try:
            # next_distr = next_distr_v.data.cpu().numpy()

            rewards = rewards.reshape(-1)
            terminates = terminates.reshape(-1).astype(bool)
            # dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
            # batch_size = len(rewards)
            proj_distr = np.zeros((self.batch_size, self.n_atoms), dtype=np.float32)

            # pdb.set_trace()

            for atom in range(self.n_atoms):
                tz_j = np.minimum(self.v_max,
                                  np.maximum(self.v_min, rewards + (self.v_min + atom * self.delta) * self.gamma))
                b_j = (tz_j - self.v_min) / self.delta
                l = np.floor(b_j).astype(np.int64)
                u = np.ceil(b_j).astype(np.int64)
                eq_mask = (u == l).astype(bool)
                proj_distr[eq_mask, l[eq_mask]] += target_z_dist[eq_mask, atom]
                ne_mask = (u != l).astype(bool)
                proj_distr[ne_mask, l[ne_mask]] += target_z_dist[ne_mask, atom] * (u - b_j)[ne_mask]
                proj_distr[ne_mask, u[ne_mask]] += target_z_dist[ne_mask, atom] * (b_j - l)[ne_mask]

            if terminates.any():
                proj_distr[terminates] = 0.0
                tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards[terminates]))
                b_j = (tz_j - self.v_min) / self.delta
                l = np.floor(b_j).astype(np.int64)
                u = np.ceil(b_j).astype(np.int64)
                eq_mask = (u == l).astype(bool)
                eq_dones = terminates.copy()
                eq_dones[terminates] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l] = 1.0
                ne_mask = (u != l).astype(bool)
                ne_dones = terminates.copy()
                ne_dones[terminates] = ne_mask.astype(bool)
                if ne_dones.any():
                    proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
                    proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
        except Exception as e:
            print(e)
            bp()
        return proj_distr

    def sample(self, batch_size=None):
        weights = None
        batch_idxes = None
        # bp()
        if self.prioritized_replay:
            experience = self.replayBuffer.sample(batch_size, beta=self.beta_schedule.value())
            (states, actions, rewards, next_states, terminates, weights, batch_idxes) = experience
        else:
            # weights and batch_idxes = None: for non-prioritized_replay_buffer
            states, actions, rewards, next_states, terminates = self.replayBuffer.sample(self.batch_size)
        return states, actions, rewards, next_states, terminates, weights, batch_idxes

    def train(self, global_model, update_target=True):
        # sample from Replay
        states, actions, rewards, next_states, terminates, weights, batch_idxes = self.sample_batch(self.batch_size)

        # update critic (create target for Q function)
        target_z_dist = self.critic_target(to_tensor(next_states, volatile=True), \
                                           self.actor_target(to_tensor(next_states, volatile=True)))

        q_dist = self.critic(to_tensor(states), to_tensor(actions))  # n_sample, n_atoms

        qdist_loss = None  # dummy variable to remove redundant warnings
        if self.dist_type == 'categorical':

            # reprojected_dist = self.reproj_categorical_dist(target_z_dist.cpu().data.numpy(), rewards, terminates)
            reprojected_dist = self.reproject2(target_z_dist.cpu().data.numpy(), rewards, terminates)

            # qdist_loss = self.critic_loss(q_dist, to_tensor(reprojected_dist, requires_grad=False))
            qdist_loss = -(to_tensor(reprojected_dist, requires_grad=False) * torch.log(q_dist + 1e-010)).sum(
                dim=1).mean()
            # qdist_loss = -torch.from_numpy(reprojected_dist)*F.log_softmax(q_dist, dim=1)
            # qdist_loss = qdist_loss.sum(dim=1).mean()
            if self.prioritized_replay:
                td_errors = -(to_tensor(reprojected_dist, requires_grad=False) * q_dist)
                td_errors = td_errors.sum(dim=1)
            # qdist_loss = td_errors.mean()
        elif self.dist_type == 'mixture_of_gaussian':
            # TODO
            pass

        # critic optimizer and backprop step (feed in target and predicted values to self.critic_loss)
        self.critic.zero_grad()
        qdist_loss.backward()
        self.copy_gradients(self.critic, global_model.critic)
        self.optimizer_global_critic.step()

        # update actor (formulate the loss wrt which actor is updated)
        # bp()
        policy_loss = self.critic(to_tensor(states), \
                                  self.actor(to_tensor(states)))
        policy_loss = -policy_loss.matmul(to_tensor(self.bin_centers)).mean()

        # actor optimizer and backprop step (loss_actor.backward())
        self.actor.zero_grad()
        policy_loss.backward()
        self.copy_gradients(self.actor, global_model.actor)
        self.optimizer_global_actor.step()

        # copy global network weights to local
        self.sync_local_global(global_model)

        # soft-update of target
        if update_target:
            self.update_target_parameters()

        if self.prioritized_replay:  # update priorities
            new_priorities = np.abs(to_numpy(td_errors)) + self.prioritized_replay_eps
            self.replayBuffer.update_priorities(batch_idxes, new_priorities)
