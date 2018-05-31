import numpy as np
import math
from models import actor, critic
import torch
import torch.optim as optim
import torch.nn as nn
from random_process import OrnsteinUhlenbeckProcess
from utils import *
from replay_memory import Replay#SequentialMemory as Replay
from pdb import set_trace as bp
from prioritized_replay_memory import PrioritizedReplayBuffer, LinearSchedule

class DDPG:
    replayBuffer = None  # type: Replay

    def __init__(self, obs_dim, act_dim, env, memory_size=50000, batch_size=64,\
                 lr_critic=1e-3, lr_actor=1e-4, gamma=0.99, tau=0.001, prioritized_replay=True,\
                 critic_dist_info=None):
        
        self.gamma          = gamma
        self.batch_size     = batch_size
        self.obs_dim        = obs_dim
        self.act_dim        = act_dim
        self.memory_size    = memory_size
        self.tau            = tau
        self.env            = env
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
            self.delta = (self.v_max-self.v_min)/float(self.n_atoms-1)
            self.bin_centers = np.array([self.v_min+i*self.delta for i in range(self.n_atoms)]).reshape(-1,1)
        elif critic_dist_info['type'] == 'mixture_of_gaussian':
            #TODO
            pass
        else:
            print("Error: Unsupported distribution type")
            # TODO
            # throw exception

        # actor
        self.actor = actor(input_size = obs_dim, output_size = act_dim)
        self.actor_target = actor(input_size = obs_dim, output_size = act_dim)
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
        self.noise = OrnsteinUhlenbeckProcess(dimension=act_dim, num_steps=5000)

        # replay buffer
        self.prioritized_replay = prioritized_replay
        if self.prioritized_replay:
            # Open AI prioritized replay memory
            self.replayBuffer = PrioritizedReplayBuffer(self.memory_size, alpha=0.6)
            prioritized_replay_beta0 = 0.4  # type: float
            prioritized_replay_beta_iters  = 100000
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,\
                                                initial_p=prioritized_replay_beta0,\
                                                final_p=1.0)
            self.prioritized_replay_eps = 1e-6
        else:
            #self.replayBuffer = Replay(self.memory_size, window_length=1) #  <- Keras RL replay memory
            self.replayBuffer = Replay(self.memory_size, self.env)         #<- self implemented memory buffer


    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()

    def assign_global_optimizer(self, optimizer_global_actor, optimizer_global_critic):
        self.optimizer_global_actor = optimizer_global_actor
        self.optimizer_global_critic = optimizer_global_critic

    def copy_gradients(self, model_local, model_global ):
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
        # bp()
        rewards = rewards.reshape((-1,1))
        terminates = terminates.reshape((-1, 1))
        batch_size = rewards.shape[0]
        m_prob = np.zeros((batch_size, self.n_atoms))

        tz = rewards + self.gamma * (1 - terminates) * self.bin_centers.T
        tz = np.minimum(self.v_max, np.maximum(self.v_min, tz))
        bj = (tz - self.v_min) / self.delta
        m_l, m_u = np.floor(bj).astype(np.int64), np.ceil(bj).astype(np.int64)
        m_l[(m_u > 0) * (m_l == m_u)] -= 1
        m_u[(m_l < (self.n_atoms - 1)) * (m_l == m_u)] += 1

        offset = np.linspace(0, (batch_size-1)*self.n_atoms, batch_size).reshape((-1,1)).repeat(self.n_atoms, axis=1).astype(np.int64)
        np.add.at(m_prob.reshape(-1), (m_l+offset).reshape(-1),(target_z_dist*(m_u.astype(np.float64) - bj)).reshape(-1))
        np.add.at(m_prob.reshape(-1), (m_u+offset).reshape(-1),(target_z_dist * (bj-m_l.astype(np.float64))).reshape(-1))

        #             m_prob[i, m_l.astype(int)] += target_z_dist[i] * (m_u - bj)
        #             m_prob[i, m_u.astype(int)] += target_z_dist[i] * (bj - m_l)


        ###
        # for i in range(batch_size):
#            bp()
#             tz = np.minimum(self.v_max, np.maximum(self.v_min, rewards[i] + self.gamma * (1 - terminates[i][0]) * self.bin_centers.T))
#             bj = (tz - self.v_min) / self.delta
#             m_l, m_u = np.floor(bj), np.ceil(bj)
#             m_prob[i, m_l.astype(int)] += target_z_dist[i] * (m_u - bj)
#             m_prob[i, m_u.astype(int)] += target_z_dist[i] * (bj - m_l)
    ##


        # for i in range(batch_size):
        #     for j in range(self.n_atoms):
        #         tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * (1 - terminates[i]) * self.bin_centers[j]))
        #         bj = (tz - self.v_min) / self.delta
        #         m_l, m_u = math.floor(bj), math.ceil(bj)
        #         m_prob[i][int(m_l)] += target_z_dist[i][j] * (m_u - bj)
        #         m_prob[i][int(m_u)] += target_z_dist[i][j] * (bj - m_l)
        # m_prob.reshape((batch_size, self.n_atoms))
        return m_prob

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


    def train(self, global_model):
        # sample from Replay
        #states, actions, rewards, next_states, terminates = self.replayBuffer.sample_and_split(self.batch_size)
        # weights and batch_inxes = None for non-prioritized_replay_buffer
        # bp()
        states, actions, rewards, next_states, terminates, weights, batch_idxes = self.sample(self.batch_size)

        # update critic (create target for Q function)
        target_z_dist = self.critic_target(to_tensor(next_states, volatile=True),\
                                            self.actor_target(to_tensor(next_states, volatile=True)))
        q_dist = self.critic(to_tensor(states), to_tensor(actions))  # n_sample, n_atoms

        qdist_loss = None#dummy variable to remove redundant warnings
        if self.dist_type == 'categorical':
            reprojected_dist = self.reproj_categorical_dist(target_z_dist.cpu().data.numpy(), rewards, terminates)
            #qdist_loss = self.critic_loss(q_dist, to_tensor(reprojected_dist, requires_grad=False))
            qdist_loss = -(to_tensor(reprojected_dist, requires_grad=False)*torch.log(q_dist)).sum(dim=1).mean()
            td_errors = -(to_tensor(reprojected_dist, requires_grad=False) * q_dist)
            td_errors = td_errors.sum(dim=1)
            #qdist_loss = td_errors.mean()
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
        policy_loss = self.critic(to_tensor(states),\
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
        self.update_target_parameters()

        if self.prioritized_replay:         # update priorities
            new_priorities = np.abs(to_numpy(td_errors)) + self.prioritized_replay_eps
            # bp()
            self.replayBuffer.update_priorities(batch_idxes, new_priorities)
