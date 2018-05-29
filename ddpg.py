from models import actor, critic
import torch
import torch.optim as optim
import torch.nn as nn
from random_process import OrnsteinUhlenbeckProcess
from utils import *
from replay_memory import Replay#SequentialMemory as Replay

class DDPG:
    def __init__(self, obs_dim, act_dim, env, memory_size=50000, batch_size=64,\
                 lr_critic=1e-3, lr_actor=1e-4, gamma=0.99, tau=0.001):
        
        self.gamma          = gamma
        self.batch_size     = batch_size
        self.obs_dim        = obs_dim
        self.act_dim        = act_dim
        self.memory_size    = memory_size
        self.tau            = tau
        self.env            = env

        # actor
        self.actor = actor(input_size = obs_dim, output_size = act_dim)
        self.actor_target = actor(input_size = obs_dim, output_size = act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # critic
        self.critic = critic(state_size = obs_dim, action_size = act_dim, output_size=1)
        self.critic_target = critic(state_size = obs_dim, action_size = act_dim, output_size=1)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # critic loss
        self.critic_loss = nn.MSELoss()
        
        # noise
        self.noise = OrnsteinUhlenbeckProcess(dimension=act_dim, num_steps=5000)

        # replay buffer 
        #self.replayBuffer = Replay(self.memory_size, window_length=1)
        self.replayBuffer = Replay(self.memory_size, self.env)

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

    def sync_grad_with_global_model(self, global_model):
        self.copy_gradients(self.actor, global_model.actor)
        self.copy_gradients(self.critic, global_model.critic)

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

    def train(self, global_model):
        # sample from Replay
        #states, actions, rewards, next_states, terminates = self.replayBuffer.sample_and_split(self.batch_size)
        states, actions, rewards, next_states, terminates = self.replayBuffer.sample(self.batch_size)

        # update critic (create target for Q function)
        target_qvalues = self.critic_target(to_tensor(next_states, volatile=True),\
                                            self.actor_target(to_tensor(next_states, volatile=True)))
        y = to_numpy(to_tensor(rewards) +\
                     self.gamma*to_tensor(1-terminates)*target_qvalues)

        q_values = self.critic(to_tensor(states),
                               to_tensor(actions))
        qvalue_loss = self.critic_loss(q_values, to_tensor(y, requires_grad=False))
        
               
        # critic optimizer and backprop step (feed in target and predicted values to self.critic_loss)
        self.critic.zero_grad()
        qvalue_loss.backward()
        self.copy_gradients(self.critic, global_model.critic)
        self.optimizer_global_critic.step()

        # update actor (formulate the loss wrt which actor is updated)
        policy_loss = -self.critic(to_tensor(states),\
                                   self.actor(to_tensor(states)))
        policy_loss = policy_loss.mean()

        # actor optimizer and backprop step (loss_actor.backward())
        self.actor.zero_grad()
        policy_loss.backward()
        self.copy_gradients(self.actor, global_model.actor)
        self.optimizer_global_actor.step()

        # copy global network weights to local
        self.sync_local_global(global_model)

        # soft-update of target
        self.update_target_parameters()