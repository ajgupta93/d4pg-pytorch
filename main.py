from __future__ import division
import gym
import os
import time
import argparse
from ddpg import DDPG
from utils import *
from shared_adam import SharedAdam
import numpy as np
from normalize_env import NormalizeAction
import torch.multiprocessing as mp

import datetime
import time
import pickle
from tensorboard import SummaryWriter



# converted_d1 = datetime.datetime.fromtimestamp(round(d1 / 1000))
# current_time_utc = datetime.datetime.utcnow()
#
# print((current_time_utc - converted_d1))
# print((current_time_utc - converted_d1).total_seconds() / 60)


from pdb import set_trace as bp

# Parameters
parser = argparse.ArgumentParser(description='async_ddpg')

parser.add_argument('--n_workers', type=int, default=4, help='how many training processes to use (default: 4)')
parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--gamma', default=0.99, type=float, help='')
parser.add_argument('--env', default='Pendulum-v0', type=str, help='Environment to use')
parser.add_argument('--max_steps', default=500, type=int, help='Maximum steps per episode')
parser.add_argument('--n_eps', default=2000, type=int, help='Maximum number of episodes')
parser.add_argument('--debug', default=True, type=bool, help='Print debug statements')
parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
parser.add_argument('--p_replay', default=0, type=int, help='Enable prioritized replay - based on TD error')
parser.add_argument('--v_min', default=-150.0, type=float, help='Minimum return')
parser.add_argument('--v_max', default=150.0, type=float, help='Maximum return')
parser.add_argument('--n_atoms', default=51, type=int, help='Number of bins')
parser.add_argument('--multithread', default=0, type=int, help='To activate multithread')
parser.add_argument('--n_steps', default=5, type=int, help='number of steps to rollout')
parser.add_argument('--logfile', default='logs', type=str, help='File name for the train log data')
parser.add_argument('--log_dir', default='train_logs', type=str, help='File name for the train log data')

args = parser.parse_args()


writer = SummaryWriter('runs/exp' +
                        ( '_' + args.env + '_') +
                        ('_PER' if args.p_replay else '' ) +        # PER
                        ( '_' + str(args.n_steps) + 'N' )  +        # N-steps
                        ( '_' + str(args.n_workers) + 'Workers' )# N-workers
                       )

env = NormalizeAction(gym.make(args.env).env)
env._max_episode_steps = args.max_steps
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Get observation and action space dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if discrete else env.action_space.shape[0]

global_returns = [(0, 0)]  # list of tuple(step, return)

def configure_env_params():
    # pass
    if args.env == 'Pendulum-v0':
        args.v_min = -300.
        args.v_max = 0.
    # elif args.env == 'InvertedPendulum-v1':
    #     args.v_min = -150.
    #     args.v_max = 150.
    # elif args.env == 'HalfCheetah-v1':
    #     args.v_min = -150.
    #     args.v_max = 150.
    # elif args.env == 'Ant-v1':
    #     args.v_min = -150.
    #     args.v_max = 150.
    # else:
    #     print("Undefined environment. Configure v_max and v_min for environment")


def global_model_eval(global_model, global_count):
    temp_model = DDPG(obs_dim=obs_dim, act_dim=act_dim, critic_dist_info=critic_dist_info)
    env = NormalizeAction(gym.make(args.env).env)
    env._max_episode_steps = 500

    while True:
        counter = to_numpy(global_count)[0]
        if counter >= 1000000:
            break

        temp_model.actor.load_state_dict(global_model.actor.state_dict())
        temp_model.critic.load_state_dict(global_model.critic.state_dict())

        temp_model.actor.eval()
        # bp()
        global global_returns
        state = env.reset()
        curr_return = 0
        step_count = 0
        while True:
            action = temp_model.actor(to_tensor(state.reshape((1,-1))))
            next_state, reward, done, _ = env.step(to_numpy(action).reshape(-1))
            curr_return += reward
            step_count+=1
            # print("Step count: ", step_count)
            if done or step_count > args.max_steps:
                break
            else:
                state = next_state
        global_returns.append((counter, 0.95*global_returns[-1][1] + 0.05*curr_return, curr_return))
        print("Global Steps: ", counter, "Global return: ", global_returns[-1][1], "Current return: ", curr_return)

        time.sleep(10)



class Worker(object):
    def __init__(self, name, optimizer_global_actor, optimizer_global_critic):
        self.env = NormalizeAction(gym.make(args.env).env)
        self.env._max_episode_steps = args.max_steps
        self.name = name
        self.ddpg = DDPG(obs_dim=obs_dim, act_dim=act_dim, env=self.env, memory_size=args.rmsize,\
                          batch_size=args.bsize, tau=args.tau, critic_dist_info=critic_dist_info, \
                          prioritized_replay=args.p_replay, gamma = args.gamma, n_steps = args.n_steps)
        self.ddpg.assign_global_optimizer(optimizer_global_actor, optimizer_global_critic)
        print('Intialized worker :',self.name)

    # warmup function to fill replay buffer initially
    def warmup(self):
        n_steps = 0
        self.ddpg.actor.eval()
        # for i in range(args.n_eps):
        #     state = self.env.reset()
        #     for j in range(args.max_steps):
        #
        state = self.env.reset()
        for n_steps in range(args.warmup):
            action = np.random.uniform(-1.0, 1.0, size=act_dim)
            next_state, reward, done, _ = self.env.step(action)
            # self.ddpg.replayBuffer.add(state, action, reward, next_state, done)

            # if j >= args.n_steps - 1:
            #     cum_reward = 0.
            #     exp_gamma = 1
            #     for k in range(-args.n_steps, 0):
            #         cum_reward += exp_gamma * episode_rewards[k]
            #         exp_gamma *= args.gamma
            #     self.ddpg.replayBuffer.add(episode_states[-args.n_steps].reshape(-1), episode_actions[-1], cum_reward,
            #                                next_state, done)

            if done:
                state = self.env.reset()
            else:
                state = next_state


    def work(self, global_ddpg, global_count):
        avg_reward_train = 0.
        avg_reward_test = 0.
        n_steps = 0
        if args.p_replay:
            self.warmup()

        self.ddpg.sync_local_global(global_ddpg)
        self.ddpg.hard_update()
        self.start_time = datetime.datetime.utcnow()

        # Logging variables
        self.train_logs = {}
        self.train_logs['avg_reward_train'] = []
        self.train_logs['avg_reward_test'] = []
        self.train_logs['total_reward_train'] = []
        self.train_logs['total_reward_test'] = []
        self.train_logs['time'] = []
        self.train_logs['x_val'] = []
        self.train_logs['info_summary'] = "Distributional DDPG_" + str(args.n_steps) + 'N'
        if args.p_replay:
            self.train_logs['info_summary'] = self.train_logs['info_summary'] + ' + PER'
        self.train_logs['x'] = 'steps'
        step_counter = 0
        for i in range(args.n_eps):
            state = self.env.reset()
            total_reward_train = 0.
            episode_states = []
            episode_rewards = []
            episode_actions = []

            for j in range(args.max_steps):
                self.ddpg.actor.eval()

                state = state.reshape(1, -1)
                noise = self.ddpg.noise.sample()
                action = np.clip(to_numpy(self.ddpg.actor(to_tensor(state))).reshape(-1, ) + noise, -1.0, 1.0)
                next_state, reward, done, _ = self.env.step(action)
                total_reward_train += reward

                #### n-steps buffer
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)


                if j >= args.n_steps-1:
                    cum_reward = 0.
                    exp_gamma = 1
                    for k in range(-args.n_steps, 0):
                        cum_reward += exp_gamma * episode_rewards[k]
                        exp_gamma *= args.gamma
                    self.ddpg.replayBuffer.add(episode_states[-args.n_steps].reshape(-1), episode_actions[-args.n_steps], cum_reward, next_state, done)

                # self.ddpg.replayBuffer.add(state.reshape(-1), action, reward, next_state, done)

                self.ddpg.actor.train()
                self.ddpg.train(global_ddpg)
                step_counter += 1
                global_count += 1

                n_steps += 1

                if done:
                    break

                state = next_state
                # print("Episode ", i, "\t Step count: ", n_steps)

            avg_reward_train = 0.95*avg_reward_train + 0.05*total_reward_train

            state = self.env.reset()
            total_reward_test = 0.

            for j in range(args.max_steps):
                self.ddpg.actor.eval()

                state = state.reshape(1, -1)
                action = to_numpy(self.ddpg.actor(to_tensor(state))).reshape(-1)
                #if j==0:
                #    action += self.ddpg.noise.sample()
                action = np.clip(action, -1.0, 1.0)
                #print(action)
                next_state, reward, done, _ = self.env.step(action)
                total_reward_test += reward
                if done:
                    break
                else:
                    state = next_state

            avg_reward_test = 0.95*avg_reward_test + 0.05*total_reward_test


            if i%1==0:
                print('Episode ',i,'\tWorker :',self.name,\
                      '\tAvg Reward Train:',avg_reward_train,'\tTotal reward train :',total_reward_train,\
                      '\tAvg Reward Test:',avg_reward_test,'\tTotal reward test :',total_reward_test, '\tSteps :',n_steps)
                writer.add_scalar('train_reward', total_reward_train, n_steps)
                writer.add_scalar('test_reward', total_reward_test, n_steps)

                self.train_logs['avg_reward_train'].append(avg_reward_train)
                self.train_logs['avg_reward_test'].append(avg_reward_test)
                self.train_logs['total_reward_train'].append(total_reward_train)
                self.train_logs['total_reward_test'].append(total_reward_test)
                self.train_logs['time'].append((datetime.datetime.utcnow()-self.start_time).total_seconds()/60)
                self.train_logs['x_val'].append(step_counter)
                with open(args.logfile, 'wb') as fHandle:
                    pickle.dump(self.train_logs, fHandle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(args.logfile_latest, 'wb') as fHandle:
                    pickle.dump(self.train_logs, fHandle, protocol=pickle.HIGHEST_PROTOCOL)

            self.ddpg.noise.reset()


if __name__ == '__main__':
    configure_env_params()
    critic_dist_info = {'type': 'categorical', \
                        'v_min': args.v_min, \
                        'v_max': args.v_max, \
                        'n_atoms': args.n_atoms}
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args.logfile_latest = args.log_dir + '/' + args.logfile + '_' + args.env + '_latest_DistDDPG_' + str(args.n_steps) + 'N' + ('+PER' if args.p_replay else '') + '.pkl'
    args.logfile = args.log_dir + '/' + args.logfile + '_' + args.env + '_DistDDPG_' + str(args.n_steps) + 'N' + ('+PER_' if args.p_replay else '') + time.strftime("%Y%m%d-%H%M%S") + '.pkl'

    global_ddpg = DDPG(obs_dim=obs_dim, act_dim=act_dim, env=env, memory_size=args.rmsize,\
                        batch_size=args.bsize, tau=args.tau, critic_dist_info=critic_dist_info, gamma = args.gamma, n_steps = args.n_steps)
    optimizer_global_actor = SharedAdam(global_ddpg.actor.parameters(), lr=(1e-4)/float(args.n_workers))
    optimizer_global_critic = SharedAdam(global_ddpg.critic.parameters(), lr=(1e-4)/float(args.n_workers))#, weight_decay=1e-02)
    global_count = to_tensor(np.zeros(1), requires_grad=False).share_memory_()

    global_ddpg.share_memory()

    if not args.multithread:
        worker = Worker(str(1), optimizer_global_actor, optimizer_global_critic)
        worker.work(global_ddpg, global_count)
    else:
        processes = []
        p = mp.Process(target=global_model_eval, args=[global_ddpg, global_count])
        p.start()
        processes.append(p)

        for i in range(args.n_workers):
            worker = Worker(str(i), optimizer_global_actor, optimizer_global_critic)
            p = mp.Process(target=worker.work, args=[global_ddpg, global_count])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
