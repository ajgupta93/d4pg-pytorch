from __future__ import division
import gym
import os
import time
import pybullet_envs
import argparse
from ddpg import DDPG
from utils import *
from shared_adam import SharedAdam
import numpy as np
from normalize_env import NormalizeAction
import torch.multiprocessing as mp
from pdb import set_trace as bp
import datetime
import time
import pickle
from tensorboard import SummaryWriter

from pdb import set_trace as bp

# Parameters
parser = argparse.ArgumentParser(description='async_ddpg')
parser.add_argument('--n_workers', type=int, default=4, help='how many training processes to use (default: 4)')
parser.add_argument('--rmsize', default=int(1e6), type=int, help='memory size')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--gamma', default=0.99, type=float, help='')
parser.add_argument('--env', default='Pendulum-v0', type=str, help='Environment to use')
parser.add_argument('--max_steps', default=50, type=int, help='Maximum steps per episode')
parser.add_argument('--n_eps', default=2000, type=int, help='Maximum number of episodes')
parser.add_argument('--debug', default=True, type=bool, help='Print debug statements')
parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory')
parser.add_argument('--p_replay', default=0, type=int, help='Enable prioritized replay - based on TD error')
parser.add_argument('--v_min', default=-50.0, type=float, help='Minimum return')
parser.add_argument('--v_max', default=0., type=float, help='Maximum return')
parser.add_argument('--n_atoms', default=51, type=int, help='Number of bins')
parser.add_argument('--multithread', default=0, type=int, help='To activate multithread')
parser.add_argument('--n_steps', default=1, type=int, help='number of steps to rollout')
parser.add_argument('--logfile', default='logs', type=str, help='File name for the train log data')
parser.add_argument('--log_dir', default='train_logs', type=str, help='File name for the train log data')
parser.add_argument('--her', default=0, type=int, help='Control variable for Hindsight experience replay')
parser.add_argument('--her_ratio', default=0.8, type=float, help='Control variable for Hindsight experience replay')
args = parser.parse_args()

path = 'runs/exp' + \
                    ( '_' + args.env + '_') + \
                    ('_PER' if args.p_replay else '' ) + \
                    ('_HER' if args.her else '' ) +  \
                    (('_'+str(args.her_ratio)) if args.her else '') + \
                    ( '_' + str(args.n_steps) + 'N' )  +  \
                    ( '_' + str( args.n_workers if args.multithread else 1 ) + 'Workers')
writer = SummaryWriter(path)
print('Path: ', path)

env = NormalizeAction(gym.make(args.env).env)
env._max_episode_steps = args.max_steps
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Get observation and action space dimensions
if not args.her:
    obs_dim = env.observation_space.shape[0]
else:
    ss = env.reset()
    state_dim = ss['observation'].shape[0]
    goal_dim = ss['achieved_goal'].shape[0]
    obs_dim = state_dim + goal_dim
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
#


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
                          prioritized_replay=args.p_replay, gamma = args.gamma, n_steps = args.n_steps,\
                          her=args.her, her_ratio=args.her_ratio, n_episode_per_worker= 5)
        self.ddpg.assign_global_optimizer(optimizer_global_actor, optimizer_global_critic)
        print('Intialized worker :',self.name)

    # warmup function to fill replay buffer initially
    def warmup(self):
        self.ddpg.actor.eval()
        self.ddpg.add_new_episodes(5)

    def work(self, global_ddpg, global_count):
        avg_reward_test = 0.
        self.ddpg.sync_local_global(global_ddpg)
        self.ddpg.hard_update()

        self.warmup()

        step_counter = 0
        for i in range(args.n_eps):
            for cycle in range(50):
                self.ddpg.add_new_episodes(10)
                self.ddpg.update_target_parameters()
                for j in range(40):  # type: int
                    self.ddpg.actor.train()
                    self.ddpg.train(global_ddpg, update_target=False)
                    step_counter += 1
                    global_count += 1
                    if j%5 == 0:
                        self.ddpg.update_target_parameters()

                success = 0
                success_steps = []
                n_trials = 10
                for k in range(n_trials):
                    total_reward_test = 0.
                    state = self.env.reset()
                    for j in range(args.max_steps):
                        self.ddpg.actor.eval()
                        state = np.concatenate((state['observation'], state['desired_goal'])).reshape(1, -1)
                        action = to_numpy(self.ddpg.actor(to_tensor(state))).reshape(-1)
                        action = np.clip(action, -1.0, 1.0)
                        next_state, reward, done, info = self.env.step(action)
                        done = bool(info['is_success'])
                        total_reward_test += reward
                        if done:
                            success += 1
                            success_steps.append(j)
                            break
                        else:
                            state = next_state
                    avg_reward_test = 0.95 * avg_reward_test + 0.05 * total_reward_test
                success_rate = float(success)/n_trials

                print("Epoch: ", i, "\t Cycle: ", cycle, "\t ",
                      '\tAvg Reward Test:',avg_reward_test,'\tTest success steps :',success_steps, '\t Success Rate', success_rate, '\tSteps :',step_counter)
                writer.add_scalar('avg_test_reward', avg_reward_test, step_counter)
                writer.add_scalar('success_rate', success_rate, step_counter)

            # self.ddpg.noise.reset()
                torch.save(self.ddpg.actor.state_dict(), path+'/actor.pth')
                torch.save(self.ddpg.critic.state_dict(), path+'/critic.pth')


if __name__ == '__main__':
    configure_env_params()
    critic_dist_info = {'type': 'categorical', \
                        'v_min': args.v_min, \
                        'v_max': args.v_max, \
                        'n_atoms': args.n_atoms}

    global_ddpg = DDPG(obs_dim=obs_dim, act_dim=act_dim, env=env, memory_size=args.rmsize,\
                        batch_size=args.bsize, tau=args.tau, critic_dist_info=critic_dist_info, gamma = args.gamma, n_steps = args.n_steps)
    optimizer_global_actor = SharedAdam(global_ddpg.actor.parameters(), lr=(1e-3)/float(args.n_workers))
    optimizer_global_critic = SharedAdam(global_ddpg.critic.parameters(), lr=(1e-3)/float(args.n_workers))#, weight_decay=1e-02)
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
