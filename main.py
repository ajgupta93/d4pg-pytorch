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



# converted_d1 = datetime.datetime.fromtimestamp(round(d1 / 1000))
# current_time_utc = datetime.datetime.utcnow()
#
# print((current_time_utc - converted_d1))
# print((current_time_utc - converted_d1).total_seconds() / 60)


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

args = parser.parse_args()


path = 'runs/exp' + \
                    ( '_' + args.env + '_') + \
                    ('_PER' if args.p_replay else '' ) + \
                    ('_HER' if args.her else '' ) +  \
                    ( '_' + str(args.n_steps) + 'N' )  +  \
                    ( '_' + str( args.n_workers if args.multithread else 1 ) + 'Workers' )

writer = SummaryWriter(path)

env = NormalizeAction(gym.make(args.env).env)
env._max_episode_steps = args.max_steps
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Get observation and action space dimensions
if not args.her:
    obs_dim = env.observation_space.shape[0]
else:
    ss = env.reset()
    state_dim  = ss['observation'].shape[0]
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


def addExperienceToBuffer(ddpg, replay_buffer, env, her=False, her_ratio=0.8):
    # add experiences to buffer
    episode_buffer = []
    state = env.reset()

    for i in range(args.max_steps):
        # concat state and action
        inp = to_tensor(np.concatenate((state['observation'], state['desired_goal'])))
        action = to_numpy(ddpg.actor(inp))
        action = np.clip( action + ddpg.noise.sample(), -1, 1)
        next_state, reward, done, info = env.step(action)
        done = bool(info['is_success'])
        episode_buffer.append([state, action, reward, next_state, done, info])
        state = next_state
        if done:
            break

    if args.her and not done:
        # create dummy goal

        for t in range(len(episode_buffer)):

            # add transition -info
            s, a, r, s_n , d, i = episode_buffer[t]
            s = np.concatenate((s['observation'], s['desired_goal']))
            s_n = np.concatenate((s_n['observation'], s_n['desired_goal']))
            replay_buffer.add(s, a, r, s_n, d)

            # cater to her_ratio
            if np.random.uniform() < her_ratio:
                # all the her sampling code should be under this IF condition, however currently it is ignored

                # Goal strategy: future || other possible strategies: final
                future_transition = episode_buffer[np.random.randint(t, len(episode_buffer))]
                dummy_goal = future_transition[3]['achieved_goal']

                her_curr_state = np.concatenate((episode_buffer[t][0]['observation'], dummy_goal))
                her_next_state = np.concatenate((episode_buffer[t][3]['observation'], dummy_goal))

                # get update reward
                substitute_goal = dummy_goal.copy()
                her_reward = env.compute_reward( episode_buffer[t][3]['achieved_goal'], substitute_goal, episode_buffer[t][5] )
                # env.state = her_curr_state
                # _, her_reward, her_done, _ = env.step(future_transition[1])

                # Add her-transition to replay buffer
                her_done = True if her_reward == 0. else False #(substitute_goal == episode_buffer[t][3]['achieved_goal']).all()
                replay_buffer.add(her_curr_state, action, her_reward, her_next_state, her_done)
#    bp()


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

        self.ddpg.actor.eval()
        # bp()
        for i in range(5000//args.max_steps):
            addExperienceToBuffer(self.ddpg, self.ddpg.replayBuffer, self.env, her=args.her, her_ratio=0.8)
        # bp()
        return

        counter = 0
        state = self.env.reset()
        episode_states = []
        episode_rewards = []
        episode_actions = []
        while counter < args.warmup:

            action = to_numpy(self.ddpg.actor(to_tensor(state.reshape(-1))))  #np.random.uniform(-1.0, 1.0, size=act_dim)
            next_state, reward, done, _ = self.env.step( np.clip(action + self.ddpg.noise.sample(), -1, 1) )

            #### n-steps buffer
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            if len(episode_states) >= args.n_steps:
                cum_reward = 0.
                exp_gamma = 1
                for k in range(-args.n_steps, 0):
                    try:
                        cum_reward += exp_gamma * episode_rewards[k]
                    except:
                        bp()
                    exp_gamma *= args.gamma
                self.ddpg.replayBuffer.add(episode_states[-args.n_steps].reshape(-1), episode_actions[-1], cum_reward,
                                           next_state, done)
            if done:
                episode_states = []
                episode_rewards = []
                episode_actions = []
                state = self.env.reset()
            else:
                state = next_state
            counter += 1


    def work(self, global_ddpg, global_count):
        avg_reward_train = 0.
        avg_reward_test = 0.
        self.ddpg.sync_local_global(global_ddpg)
        self.ddpg.hard_update()
        self.start_time = datetime.datetime.utcnow()

        self.warmup()

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

            # state = self.env.reset()
            # total_reward_train = 0.
            # episode_states = []
            # episode_rewards = []
            # episode_actions = []
            #
            # for j in range(args.max_steps):
            #     self.ddpg.actor.eval()
            #
            #     state = state.reshape(1, -1)
            #     noise = self.ddpg.noise.sample()
            #     action = np.clip(to_numpy(self.ddpg.actor(to_tensor(state))).reshape(-1, ) + noise, -1.0, 1.0)
            #     next_state, reward, done, _ = self.env.step(action)
            #     total_reward_train += reward
            #
            #     #### n-steps buffer
            #     episode_states.append(state)
            #     episode_actions.append(action)
            #     episode_rewards.append(reward)
            #
            #
            #     if j >= args.n_steps-1:
            #         cum_reward = 0.
            #         exp_gamma = 1
            #         for k in range(-args.n_steps, 0):
            #             cum_reward += exp_gamma * episode_rewards[k]
            #             exp_gamma *= args.gamma
            #         self.ddpg.replayBuffer.add(episode_states[-args.n_steps].reshape(-1), episode_actions[-args.n_steps], cum_reward, next_state, done)
            #
            #     # self.ddpg.replayBuffer.add(state.reshape(-1), action, reward, next_state, done)
            #
        for i in range(args.n_eps):
            for cycle in range(50):
                for episode_count in range(16):
                    addExperienceToBuffer(self.ddpg, self.ddpg.replayBuffer, self.env, her=args.her, her_ratio=0.8)
                for j in range(40):
                    self.ddpg.actor.train()
                    self.ddpg.train(global_ddpg)
                    step_counter += 1
                    global_count += 1

                success = 0
                success_steps = []
                nTrials = 10
                for k in range(nTrials):
                    total_reward_test = 0.
                    episode_rewards = []
                    episode_states = []
                    episode_success = []
                    state = self.env.reset()
                    cc = 0
                    for j in range(args.max_steps):
                        cc+=1
                        self.ddpg.actor.eval()
                        state = np.concatenate((state['observation'], state['desired_goal']))
                        state = state.reshape(1, -1)
                        action = to_numpy(self.ddpg.actor(to_tensor(state))).reshape(-1)
                        action = np.clip(action, -1.0, 1.0)
                        next_state, reward, done, info = self.env.step(action)
                        done = bool(info['is_success'])
                        total_reward_test += reward
                        episode_rewards.append((j,reward))
                        episode_states.append((j,state))
                        episode_success.append((j,info['is_success']))
                        #if reward == 0 and j != 49:
                        #    bp()
                        if done:
                            success+=1
                            success_steps.append(j)
                            break
                        else:
                            state = next_state
                    #if total_reward_test > -50:
                    #    print("Reward: ", total_reward_test, "\t Done: ", done, "\t success: ", success)
                    #    print("Episode rewards \n", episode_rewards, "\n")
                    #    print("Episode rewards \n", episode_states, "\n")

                        #bp()
                    avg_reward_test = 0.95 * avg_reward_test + 0.05 * total_reward_test
                success_rate = float(success)/nTrials

                print("Epoch: ", i, "\t Cycle: ", cycle, "\t ",
                      '\tAvg Reward Test:',avg_reward_test,'\tTest success steps :',success_steps, '\t Success Rate', success_rate, '\tSteps :',step_counter)
                # writer.add_scalar('train_reward', total_reward_train, n_steps)
                writer.add_scalar('avg_test_reward', avg_reward_test, step_counter)
                writer.add_scalar('success_rate', success_rate, step_counter)

                # self.train_logs['avg_reward_train'].append(avg_reward_train)
                # self.train_logs['avg_reward_test'].append(avg_reward_test)
                # # self.train_logs['total_reward_train'].append(total_reward_train)
                # self.train_logs['total_reward_test'].append(total_reward_test)
                # self.train_logs['time'].append((datetime.datetime.utcnow()-self.start_time).total_seconds()/60)
                # self.train_logs['x_val'].append(step_counter)
                # with open(args.logfile, 'wb') as fHandle:
                #     pickle.dump(self.train_logs, fHandle, protocol=pickle.HIGHEST_PROTOCOL)
                # with open(args.logfile_latest, 'wb') as fHandle:
                #     pickle.dump(self.train_logs, fHandle, protocol=pickle.HIGHEST_PROTOCOL)

            # self.ddpg.noise.reset()
                torch.save(self.ddpg.actor.state_dict(), path+'/actor.pth')
                torch.save(self.ddpg.critic.state_dict(), path+'/critic.pth')


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
