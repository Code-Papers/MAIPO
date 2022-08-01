import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import common.tf_util as U
from trainer.happo import MAPOAgentTrainer, MAPOAgentsTrainer
import tensorflow.contrib.layers as layers
from trainer.runner import Runner
import subprocess
from smac.env import StarCraft2Env

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for SMAC")
    # Environment
    parser.add_argument("--scenario", type=str, default="8m", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=120, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=120000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="mapo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="mapo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="lambda return")
    parser.add_argument("--clip-range", type=float, default=0.1, help="clip range in MAPO")
    parser.add_argument("--nsteps", type=int, default=512, help="the number of steps in one epoch")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="8m", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(obs_shape_n, state_shape, num_actions, arglist):
    trainers = []
    num_agents = len(obs_shape_n)
    model = mlp_model
    agent_trainer = MAPOAgentTrainer
    clip_range = arglist.clip_range ** (1/num_agents)
    for i in range(num_agents):
        trainers.append(agent_trainer(
            "agent_%d" % i, model, obs_shape_n, num_actions, i, clip_range, arglist))
    group_trainer = MAPOAgentsTrainer("group_trainer", model, state_shape, arglist.clip_range, arglist)
    return trainers, group_trainer

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = StarCraft2Env(map_name=arglist.scenario)
        env_info = env.get_env_info()
        max_episode_len = env_info['episode_limit']
        num_agents = env_info['n_agents']
        num_actions = env_info['n_actions']
        # Create agent trainers
        obs_shape_n = [(env_info['obs_shape'],) for i in range(num_agents)]
        state_shape = (env_info['state_shape'], )
        trainers, group_trainer = get_trainers(obs_shape_n, state_shape, num_actions, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        # episode_rewards = [0.0]  # sum of rewards for all agents
        # agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        # final_ep_rewards = []  # sum of rewards for training curve
        # agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        # print(len(trainers))
        # print("obs_n:")
        # print(type(obs_n))
        # print(type(obs_n[0]))
        # print(obs_n[1])
        # print("obs_shape:")
        # print(obs_n[0])
        # print("obs_space:")
        # print(env.observation_space[0].shape)
        # print("act_space:")
        # print(env.action_space)
        # episode_step = 0
        train_step = 0
        t_start = time.time()
        runner = Runner(env, max_episode_len, arglist)
        saving_step = arglist.save_rate

        command = "mkdir " + arglist.save_dir + arglist.exp_name
        error_message = subprocess.call(command, shell=True)
        print(error_message)
        save_dir = arglist.save_dir + arglist.exp_name + "/"

        print('Starting iterations...')
        while True: 
            # collect experience
            samples, episode_rewards = runner.run(trainers, group_trainer)
            group_return = samples[2]
            group_adv    = samples[4]
            group_value  = samples[5]
            group_state  = samples[7]
            for i, agent in enumerate(trainers):
                agent_obs_t   = samples[0][:, i]
                agent_act     = samples[1][:, i]
                agent_logp    = samples[3][:, i]
                agent_ava_action = samples[6][:, i]

                # print("obs")
                # print(agent_obs_t)
                # print("act")
                # print(agent_act)
                # print("ret")
                # print(agent_ret)
                # print("logp")
                # print(agent_logp)
                # print("adv")
                # print(agents_adv)

                agent_samples = [agent_obs_t, agent_act, agent_logp, agent_ava_action]
                agent.experience(agent_samples)
                
            v_loss = group_trainer.update(arglist.batch_size, group_state, group_value, group_return)
            # print(v_loss)
            
            M_adv = group_adv 
            for agent in trainers:
                loss, M_adv = agent.update(arglist.batch_size, M_adv)
                # print(loss)

                # print("p_loss:")
                # print(loss[0])
                # print("v_loss:")
                # print(loss[1])
                # print('end')

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            # if arglist.benchmark:
            #     for i, info in enumerate(info_n):
            #         agent_info[-1][i].append(info_n['n'])
            #     if train_step > arglist.benchmark_iters and (done or terminal):
            #         file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
            #         print('Finished benchmarking, now saving...')
            #         with open(file_name, 'wb') as fp:
            #             pickle.dump(agent_info[:-1], fp)
            #         break
            #     continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue
            
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            # save model, display training output
            # if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            if (len(episode_rewards) / saving_step > 1):
                saving_step += arglist.save_rate
                U.save_state(save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                U.save_state(save_dir, saver=saver)
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))

                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
