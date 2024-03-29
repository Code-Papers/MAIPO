import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import subprocess

import common.tf_util as U
from trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from smac.env import StarCraft2Env

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="8m", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=120000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
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
    model = mlp_model
    trainer = MADDPGAgentTrainer
    num_agents = len(obs_shape_n)
    for i in range(num_agents):
        trainers.append(trainer(
            "agent_%d" % i, num_agents, model, obs_shape_n[i], state_shape, num_actions, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = StarCraft2Env(map_name=arglist.scenario)
        # Create agent trainers
        env_info = env.get_env_info()
        max_episode_len = env_info['episode_limit']
        num_agents = env_info['n_agents']
        num_actions = env_info['n_actions']
        obs_shape_n = [(env_info['obs_shape'],) for i in range(num_agents)]
        state_shape = (env_info['state_shape'], )
        trainers = get_trainers(obs_shape_n, state_shape, num_actions, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        final_ep_rewards = []  # sum of rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        env.reset()
        obs_n = env.get_obs()
        state = env.get_state()
        # print(len(trainers))
        # print("obs_n:")
        # print(obs_n)
        episode_step = 0
        train_step = 0
        t_start = time.time()

        # command = "mkdir " + arglist.save_dir + arglist.exp_name
        # error_message = subprocess.call(command, shell=True)
        # print(error_message)
        # save_dir = arglist.save_dir + arglist.exp_name + "/"

        print('Starting iterations...')
        while True:
            # get action
            action_n = []
            avail_actions = []
            for i, agent in enumerate(trainers):
                avail_action = env.get_avail_agent_actions(i)
                avail_actions.append(avail_action)
                action_scope = np.nonzero(avail_action)[0]
                action = agent.action([obs_n[i]], [avail_action])[0]
                action_n.append(min(max(action, action_scope[0]), action_scope[-1]))
            # environment step
            # print("action_n:", type(action_n))
            # print(action_n)
            rew, done, info = env.step(action_n)
            new_obs_n = env.get_obs()
            new_state = env.get_state()
            # print("new_obs_n:", type(new_obs_n))
            # print(new_obs_n)
            # print("rew_n:", type(rew_n))
            # print(rew_n)
            # print("done_n:", type(done_n))
            # print(done_n)
            episode_step += 1
            terminal = (episode_step >= max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew, new_obs_n[i], avail_actions[i], done, state, new_state)
            obs_n = new_obs_n
            state = new_state

            episode_rewards[-1] += rew

            if done or terminal:
                env.reset()
                obs_n = env.get_obs()
                state = env.get_state()
                episode_step = 0
                episode_rewards.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                # print('loss')
                # print(loss)

            # save model, display training output
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                # U.save_state(save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
