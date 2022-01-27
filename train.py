import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import common.tf_util as U
from trainer.maipo import MAPOAgentTrainer
import tensorflow.contrib.layers as layers
from trainer.runner import Runner
import subprocess

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=120000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="mapo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="mapo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="lambda return")
    parser.add_argument("--trust-region", type=float, default=1e-6, help="clip range in MAPO")
    parser.add_argument("--low-distance", type=float, default=1e-7, help="low bound of policy distance")
    parser.add_argument("--nsteps", type=int, default=512, help="the number of steps in one epoch")
    parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--explore-rate", type=int, default=50, help="the frequency of changing to the exploration objective")
    parser.add_argument("--hclip-range", type=float, default=0.5, help="clip range of the exploration objective")
    parser.add_argument("--penalty", type=float, default=1.0, help="the penalty coefficient of KL distance")
    parser.add_argument("--explore", type=float, default=1.0, help="the coefficient of entropy")
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

def cyclical(counter):
    cycle = 1 + counter // (2 * 4096)
    x = abs(counter / 4096 - 2 * cycle + 1)
    y = 1e-5 + (1.0 - 1e-5) * max(0, (1-x))
    return y

def lin(counter):
    y = 1 - 0.8 * counter/120000.0
    return y

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name):
    if scenario_name == "simple_reference":
        from pettingzoo.mpe import simple_reference_v2
        env = simple_reference_v2.parallel_env()
    elif scenario_name == "simple_speaker_listener":
        from pettingzoo.mpe import simple_speaker_listener_v3
        env = simple_speaker_listener_v3.parallel_env()
    elif scenario_name == "simple_spread":
        from pettingzoo.mpe import simple_spread_v2
        env = simple_spread_v2.parallel_env()
    else:
        raise Exception("cannot find the inputted environment")
    return env

def get_trainers(env, num_agents, num_adversaries, obs_shape_n, action_space, arglist):
    trainers = []
    model = mlp_model
    trainer = MAPOAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_space, i, arglist))
    for i in range(num_adversaries, num_agents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_space, i, arglist))
    return trainers

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario)
        obs_n = env.reset()
        num_agents = env.num_agents
        # Create agent trainers
        obs_shape_n = [env.observation_space(agent).shape for agent in env.agents]
        action_space = [env.action_space(agent) for agent in env.agents]
        num_adversaries = min(num_agents, arglist.num_adversaries)
        trainers = get_trainers(env, num_agents, num_adversaries, obs_shape_n,  action_space, arglist)
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
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        saver = tf.train.Saver()
        train_step = 0
        t_start = time.time()
        runner = Runner(env, obs_n, arglist)
        saving_step = arglist.save_rate

        command = "mkdir " + arglist.save_dir + arglist.exp_name
        error_message = subprocess.call(command, shell=True)
        print(error_message)
        save_dir = arglist.save_dir + arglist.exp_name + "/"
        # stand_clip_range = arglist.clip_range ** (1 / (num_agents + 1))
        explore = [0.0 for i in range(num_agents)]

        print('Starting iterations...')
        while True:
            # collect experience
            samples, episode_rewards, agent_rewards = runner.run(trainers)
            agents_obs = []
            v_clip_range = 0.2

            for i, agent in enumerate(trainers):
                agent_obs_t   = np.array([np.array(obs) for obs in samples[0][:, i]])
                agents_obs.append(agent_obs_t)
                agent_act     = np.array([np.array(act) for act in samples[1][:, i]])
                agent_ret     = samples[2][:, i]
                agent_logp    = samples[3][:, i]
                agents_adv    = samples[4]
                agent_val     = samples[5][:, i]
                agent_ent     = samples[6][:, i]
                agent_pd      = np.array([np.array(act) for act in samples[7][:, i]])

                agent_samples = [agent_obs_t, agent_act, agent_ret, agent_logp, agents_adv, agent_val, agent_ent, agent_pd]
                agent.experience(agent_samples)
             
            # explore = lin(len(episode_rewards))
            penalty = arglist.penalty
            distance = []
            # max_clip_range = 0.01 * lin(len(episode_rewards))
            max_clip_range = arglist.trust_region ** (1/num_agents)
            min_clip_range = arglist.low_distance
            for i, agent in enumerate(trainers):
                loss = agent.update(arglist.batch_size, agents_obs, v_clip_range, explore[i], penalty, max_clip_range, min_clip_range)
                distance.append(loss[2])
                # print("loss")
                # print(loss[2]) 
                     
            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # save model, display training output
            # if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            if (len(episode_rewards) / saving_step > 1):
                saving_step += arglist.save_rate
                U.save_state(save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    print("the average distance: {}, explorer: {}".format(distance, explore))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                U.save_state(save_dir, saver=saver)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(agent_rewards, fp)

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))

                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
