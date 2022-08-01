import numpy as np
import random
import tensorflow as tf
import common.tf_util as U

from common.distributions import make_pdtype
# from maddpg import AgentTrainer
from trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(obs_shape, state_shape, num_agents, num_actions, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(num_actions) for _ in range(num_agents)]

        # set up placeholders
        state_ph = U.BatchInput(state_shape, name="state").get()
        obs_ph = U.BatchInput(obs_shape, name="obs"+str(p_index)).get()
        ava_act_ph = tf.placeholder(tf.float32, [None, num_actions], name="available_action"+str(p_index))
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(num_agents)]

        p_input = obs_ph

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        real_p = tf.multiply(p, ava_act_ph) - 1e10 * (1-ava_act_ph)
        act_pd = act_pdtype_n[p_index].pdfromflat(real_p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_sample
        team_action = tf.transpose(act_input_n)
        q_input = tf.concat([state_ph, team_action], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[state_ph] + [obs_ph] + [ava_act_ph] + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph] + [ava_act_ph], outputs=act_sample)
        p_values = U.function([obs_ph] + [ava_act_ph], real_p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        real_target_p = tf.multiply(target_p, ava_act_ph) - 1e10 * (1-ava_act_ph)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(real_target_p).sample()
        target_act = U.function(inputs=[obs_ph]+[ava_act_ph], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(state_shape, num_agents, num_actions, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(num_actions) for _ in range(num_agents)]

        # set up placeholders
        state_ph = U.BatchInput(state_shape, name="state").get()
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(num_agents)]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        team_action = tf.transpose(act_ph_n)
        q_input = tf.concat([state_ph, team_action], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[state_ph] + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function([state_ph] + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function([state_ph] + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(object):
    def __init__(self, name, num_agents, model, obs_shape, state_shape, num_actions, agent_index, args, local_q_func=False):
        self.name = name
        self.n = num_agents
        self.agent_index = agent_index
        self.args = args

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            state_shape = state_shape,
            num_agents=num_agents,
            num_actions=num_actions,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            obs_shape=obs_shape,
            state_shape=state_shape,
            num_agents = self.n,
            num_actions=num_actions,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs, ava_action):
        return self.act(*([obs] + [ava_action]))

    def experience(self, obs, act, rew, new_obs, ava_action, done, sta, new_sta):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, ava_action, float(done), sta, new_sta)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        # if len(self.replay_buffer) < 4: # replay buffer is not large enough
        #     return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # self.replay_sample_index = self.replay_buffer.make_index(4)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        ava_action_n = []
        for i in range(self.n):
            obs, act, rew, obs_next, ava, done, _, _ = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            ava_action_n.append(ava)
        obs, act, rew, obs_next, ava_act, done, sta, sta_next = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](*([obs_next_n[i]] + [ava_action_n[i]])) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*([sta] + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*([sta] + act_n + [target_q]))
        # train p network
        p_loss = self.p_train(*([sta] + [obs] + [ava_act] + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
