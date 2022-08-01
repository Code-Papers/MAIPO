import numpy as np
import random
from numpy.core.fromnumeric import clip
from numpy.lib.shape_base import vsplit
import tensorflow as tf
from tensorflow.python.ops.gen_linalg_ops import self_adjoint_eig_eager_fallback
import common.tf_util as U

from common.distributions import make_pdtype
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.gen_math_ops import log, tanh

def Agent_train(obs_shape_n, state_shape, act_space_n, p_index, p_func, v_func, optimizer, max_policy_dis, min_policy_dis, clip_range=0.2, grad_norm_clipping=0.5, num_units=64, scope="trainer", reuse=None):
    '''
    List: obs_shape_n,    -the obsevation shape of each agent e.g. obs_shape_n[i] = 1, it shows the obseration shape of agent i is 1
    List: act_space_n,    -the action space of each agent, including the action shape, e.g. act_space_n[i] = [action1_limit, action2_limit, ...], where action1_limit = [max, min]
    Int: p_index,         -the ID of current agent
    Fun: p_func, v_func,  -the funcion of policy and value networks 
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # creat distributions
        act_pdtype = make_pdtype(act_space_n)
        # set up placeholders
        A = act_pdtype.sample_placeholder([None], name="action"+str(p_index))
        AVA_A = tf.placeholder(tf.float32, [None, act_space_n], name="available_action"+str(p_index))
        PD = act_pdtype.param_placeholder([None], name="policydistribution"+str(p_index))
        X = U.BatchInput(obs_shape_n[p_index], name="Agentobservation"+str(p_index)).get()
        S = U.BatchInput(state_shape, name="groupstate").get()
        ADV = tf.placeholder(tf.float32, [None], name="advantage"+str(p_index))
        R = tf.placeholder(tf.float32, [None], name="return"+str(p_index))
        OLDLOGPAC = tf.placeholder(tf.float32, [None], name="oldlogpac"+str(p_index))
        OLDVPRED = tf.placeholder(tf.float32, [None], name="oldvalue"+str(p_index))

        # policy (actor): from X to action distribution
        p = p_func(X, int(act_pdtype.param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        real_p = tf.multiply(p, AVA_A) - 1e10 * (1-AVA_A)
        act_pd = act_pdtype.pdfromflat(real_p)

        act_sample = act_pd.sample()
        
        kl_dis = act_pd.kl(PD)
        kl_dis_low = tf.maximum(kl_dis, max_policy_dis) / min_policy_dis
        kl_dis_up  = tf.minimum(kl_dis, min_policy_dis) / min_policy_dis

        # log(probability) of A under current action distribution
        log_pac = act_pd.logp(A)

        # calculate ratio (pi(A|S) current policy / pi_old(A|S) old policy)
        ratio = tf.exp(log_pac - OLDLOGPAC)

        # defining Loss = - J is equivalent to max J
        # pg_losses = -ADV * ratio

        # pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range) 
        
        # final PG loss
        # pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # pg_loss = tf.reduce_mean(pg_losses2)
        pg_loss1 = tf.reduce_mean(-ADV * ratio)
        pg_loss2 = tf.reduce_mean(kl_dis_low)
        pg_loss3 = tf.reduce_mean(-kl_dis_up)
        pg_loss = pg_loss1 + pg_loss2 + pg_loss3
        p_optimize_expr = U.minimize_and_clip(optimizer, pg_loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        p_train = U.function(inputs=[X] + [A] + [OLDLOGPAC] + [ADV] + [AVA_A] + [PD], outputs=pg_loss, updates=[p_optimize_expr])
        log_px = U.function(inputs=[X] + [A] + [AVA_A], outputs=log_pac)
        act = U.function(inputs=[X] + [AVA_A], outputs=act_sample)
        distribution = U.function(inputs=[X] + [AVA_A], outputs=real_p)
        distance = U.function(inputs=[X] + [AVA_A] + [PD], outputs=kl_dis)

        # value (critic): from X to V(X), S is the state of environment
        vpred = v_func(S, 1, scope="v_func", num_units=num_units)[:,0]
        v_func_vars = U.scope_vars(U.absolute_scope_name("v_func"))
        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - clip_range, clip_range)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        # vf_loss = tf.reduce_mean(vf_losses1)
        v_optimize_expr = U.minimize_and_clip(optimizer, vf_loss, v_func_vars, grad_norm_clipping)

        v_train = U.function(inputs=[S] + [OLDVPRED] + [R], outputs=vf_loss, updates=[v_optimize_expr])
        vs      = U.function(inputs=[S], outputs=vpred)

        return act, p_train, log_px, v_train, vs, distribution, distance

class MAPOAgentTrainer(object):
    def __init__(self, name, model, obs_shape_n, state_shape, num_actions, agent_index, clip_range, max_policy_dis, min_policy_dis, args):
        self.args = args
        self.actor, self.p_train, self.log_px, self.v_train, self.vs, self.pol_distribution, self.pol_distance = Agent_train( 
            obs_shape_n=obs_shape_n, 
            state_shape=state_shape,
            act_space_n=num_actions, 
            p_index=agent_index, 
            p_func=model, 
            v_func=model, 
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            clip_range=clip_range,
            max_policy_dis=max_policy_dis, 
            min_policy_dis=min_policy_dis, 
            grad_norm_clipping=0.5, 
            num_units=args.num_units,
            scope=name) 
        self.agent_index = agent_index
    # optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),

    def log_p(self, obs, act, ava_action):
        return self.log_px(*([obs] + [act] + [ava_action]))

    def value_s(self, state):
        return self.vs(*([state]))
    
    def action(self, obs, ava_action):
        return self.actor(*([obs] + [ava_action]))
    
    def policy_para(self, obs, ava_action):
        return self.pol_distribution(*([obs] + [ava_action]))
    
    def policy_distance(self, obs, ava_action, pd):
        return self.pol_distance(*([obs] + [ava_action] + [pd]))
    
    def step(self, obs, state, ava_action):
        act  = self.action([obs], [ava_action])
        logp = self.log_p([obs], act, [ava_action])
        val  = self.value_s([state])
        pd   = self.policy_para([obs], [ava_action])
        return act, logp, val, pd
    
    def experience(self, raw_samples):

         # deliver the samples
        self.obs_t      = raw_samples[0]
        self.act        = raw_samples[1]
        self.ret        = raw_samples[2]
        self.logp       = raw_samples[3]
        self.adv        = raw_samples[4]
        self.val_st     = raw_samples[5]
        self.ava_action = raw_samples[6]
        self.pd         = raw_samples[7]
        self.state      = raw_samples[8]

    def update(self, num_batch):

        p_loss, v_loss, distance = [], [], []

        # compute the number of batch used to train
        num_samples = len(self.ret)

        # divide samples into n batches to train
        for start in range(0, num_samples, num_batch):
            end = start + num_batch
            if (end <= num_samples):
                obs = self.obs_t[start:end]
                act = self.act[start:end]
                logp = self.logp[start:end]
                adv = self.adv[start:end]
                ret = self.ret[start:end]
                val = self.val_st[start:end]
                ava_action = self.ava_action[start:end]
                pd  = self.pd[start:end]
                sta = self.state[start:end]
                p_loss.append(self.p_train(*([obs] + [act] + [logp] + [adv] + [ava_action] + [pd])))
                v_loss.append(self.v_train(*([sta] + [val] + [ret])))
                distance.append(self.policy_distance(obs, ava_action, pd))
            else:
                print("ERROR: Index exceeds the number of training samples")
        distance = np.mean(distance)   
        return [p_loss, v_loss, distance]