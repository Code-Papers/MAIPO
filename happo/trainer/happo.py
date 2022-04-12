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

def Agent_train(obs_shape_n, act_space_n, p_index, p_func, v_func, optimizer, clip_range=0.2, grad_norm_clipping=0.5, num_units=64, scope="trainer", reuse=None):
    '''
    List: obs_shape_n,    -the obsevation shape of each agent e.g. obs_shape_n[i] = 1, it shows the obseration shape of agent i is 1
    List: act_space_n,    -the action space of each agent, including the action shape, e.g. act_space_n[i] = [action1_limit, action2_limit, ...], where action1_limit = [max, min]
    Int: p_index,         -the ID of current agent
    Fun: p_func, v_func,  -the funcion of policy and value networks 
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # creat distributions
        act_pdtype = make_pdtype(act_space_n[p_index])

        # set up placeholders
        A = act_pdtype.sample_placeholder([None], name="action"+str(p_index))
        obs_ph_n = [U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get() for i in range(len(obs_shape_n))]
        X = U.BatchInput(obs_shape_n[p_index], name="Agentobservation"+str(p_index)).get()
        ADV = tf.placeholder(tf.float32, [None], name="advantage"+str(p_index))
        R = tf.placeholder(tf.float32, [None], name="return"+str(p_index))
        OLDLOGPAC = tf.placeholder(tf.float32, [None], name="oldlogpac"+str(p_index))
        OLDVPRED = tf.placeholder(tf.float32, [None], name="oldvalue"+str(p_index))

        # policy (actor): from X to action distribution
        p = p_func(X, int(act_pdtype.param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype.pdfromflat(p)

        act_sample = act_pd.sample()

        # log(probability) of A under current action distribution
        log_pac = act_pd.logp(A)

        # calculate ratio (pi(A|S) current policy / pi_old(A|S) old policy)
        ratio = tf.exp(log_pac - OLDLOGPAC)

        # defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range) 
        
        # final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # pg_loss = tf.reduce_mean(pg_losses2)
        p_optimize_expr = U.minimize_and_clip(optimizer, pg_loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        p_train = U.function(inputs=[X] + [A] + [OLDLOGPAC] + [ADV], outputs=pg_loss, updates=[p_optimize_expr])
        log_px = U.function(inputs=[X] + [A], outputs=log_pac)
        act = U.function(inputs=[X], outputs=act_sample)

        # value (critic): from S to V(S), S is the state of environment
        S = tf.concat(obs_ph_n, 1)
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

        v_train = U.function(inputs=obs_ph_n + [OLDVPRED] + [R], outputs=vf_loss, updates=[v_optimize_expr])
        vs      = U.function(inputs=obs_ph_n, outputs=vpred)

        return act, p_train, log_px, v_train, vs

def Agents_train(obs_shape_n, v_func, optimizer, clip_range=0.2, grad_norm_clipping=0.5, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        R = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        obs_ph_n = []
        for i in range(len(obs_shape_n)):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        v_input = tf.concat(obs_ph_n, 1)
        vpred = v_func(v_input, 1, scope="v_func", num_units=num_units)[:, 0]
        v_func_vars = U.scope_vars(U.absolute_scope_name("v_func"))
        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - clip_range, clip_range)

        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        v_optimize_expr = U.minimize_and_clip(optimizer, vf_loss, v_func_vars, grad_norm_clipping)

        v_train = U.function(inputs=obs_ph_n + [OLDVPRED] + [R], outputs=vf_loss, updates=[v_optimize_expr])
        vs      = U.function(inputs=obs_ph_n, outputs=vpred)

    return v_train, vs

class HAPPOAgentTrainer(object):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, clip_range, args):
        self.args = args
        self.actor, self.p_train, self.log_px, self.v_train, self.vs = Agent_train(
            scope=name, 
            obs_shape_n=obs_shape_n, 
            act_space_n=act_space_n, 
            p_index=agent_index, 
            p_func=model, 
            v_func=model, 
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            clip_range=clip_range, 
            grad_norm_clipping=10, 
            num_units=args.num_units) 
        self.agent_index = agent_index
    # optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),

    def log_p(self, obs, act):
        return self.log_px(*([obs] + [act]))

    def value_s(self, state):
        return self.vs(*(state))
    
    def action(self, obs):
        return self.actor(*([obs]))
    
    def step(self, state):
        act  = self.action(state[self.agent_index])
        logp = self.log_p(state[self.agent_index], act)
        val  = self.value_s(state)
        return act, logp, val
    
    def experience(self, raw_samples):

        self.obs_t, self.obs_tp1, self.act, self.rew, self.ret, self.val_st, self.val_stp1, self.logp = [], [], [], [], [], [], [], []

         # deliver the samples
        self.obs_t      = raw_samples[0]
        self.act        = raw_samples[1]
        self.logp       = raw_samples[2]

    def update(self, num_batch, M_adv):

        p_loss, v_loss = [], []

        # compute the number of batch used to train
        num_samples = len(self.logp)

        # divide samples into n batches to train
        for start in range(0, num_samples, num_batch):
            end = start + num_batch
            if (end <= num_samples):
                obs = self.obs_t[start:end]
                act = self.act[start:end]
                logp = self.logp[start:end]
                adv = M_adv[start:end]
                p_loss.append(self.p_train(*([obs] + [act] + [logp] + [adv])))
            else:
                print("ERROR: Index exceeds the number of training samples")
        new_logp = self.log_p(self.obs_t, self.act) 
        ratio_log = np.array(new_logp) - np.array(self.logp)
        ratio    = np.exp(np.minimum(ratio_log, 1))
        M_adv    = ratio * np.array(M_adv)
        M_adv    = M_adv.tolist()
        return p_loss, M_adv
    
class HAPPOAgentsTrainer(object):
    def __init__(self, name, model, obs_shape_n, clip_range, args):
        self.args = args
        self.v_train, self.vs = Agents_train(
            scope=name, 
            obs_shape_n=obs_shape_n, 
            v_func=model, 
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr), 
            clip_range=clip_range,
            grad_norm_clipping=10,
            num_units=args.num_units)
    
    def value_s(self, obs):
        return self.vs(*(obs))
    
    def update(self, num_batch, state, val_st, ret):

        v_loss = []

        # compute the number of batch used to train
        num_samples = len(ret)

        # divide samples into n batches to train
        for start in range(0, num_samples, num_batch):
            end = start + num_batch
            if (end <= num_samples):
                obs_t = [state[i][start:end] for i in range(len(state))]
                val = val_st[start:end]
                ret = ret[start:end]
                v_loss.append(self.v_train(*(obs_t + [val] + [ret])))

        return v_loss