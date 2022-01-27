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

def Agent_train(obs_shape_n, act_space_n, p_index, p_func, v_func, optimizer, grad_norm_clipping=0.5, num_units=64, scope="trainer", reuse=None):
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
        PD = act_pdtype.param_placeholder([None], name="policydistribution"+str(p_index))
        X = U.BatchInput(obs_shape_n[p_index], name="Agentobservation"+str(p_index)).get()
        ADV = tf.placeholder(tf.float32, [None], name="advantage"+str(p_index))
        R = tf.placeholder(tf.float32, [None], name="return"+str(p_index))
        OLDLOGPAC = tf.placeholder(tf.float32, [None], name="oldlogpac"+str(p_index))
        OLDVPRED = tf.placeholder(tf.float32, [None], name="oldvalue"+str(p_index))
        OLDENTROPY = tf.placeholder(tf.float32, [None], name="policyentropy"+str(p_index))
        VCLIPRANGE = tf.placeholder(tf.float32, [], name="valuecliprange"+str(p_index))
        MAXCLIPRANGE = tf.placeholder(tf.float32, [], name="policycliprangemax"+str(p_index))
        MINCLIPRANGE = tf.placeholder(tf.float32, [], name="policycliprangemax"+str(p_index))
        EXPLORE = tf.placeholder(tf.float32, [], name="policyentropy"+str(p_index))
        PENALTY = tf.placeholder(tf.float32, [], name="penaltycoefficient"+str(p_index))

        # policy (actor): from X to action distribution
        p = p_func(X, int(act_pdtype.param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype.pdfromflat(p)

        act_sample = act_pd.sample()
        
        # the entropy of pi(s)
        act_entropy = act_pd.entropy()
        kl_dis = act_pd.kl(PD)
        # kl_dis_cliped = tf.clip_by_value(kl_dis, MINCLIPRANGE, MAXCLIPRANGE)
        kl_dis_low = tf.maximum(kl_dis, MAXCLIPRANGE)
        kl_dis_up  = tf.minimum(kl_dis, MINCLIPRANGE)
        kl_low_exp = kl_dis_low / 1e-7
        kl_up_exp  = kl_dis_up  / 1e-7
        
        # the KL distance about X between the new policy and the old policy
        # kl_dis_maxcliped = tf.floor(kl_dis_cliped / MAXCLIPRANGE) * 100.0
        # kl_dis_mincliped = tf.floor((MAXCLIPRANGE-kl_dis_cliped)/(MAXCLIPRANGE - MINCLIPRANGE)) * 100.0
        # kl_dis_map = kl_dis_maxcliped + kl_dis_mincliped

        # log(probability) of A under current action distribution
        log_pac = act_pd.logp(A)

        # calculate ratio (pi(A|S) current policy / pi_old(A|S) old policy)
        ratio = tf.exp(log_pac - OLDLOGPAC)
        # entropy_ratio = act_entropy / OLDENTROPY 
        
        pg_loss1 = tf.reduce_mean(-ADV * ratio)
        # pg_loss2 = tf.reduce_mean(-entropy_ratio)
        # pg_loss2 = tf.reduce_mean(-tf.clip_by_value(entropy_ratio, 0.9, 1.1))
        # pg_loss2 = tf.reduce_mean(-act_entropy)
        pg_loss3 = tf.reduce_mean(kl_low_exp)
        pg_loss4 = tf.reduce_mean(-kl_up_exp)
        # pg_loss = pg_loss1 + EXPLORE * pg_loss2 + PENALTY * pg_loss3
        pg_loss = pg_loss1 + pg_loss3 + pg_loss4
        # pg_loss = (1.0 - EXPLORE) * (pg_loss1 + PENALTY * pg_loss3) + EXPLORE * pg_loss2

        # defining Loss = - J is equivalent to max J
        # pg_losses1 = -ADV * ratio

        # pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - PCLIPRANGE, 1.0 + PCLIPRANGE)
        
        # final PG loss
        # pg_loss3 = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        
        # pg_loss4 = tf.reduce_mean(tf.clip_by_value(entropy_ratio, 1.0 - HCLIPRANGE, 1.0 + HCLIPRANGE))
        # pg_loss4 = tf.reduce_mean(act_entropy)

        # pg_loss = pg_loss3 + EXPLORE * pg_loss4
        # pg_loss = (1.0 - EXPLORE) * pg_loss3 + EXPLORE * pg_loss4 
            
        p_optimize_expr = U.minimize_and_clip(optimizer, pg_loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        p_train = U.function(inputs=[X] + [A] + [OLDLOGPAC] + [ADV] + [OLDENTROPY] + [EXPLORE] + [PD] + [PENALTY] + [MAXCLIPRANGE] + [MINCLIPRANGE], outputs=pg_loss, updates=[p_optimize_expr])
        log_px = U.function(inputs=[X] + [A], outputs=log_pac)
        act = U.function(inputs=[X], outputs=act_sample)
        ent = U.function(inputs=[X], outputs=act_entropy)
        dis_para = U.function(inputs=[X], outputs=p)
        distance = U.function(inputs=[X] + [PD], outputs=kl_dis)

        # value (critic): from S to V(S), S is the state of environment
        S = tf.concat(obs_ph_n, 1)
        vpred = v_func(S, 1, scope="v_func", num_units=num_units)[:,0]
        v_func_vars = U.scope_vars(U.absolute_scope_name("v_func"))
        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - VCLIPRANGE, VCLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        # vf_loss = tf.reduce_mean(vf_losses1)
        v_optimize_expr = U.minimize_and_clip(optimizer, vf_loss, v_func_vars, grad_norm_clipping)

        v_train = U.function(inputs=obs_ph_n + [OLDVPRED] + [R] + [VCLIPRANGE], outputs=vf_loss, updates=[v_optimize_expr])
        vs      = U.function(inputs=obs_ph_n, outputs=vpred)

        return act, p_train, log_px, ent, v_train, vs, dis_para, distance

class MAPOAgentTrainer(object):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args):
        self.args = args
        self.actor, self.p_train, self.log_px, self.ent, self.v_train, self.vs, self.dis_para, self.distance = Agent_train(
            scope=name, 
            obs_shape_n=obs_shape_n, 
            act_space_n=act_space_n, 
            p_index=agent_index, 
            p_func=model, 
            v_func=model, 
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5, 
            num_units=args.num_units)
        self.agent_index = agent_index
    # optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),

    def log_p(self, obs, act):
        return self.log_px(*([obs] + [act]))

    def value_s(self, state):
        return self.vs(*(state))
    
    def action(self, obs):
        return self.actor(*([obs]))
    
    def ent_s(self, obs):
        return self.ent(*([obs]))
    
    def policy_para(self, obs):
        return self.dis_para(*([obs]))
    
    def policy_distance(self, obs, pd):
        return self.distance(*([obs] + [pd]))
    
    def step(self, state):
        act  = self.action(state[self.agent_index])
        logp = self.log_p(state[self.agent_index], act)
        val  = self.value_s(state)
        ent  = self.ent_s(state[self.agent_index])
        pd   = self.policy_para(state[self.agent_index])
        return act, logp, val, ent, pd
    
    def experience(self, raw_samples):

        # deliver the samples
        self.obs_t      = raw_samples[0]
        self.act        = raw_samples[1]
        self.ret        = raw_samples[2]
        self.logp       = raw_samples[3]
        self.adv        = raw_samples[4]
        self.val_st     = raw_samples[5]
        self.ent_st     = raw_samples[6]
        self.policy_dis = raw_samples[7]

        # other_adv = np.delete(self.adv, self.agent_index, axis=1)
        # other_adv = np.sum(other_adv, axis=1)
        self.adv = self.adv[:, self.agent_index]

    def update(self, num_batch, state, v_clip_range, explore, penalty, max_clip_range, min_clip_range):

        p_loss, v_loss, distance = [], [], []

        # compute the number of batch used to train
        num_samples = len(self.ret)

        # divide samples into n batches to train
        for start in range(0, num_samples, num_batch):
            end = start + num_batch
            if (end <= num_samples):
                state_ep = [obsi[start:end] for obsi in state]
                obs = self.obs_t[start:end]
                act = self.act[start:end]
                logp = self.logp[start:end]
                adv = self.adv[start:end]
                ret = self.ret[start:end]
                val = self.val_st[start:end]
                ent = self.ent_st[start:end]
                pol_dis = self.policy_dis[start:end]
                
                p_loss.append(self.p_train(*([obs] + [act] + [logp] + [adv] + [ent] + [explore] + [pol_dis] + [penalty] + [max_clip_range] + [min_clip_range])))
                v_loss.append(self.v_train(*(state_ep + [val] + [ret] + [v_clip_range])))
                distance.append(self.policy_distance(obs, pol_dis))
                
        distance = np.mean(distance)
        return [p_loss, v_loss, distance]