import numpy as np

class Runner(object):
    """
    We use this object to generate samples and record the episode's reward of MAPO
    __init__
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, env, obs_0, args):
        self.env      = env
        # self.trainers = trainers
        self.nsteps   = args.nsteps
        self.gamma    = args.gamma
        self.lam      = args.lam
        self.obs0     = obs_0
        self.episode_rewards = [0]
        self.agent_rewards = [[0.0] for _ in env.agents]
        self.episode_step = 0
        self.arg = args 

    def run(self, trainers):
        mb_state, mb_value, mb_action, mb_logp, mb_ent, mb_reward, mb_state1, mb_pd, mb_done, env_done, mb_info = [], [], [], [], [], [], [], [], [], [], []
        obs_n_dic = self.obs0
        for _ in range(self.nsteps):
            action_n, logp_n, value_n, ent_n, pd_n = [], [], [], [], []
            action_n_int = {}
            obs = [[obs_n_dic[agent]] for agent in self.env.agents]
            for i, agent in enumerate(trainers):
                agent_act, agent_logp, agent_val, agent_ent, agent_pd = agent.step(obs)
                action_n.append(agent_act[0])
                action_n_int[self.env.agents[i]] = int(agent_act[0][0])
                logp_n.append(agent_logp[0])
                value_n.append(agent_val[0])
                ent_n.append(agent_ent[0])
                pd_n.append(agent_pd[0])
            # print("actions")
            # print(action_n)
            # print("logp_n")
            # print(logp_n)
            # print("value_n")
            # print(value_n)
            # print("ent_n")
            # print(ent_n)
            # print(pd_n)
            
            obs = list(obs_n_dic.values())
            mb_state.append(obs)
            mb_value.append(value_n)
            mb_action.append(action_n)
            mb_logp.append(logp_n)
            mb_ent.append(ent_n)
            mb_pd.append(pd_n)
            obs_n_dic, rew_n_dic, done_n_dic, info_n_dic = self.env.step(action_n_int)
            # logp, adv, ret
            obs_n, rew_n, done_n, info_n = list(obs_n_dic.values()), list(rew_n_dic.values()), list(done_n_dic.values()), list(info_n_dic.values())
            # print("obs_n")
            # print(obs_n)
            # print("np obs_n")
            # print(np.array(obs_n))
            # print("rew_n")
            # print(rew_n)
            # print("done_n")
            # print(done_n)
            # logp, adv, ret
            mb_state1.append(obs_n)
            mb_reward.append(rew_n)
            mb_done.append(done_n)
            mb_info.append(info_n)
            self.episode_step += 1
            done = all(done_n)
            env_done.append(done)
            terminal = (self.episode_step >= self.arg.max_episode_len)
            for i, rew in enumerate(rew_n):
                self.episode_rewards[-1] += rew
                self.agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n_dic = self.env.reset()
                self.episode_step = 0
                self.episode_rewards.append(0)
                for a in self.agent_rewards:
                    a.append(0)
        self.obs0 = obs_n_dic

        last_values = []
        obs = [[obs_n[i]] for i in range(len(obs_n))]
        for i, agent in zip(range(len(obs_n)), trainers):
            if done_n[i]:
                last_values.append(0)
            else:
                last_values.append(agent.value_s(obs)[0])
        
        mb_state  = np.array(mb_state)
        mb_value  = np.array(mb_value)
        mb_action = np.array(mb_action)
        mb_logp   = np.array(mb_logp)
        mb_ent    = np.array(mb_ent)
        mb_pd     = np.array(mb_pd)
        mb_reward = np.array(mb_reward)
        mb_state1 = np.array(mb_state1)
        mb_done   = np.array(mb_done)
        env_done  = np.array(env_done)

        mb_return = np.zeros_like(mb_reward)
        mb_adv    = np.zeros_like(mb_reward)

        for t in reversed(range(self.nsteps)):
            for i in range(len(trainers)):
                nextnonterminal = 1.0 - mb_done[t, i]
                if t == self.nsteps - 1:
                    nextvalues = last_values[i]
                    delta = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues - mb_value[t, i]
                    mb_adv[t, i] = delta
                else:
                    nextvalues = mb_value[t+1, i]
                    delta = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues - mb_value[t, i]
                    mb_adv[t, i] = delta + self.gamma * self.lam * nextnonterminal * mb_adv[t+1, i]
        mb_return = mb_adv + mb_value

        # samples = [mb_state, mb_action, mb_reward, mb_state1, mb_done, env_done]
        samples = [mb_state, mb_action, mb_return, mb_logp, mb_adv, mb_value, mb_ent, mb_pd]
        return samples, self.episode_rewards, self.agent_rewards
