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
    

    def run(self, trainers, group_trainer):
        mb_state, mb_value, mb_gru_value, mb_action, mb_logp, mb_reward, mb_state1, mb_done, env_done, mb_info = [], [], [], [], [], [], [], [], [],  []
        obs_n_dic = self.obs0
        for _ in range(self.nsteps):
            action_n, logp_n, value_n = [], [], []
            action_n_int = {}
            obs = [[obs_n_dic[agent]] for agent in self.env.agents]
            group_value = group_trainer.value_s(obs)[0]
            for i, agent in enumerate(trainers):
                agent_act, agent_logp, agent_val = agent.step(obs)
                action_n.append(agent_act[0])
                action_n_int[self.env.agents[i]] = int(agent_act[0][0])
                logp_n.append(agent_logp[0])
                value_n.append(agent_val[0])
            # print("actions")
            # print(action_n)
            # print("logp_n")
            # print(logp_n)
            # print("value_n")
            # print(value_n)
            # print("action_n_int")
            # print(action_n_int)
            
            obs = [obs_n_dic[agent] for agent in self.env.agents]
            mb_state.append(obs)
            mb_gru_value.append(group_value)
            mb_value.append(value_n)
            mb_action.append(action_n)
            mb_logp.append(logp_n)
            obs_n_dic, rew_n_dic, done_n_dic, info_n_dic = self.env.step(action_n_int)
            # logp, adv, ret
            obs_n, rew_n, done_n, info_n = list(obs_n_dic.values()), list(rew_n_dic.values()), list(done_n_dic.values()), list(info_n_dic.values())
            
            # print("obs_n")
            # print(obs_n)
            # print("rew_n")
            # print(rew_n)
            # print("done_n")
            # print(done_n)
            
            mb_state1.append(obs_n)
            mb_reward.append(rew_n)
            mb_done.append(done_n)
            mb_info.append(info_n)
            self.episode_step += 1
            done = all(done_n)
            env_done.append(done)
            terminal = (self.episode_step >= self.arg.max_episode_len)
            for i, rew in enumerate(rew_n):
                if (done_n[i] == 0):
                    self.episode_rewards[-1] += rew
                    self.agent_rewards[i][-1] += rew

            if done or terminal:
            # if done:
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
        last_gru_value = 0 if done else group_trainer.value_s(obs)[0]
        mb_state  = np.array(mb_state)
        mb_gru_value = np.array(mb_gru_value)
        mb_value  = np.array(mb_value)
        mb_action = np.array(mb_action)
        mb_logp   = np.array(mb_logp)
        mb_reward = np.array(mb_reward)
        mb_gru_reward = np.sum(mb_reward, axis=1)
        mb_state1 = np.array(mb_state1)
        mb_done   = np.array(mb_done)
        env_done  = np.array(env_done)

        # mb_return = np.zeros_like(mb_reward)
        # mb_adv    = np.zeros_like(mb_reward)
        mb_return = np.zeros_like(mb_gru_value)
        mb_adv    = np.zeros_like(mb_gru_value)
        
        for t in reversed(range(self.nsteps)):
            nextnonterminal = 1.0 - env_done[t]
            if t == self.nsteps - 1:
                nextvalue = last_gru_value
                delta = mb_gru_reward[t] + self.gamma * nextnonterminal * nextvalue - mb_gru_value[t]
                mb_adv[t] = delta
            else:
                nextvalue = mb_gru_value[t+1]
                delta = mb_gru_reward[t] + self.gamma * nextnonterminal * nextvalue - mb_gru_value[t]
                mb_adv[t] = delta + self.gamma * self.lam * nextnonterminal * mb_adv[t+1]
        mb_return = mb_adv + mb_gru_value

        # for t in reversed(range(self.nsteps)):
        #     for i in range(len(trainers)):
        #         nextnonterminal = 1.0 - mb_done[t, i]
        #         if t == self.nsteps - 1:
        #             nextvalues = last_values[i]
        #             delta = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues - mb_value[t, i]
        #             mb_adv[t, i] = delta
        #             # mb_return[t, i] = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues
        #         else:
        #             nextvalues = mb_value[t+1, i]
        #             delta = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues - mb_value[t, i]
        #             mb_adv[t, i] = delta + self.gamma * self.lam * nextnonterminal * mb_adv[t+1, i]
        #             # mb_return[t, i] = mb_reward[t, i] + self.gamma * nextnonterminal * mb_return[t+1, i]
        #             # mb_return[t, i] = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues 
        # mb_return = mb_adv + mb_value

        # samples = [mb_state, mb_action, mb_reward, mb_state1, mb_done, env_done]
        samples = [mb_state, mb_action, mb_return, mb_logp, mb_adv, mb_value, mb_gru_value]
        return samples, self.episode_rewards, self.agent_rewards
