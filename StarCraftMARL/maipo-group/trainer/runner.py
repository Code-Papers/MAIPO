import numpy as np

class Runner(object):
    """
    We use this object to generate samples and record the episode's reward of MAPO
    __init__
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, env, max_episode_len, args):
        self.env      = env
        self.env.reset()
        # self.trainers = trainers
        self.nsteps   = args.nsteps
        self.gamma    = args.gamma
        self.lam      = args.lam
        self.obs0     = self.env.get_obs()
        self.state0   = env.get_state()
        self.episode_rewards = [0]
        self.episode_step = 0
        self.arg = args 
        self.max_episode_len = max_episode_len

    def run(self, trainers):
        mb_state, mb_value, mb_ava_actions, mb_action, mb_logp, mb_reward, mb_state1, mb_done, mb_pd, mb_gru_state, mb_info = [], [], [], [], [], [], [], [], [], [], []
        obs_n = self.obs0
        num_agents = len(trainers)
        gru_state = self.state0
        for _ in range(self.nsteps):
            action_n, ava_actions, logp_n, value_n, pd_n = [], [], [], [], []
            for i, agent in enumerate(trainers):
                avail_actions = self.env.get_avail_agent_actions(i)
                # print(i)
                # print(np.shape(obs_n[i]))
                # print(obs_n[i])
                agent_act, agent_logp, agent_val, agent_pd = agent.step(obs_n[i], gru_state, avail_actions)
                action_scope = np.nonzero(avail_actions)[0]
                agent_act = min(max(agent_act[0], action_scope[0]), action_scope[-1])
                # print('agent_act: {}'.format(agent_act))
                # print('agent_logp: {}'.format(agent_logp))
                # print('agent_val: {}'.format(agent_val))
                # print('agent_pd: {}'.format(agent_pd))
                action_n.append(agent_act)
                logp_n.append(agent_logp[0])
                value_n.append(agent_val[0])
                ava_actions.append(avail_actions)
                pd_n.append(agent_pd[0])
            # print("actions")
            # print(action_n)
            # print("logp_n")
            # print(logp_n)
            # print("value_n")
            # print(value_n)
            # print('finish one loop')

            mb_state.append(obs_n)
            mb_gru_state.append(gru_state)
            mb_value.append(value_n)
            mb_action.append(action_n)
            mb_logp.append(logp_n)
            mb_ava_actions.append(ava_actions)
            mb_pd.append(pd_n)
            rew, done, info_n = self.env.step(action_n)
            rew_n = [rew for i in range(num_agents)]
            done_n = [done for i in range(num_agents)]
            obs_n = self.env.get_obs()
            gru_state = self.env.get_state()
            # print(np.shape(obs_n))
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
            self.episode_rewards[-1] = self.episode_rewards[-1] + rew
            self.episode_step += 1
            terminal = (self.episode_step >= self.max_episode_len)

            if done or terminal:
            # if done:
                self.env.reset()
                obs_n = self.env.get_obs()
                gru_state = self.env.get_state()
                self.episode_step = 0
                self.episode_rewards.append(0)
                
        self.obs0 = obs_n
        self.state0 = gru_state

        last_values = []
        for i, agent in zip(range(num_agents), trainers):
            if done_n[i]:
                last_values.append(0)
            else:
                last_values.append(agent.value_s([gru_state])[0])
        
        mb_state  = np.array(mb_state)
        mb_value  = np.array(mb_value)
        mb_action = np.array(mb_action)
        mb_logp   = np.array(mb_logp)
        mb_pd     = np.array(mb_pd)
        mb_reward = np.array(mb_reward)
        mb_state1 = np.array(mb_state1)
        mb_done   = np.array(mb_done)
        mb_ava_actions = np.array(mb_ava_actions)
        mb_gru_state = np.array(mb_gru_state)

        mb_return = np.zeros_like(mb_reward)
        mb_adv    = np.zeros_like(mb_reward)

        for t in reversed(range(self.nsteps)):
            for i in range(len(trainers)):
                nextnonterminal = 1.0 - mb_done[t, i]
                # if t == self.nsteps - 1:
                if t == self.nsteps - 1:
                    nextvalues = last_values[i]
                    delta = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues - mb_value[t, i]
                    mb_adv[t, i] = delta
                    # mb_return[t, i] = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues
                else:
                    nextvalues = mb_value[t+1, i]
                    delta = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues - mb_value[t, i]
                    mb_adv[t, i] = delta + self.gamma * self.lam * nextnonterminal * mb_adv[t+1, i]
                    # mb_return[t, i] = mb_reward[t, i] + self.gamma * nextnonterminal * mb_return[t+1, i]
                    # mb_return[t, i] = mb_reward[t, i] + self.gamma * nextnonterminal * nextvalues 
        mb_return = mb_adv + mb_value
        # print(np.shape(mb_state))
        # print(np.shape(mb_action))
        # print(np.shape(mb_return))
        # print(np.shape(mb_logp))
        # print(np.shape(mb_adv))
        # print(np.shape(mb_value))
        # print(np.shape(mb_ava_actions))

        # samples = [mb_state, mb_action, mb_reward, mb_state1, mb_done, env_done]
        samples = [mb_state, mb_action, mb_return, mb_logp, mb_adv, mb_value, mb_ava_actions, mb_pd, mb_gru_state]
        return samples, self.episode_rewards
