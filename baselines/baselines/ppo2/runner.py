import numpy as np
from baselines.common.runners import AbstractEnvRunner
from collections import deque

MAX_ENT = -np.log(1 / 6)


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model, nsteps, gamma, lam, others=None, population=None, ent_pool_coef=0.0, history_len=10):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.population = population
        self.ent_pool_coef = ent_pool_coef
        self.entropy_pop_delta_history = deque(maxlen=history_len)
        self.entropy_pop_new_history = deque(maxlen=history_len)
        self.neg_logp_pop_new_history = deque(maxlen=history_len)
        self.neg_logp_pop_delta_history = deque(maxlen=history_len)
        self.entropy_pop_delta_mean = 0.0
        self.entropy_pop_new_mean = 0.0
        self.neg_logp_pop_new_mean = 0.0
        self.neg_logp_pop_delta_mean = 0.0
        self.others = others

    def remote_set_index(self, index):
        self.env.venv.remote_set_agent_idx(index)

    def remote_get_index(self):
        return self.env.venv.remote_get_agent_idx()

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        if self.others != None:
            index, other_agent = self.others
            self.env.other_agent = other_agent
            if self.remote_get_index() != index:
                self.remote_set_index(index)

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        # For n in range number of steps

        import time
        tot_time = time.time()
        int_time = 0
        num_envs = len(self.curr_state)

        if self.env.trajectory_sp:
            # Selecting which environments should run fully in self play
            sp_envs_bools = np.random.random(num_envs) < self.env.self_play_randomization
            print("SP envs: {}/{}".format(sum(sp_envs_bools), num_envs))

        other_agent_simulation_time = 0

        from overcooked_ai_py.mdp.actions import Action

        def other_agent_action():
            if self.env.use_action_method:
                other_agent_actions = self.env.other_agent.actions(self.curr_state, self.other_agent_idx)
                return [Action.ACTION_TO_INDEX[a] for a in other_agent_actions]
            else:
                other_agent_actions = self.env.other_agent.direct_policy(self.obs1)
                return other_agent_actions

        def entropy(action_probs, eps=1e-4):
            """
            action_probs shape: (num_examples, num_classes)
            output shape: (num_examples)
            """
            assert action_probs.shape[1] == 6, 'action_probs.shape[1] == 6'
            neg_p_logp = - action_probs * np.log(action_probs)
            entropy = np.sum(neg_p_logp, axis=1)
            assert np.max(entropy) <= MAX_ENT + 1e5, 'entropy_max <= MAX_ENT'
            return entropy

        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            overcooked = 'env_name' in self.env.__dict__.keys() and self.env.env_name == "Overcooked-v0"
            if overcooked:
                if self.population is not None:
                    pop_len = len(self.population)
                    action_probs_np = np.zeros((pop_len, self.obs0.shape[0], 6))  ## 6 is the action_dim
                    actions_np = np.zeros((pop_len, self.obs0.shape[0]))
                    agent_i = 0
                    for agent in self.population:
                        actions, _, _, _, action_probs = agent.model.step(self.obs0, S=self.states, M=self.dones)
                        actions_np[agent_i] = actions.copy()
                        action_probs_np[agent_i] = action_probs.copy()
                        agent_i += 1
                    action_probs_pop_np = np.mean(action_probs_np, axis=0)
                    # entropy_pop = entropy(action_probs_pop_np)

                actions, values, self.states, neglogpacs, action_probs_agent0 = self.model.step(self.obs0,
                                                                                                S=self.states,
                                                                                                M=self.dones)

                if self.population is not None:
                    action_probs_np_new = np.append(action_probs_np, np.expand_dims(action_probs_agent0, axis=0),
                                                    axis=0)
                    action_probs_pop_np_new = np.mean(action_probs_np_new, axis=0)
                    entropy_pop = entropy(action_probs_pop_np)
                    entropy_pop_new = entropy(action_probs_pop_np_new)
                    entropy_pop_delta = entropy_pop_new - entropy_pop

                    sampled_action_prob_pop_np = np.take(action_probs_pop_np, actions)
                    neg_logp_pop = - np.log(sampled_action_prob_pop_np)
                    sampled_action_prob_pop_np_new = np.take(action_probs_pop_np_new, actions)
                    neg_logp_pop_new = - np.log(sampled_action_prob_pop_np_new)
                    neg_logp_pop_delta = neg_logp_pop_new - neg_logp_pop

                import time
                current_simulation_time = time.time()

                # Randomize at either the trajectory level or the individual timestep level
                if self.env.trajectory_sp:

                    # If there are environments selected to not run in SP, generate actions
                    # for the other agent, otherwise we skip this step.
                    if sum(sp_envs_bools) != num_envs:
                        other_agent_actions_bc = other_agent_action()

                    # If there are environments selected to run in SP, generate self-play actions
                    if sum(sp_envs_bools) != 0:
                        other_agent_actions_sp, _, _, _ = self.model.step(self.obs1, S=self.states, M=self.dones)

                    # Select other agent actions for each environment depending on whether it was selected
                    # for self play or not
                    other_agent_actions = []
                    for i in range(num_envs):
                        if sp_envs_bools[i]:
                            sp_action = other_agent_actions_sp[i]
                            other_agent_actions.append(sp_action)
                        else:
                            bc_action = other_agent_actions_bc[i]
                            other_agent_actions.append(bc_action)

                else:
                    other_agent_actions = np.zeros_like(self.curr_state)

                    if self.env.self_play_randomization < 1:
                        # Get actions through the action method of the agent
                        other_agent_actions = other_agent_action()

                    # Naive non-parallelized way of getting actions for other
                    if self.env.self_play_randomization > 0:
                        self_play_actions, _, _, _ = self.model.step(self.obs1, S=self.states, M=self.dones)
                        self_play_bools = np.random.random(num_envs) < self.env.self_play_randomization

                        for i in range(num_envs):
                            is_self_play_action = self_play_bools[i]
                            if is_self_play_action:
                                other_agent_actions[i] = self_play_actions[i]

                # NOTE: This has been discontinued as now using .other_agent_true takes about the same amount of time
                # elif self.env.other_agent_bc:
                #     # Parallelise actions with direct action, using the featurization function
                #     featurized_states = [self.env.mdp.featurize_state(s, self.env.mlp) for s in self.curr_state]
                #     player_featurizes_states = [s[idx] for s, idx in zip(featurized_states, self.other_agent_idx)]
                #     other_agent_actions = self.env.other_agent.direct_policy(player_featurizes_states, sampled=True, no_wait=True)

                other_agent_simulation_time += time.time() - current_simulation_time

                joint_action = [(actions[i], other_agent_actions[i]) for i in range(len(actions))]

                mb_obs.append(self.obs0.copy())
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
                mb_obs.append(self.obs.copy())

            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            if overcooked:
                obs, rewards, self.dones, infos = self.env.step(joint_action)
                if self.population is not None:
                    rewards_np = np.array(rewards)
                    rewards_np = rewards_np + self.ent_pool_coef * neg_logp_pop_new

                    self.entropy_pop_delta_history.append(np.mean(entropy_pop_delta))
                    self.entropy_pop_new_history.append(np.mean(entropy_pop_new))
                    self.neg_logp_pop_new_history.append(np.mean(neg_logp_pop_new))
                    self.neg_logp_pop_delta_history.append(np.mean(neg_logp_pop_delta))

                    self.entropy_pop_delta_mean = np.mean(self.entropy_pop_delta_history)
                    self.entropy_pop_new_mean = np.mean(self.entropy_pop_new_history)
                    self.neg_logp_pop_new_mean = np.mean(self.neg_logp_pop_new_history)
                    self.neg_logp_pop_delta_mean = np.mean(self.neg_logp_pop_delta_history)

                    rewards = rewards_np.tolist()
                both_obs = obs["both_agent_obs"]
                self.obs0[:] = both_obs[:, 0, :, :]
                self.obs1[:] = both_obs[:, 1, :, :]
                self.curr_state = obs["overcooked_state"]
                self.other_agent_idx = obs["other_agent_env_idx"]
            else:
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        # print("Other agent actions took", other_agent_simulation_time, "seconds")
        tot_time = time.time() - tot_time
        if _ == 1:
            print("Total simulation time for {} steps: {} \t Other agent action time: {} \t {} steps/s".format(self.nsteps,
                                                                                                               tot_time,
                                                                                                               int_time,
                                                                                                               self.nsteps / tot_time))

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
