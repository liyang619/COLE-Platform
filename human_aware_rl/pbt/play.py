from baselines.ppo2.policies import build_policy

import numpy as np
from baselines.common.runners import AbstractEnvRunner
import tensorflow as tf
import time

from baselines.common.tf_util import get_session

MAX_ENT = -np.log(1 / 6)


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model, nsteps):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.sparse_R = [0] * env.num_envs
        self.dones = [False] * env.num_envs

    def run(self):

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


        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            if not all(self.dones):
                overcooked = 'env_name' in self.env.__dict__.keys() and self.env.env_name == "Overcooked-v0"
                if overcooked:
                    actions, values, self.states, neglogpacs, action_probs_agent0 = self.model.step(self.obs0,
                                                                                                    S=self.states,
                                                                                                    M=self.dones)

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


                    other_agent_simulation_time += time.time() - current_simulation_time

                    joint_action = [(actions[i], other_agent_actions[i]) for i in range(len(actions))]



                # Take actions in env and look the results
                # Infos contains a ton of useful informations
                if overcooked:
                    obs, rewards, self.dones, infos = self.env.step(joint_action)

                    both_obs = obs["both_agent_obs"]
                    self.obs0[:] = both_obs[:, 0, :, :]
                    self.obs1[:] = both_obs[:, 1, :, :]
                    self.curr_state = obs["overcooked_state"]
                    self.other_agent_idx = obs["other_agent_env_idx"]
                for i, info in enumerate(infos):

                    if self.dones[i] == True:
                        self.sparse_R[i] = info.get('episode')['ep_sparse_r']
            else:
                break
        tot_time = time.time() - tot_time

        return np.mean(self.sparse_R)


def play(env, model, **network_kwargs):
    #################
    # Added by Yang Li
    # Aim to play with others quickly
    #
    #################
    t0 = time.time()
    network = "conv_and_mlp"
    def model_fn(**kwargs):
        return model
    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    ob_space = env.observation_space
    ac_space = env.action_space

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=0,
                     nsteps=400, ent_coef=0, vf_coef=0,
                     max_grad_norm=0, scope='', load_path=None)

    runner = Runner(env=env, model=model, nsteps=400)
    data = runner.run()
    # print("debug play costs {}s".format(time.time()-t0))
    return data
