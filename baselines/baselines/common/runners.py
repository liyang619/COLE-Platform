import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)

        overcooked = 'env_name' in env.__dict__.keys() and env.env_name == "Overcooked-v0"
        if overcooked:
            self.obs0 = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            self.obs1 = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)

            obs = env.reset()
            both_obs = obs["both_agent_obs"]
            self.obs0[:] = both_obs[:, 0, :, :]
            self.obs1[:] = both_obs[:, 1, :, :]
            self.curr_state = obs["overcooked_state"]
            self.other_agent_idx = obs["other_agent_env_idx"]
        else:
            self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

