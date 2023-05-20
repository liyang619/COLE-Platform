import tensorflow as tf
import numpy as np
from overcooked_ai_py.mdp.actions import Direction, Action
import os
import re

def get_trailing_number(s):
    """
    Get the trailing number from a string,
    i.e. 'file123' -> '123'
    """
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def get_max_iter(agent_folder):
    """Return biggest PBT iteration that has been run"""
    saved_iters = []
    for folder_s in os.listdir(agent_folder):
        folder_iter = get_trailing_number(folder_s) 
        if folder_iter is not None:
            saved_iters.append(folder_iter)
    if len(saved_iters) == 0:
        raise ValueError("Agent folder {} seemed to not have any pbt_iter subfolders".format(agent_folder))
    return max(saved_iters)

class Agent(object):

    def action(self, state):
        return NotImplementedError()

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        pass

def get_pbt_agent_from_config(save_dir=None, sim_threads=0, seed=0, agent_idx=0, best=False, agent_to_load_path=None,iter=0):
    if agent_to_load_path is None:
        agent_folder = save_dir + 'seed_{}/agent{}'.format(seed, agent_idx)
        if best:
            agent_to_load_path = agent_folder  + "/best"
        else:
            agent_to_load_path = agent_folder  + "/pbt_iter" + str(iter)
    agent = get_agent_from_saved_model(agent_to_load_path, sim_threads)
    return agent

def get_agent_from_saved_model(save_dir, sim_threads):
    """Get Agent corresponding to a saved model"""
    # NOTE: Could remove dependency on sim_threads if get the sim_threads from config or dummy env
    state_policy, processed_obs_policy = get_model_policy_from_saved_model(save_dir, sim_threads)
    return AgentFromPolicy(state_policy, processed_obs_policy)


def get_model_policy_from_saved_model(save_dir, sim_threads):
    """Get a policy function from a saved model"""
    predictor = tf.contrib.predictor.from_saved_model(save_dir)
    step_fn = lambda obs: predictor({"obs": obs})["action_probs"]
    return get_model_policy(step_fn, sim_threads)

def get_model_policy(step_fn, sim_threads, is_joint_action=False):
    """
    Returns the policy function `p(s, index)` from a saved model at `save_dir`.
    
    step_fn: a function that takes in observations and returns the corresponding
             action probabilities of the agent
    """
    def encoded_state_policy(observations, stochastic=True, return_action_probs=False):
        """Takes in SIM_THREADS many losslessly encoded states and returns corresponding actions"""
        action_probs_n = step_fn(np.repeat(observations,30,0))

        if return_action_probs:
            return action_probs_n
        
        if stochastic:
            action_idxs = [np.random.choice(len(Action.ALL_ACTIONS), p=action_probs) for action_probs in action_probs_n]
        else:
            action_idxs = [np.argmax(action_probs) for action_probs in action_probs_n]

        return np.array(action_idxs)

    def state_policy(mdp_state, mdp, agent_index, stochastic=True, return_action_probs=False):
        """Takes in a Overcooked state object and returns the corresponding action"""
        obs = mdp.lossless_state_encoding(mdp_state)[agent_index]
        padded_obs = np.array([obs] + [np.zeros(obs.shape)] * (sim_threads - 1))
        action_probs = step_fn(padded_obs)[0] # Discards all padding predictions

        if return_action_probs:
            return action_probs

        if stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)

        if is_joint_action:
            # NOTE: Probably will break for this case, untested
            action_idxs = Action.INDEX_TO_ACTION_INDEX_PAIRS[action_idx]
            joint_action = [Action.INDEX_TO_ACTION[i] for i in action_idxs]
            return joint_action

        return Action.INDEX_TO_ACTION[action_idx]

    return state_policy, encoded_state_policy

class AgentFromPolicy(Agent):
    """
    Defines an agent from a `state_policy` and `direct_policy` functions
    """
    
    def __init__(self, state_policy, direct_policy, stochastic=True, action_probs=False):
        """
        state_policy (fn): a function that takes in an OvercookedState instance and returns corresponding actions
        direct_policy (fn): a function that takes in a preprocessed OvercookedState instances and returns actions
        stochastic (Bool): Whether the agent should sample from policy or take argmax
        action_probs (Bool): Whether agent should return action probabilities or a sampled action
        """
        self.state_policy = state_policy
        self.direct_policy = direct_policy
        self.history = []
        self.stochastic = stochastic
        self.action_probs = action_probs
        self.context=[]

    def action(self, state):
        """
        The standard action function call, that takes in a Overcooked state
        and returns the corresponding action.

        Requires having set self.agent_index and self.mdp
        """
        self.history.append(state)
        try:
            return self.state_policy(state, self.mdp, self.agent_index, self.stochastic, self.action_probs)
        except AttributeError as e:
            raise AttributeError("{}. Most likely, need to set the agent_index or mdp of the Agent before calling the action method.".format(e))

    def direct_action(self, obs):
        """
        A action called optimized for multi-threaded environment simulations
        involving the agent. Takes in SIM_THREADS (as defined when defining the agent)
        number of observations in post-processed form, and returns as many actions.
        """
        return self.direct_policy(obs)[0]

    def reset(self):
        self.history = []