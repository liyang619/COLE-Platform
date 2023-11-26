import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from collections import defaultdict
from multiprocessing import Process
from tensorflow.saved_model import simple_save

from sacred import Experiment
from sacred.observers import FileStorageObserver
import time
import pandas as pd
from shapley_value import Shapley_Value
from utils import probability, ucb_shapley_value, inversed_ucb_shapley_value, ucb_eta
from collections import defaultdict
from game_graph import GameGraph
import math
import ast
import copy

PBT_DATA_DIR = "PATH_TO_SAVE"

ex = Experiment('PBT')
ex.observers.append(FileStorageObserver.create(PBT_DATA_DIR + "pbt_exps_timer"))

from overcooked_ai_py.utils import profile, load_pickle, save_pickle, save_dict_to_file, load_dict_from_file, \
    save_dict_to_json
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
from itertools import permutations, chain, combinations
from human_aware_rl.utils import create_dir_if_not_exists, delete_dir_if_exists, reset_tf, set_global_seed
from human_aware_rl.baselines_utils import create_model, get_vectorized_gym_env, update_model, get_agent_from_model, \
    save_baselines_model, overwrite_model, load_baselines_model, LinearAnnealer,get_agent_from_saved_model,load_trained_model

from human_aware_rl.pbt.pre_load import pre_load


class PBTAgent(object):
    """An agent that can be saved and loaded and all and the main data it contains is the self.model

    Goal is to be able to pass in save_locations or PBTAgents to workers that will load such agents
    and train them together.
    """

    def __init__(self, agent_name, start_params, save_path=None, start_logs=None, model=None, gym_env=None):
        self.params = start_params
        self.logs = start_logs if start_logs is not None else {
            "agent_name": agent_name,
            "avg_rew_0": [],
            "avg_rew_1": [],
            "params_hist": defaultdict(list),
            "num_ppo_runs": 0,
            "reward_shaping": []
        }
        self.id = save_path
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            self.model = model if model is not None else create_model(gym_env, agent_name, **start_params)

    @property
    def num_ppo_runs(self):
        return self.logs["num_ppo_runs"]

    @property
    def agent_name(self):
        return self.logs["agent_name"]

    def get_agent(self):
        return get_agent_from_model(self.model, self.params["sim_threads"])

    def update(self, gym_env, population=None, other_partners=None):
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            train_info = update_model(gym_env, self.model, population=population, other_partners=other_partners, **self.params)
            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1

    def update_avg_rew_logs(self, data0, data1):
        self.logs["avg_rew_0"].append(np.mean(data0))
        self.logs["avg_rew_1"].append(np.mean(data1))

    def save(self, save_folder):
        """Save agent model, logs, and parameters"""
        create_dir_if_not_exists(save_folder)
        save_baselines_model(self.model, save_folder)
        save_dict_to_file(dict(self.logs), save_folder + "logs")
        save_dict_to_file(self.params, save_folder + "params")

    @staticmethod
    def from_dir(load_folder, agent_name):
        try:
            logs = load_dict_from_file(load_folder + "logs.txt")
        except:
            logs = None
        params = load_dict_from_file(load_folder + "params.txt")
        model = load_baselines_model(load_folder[0:-1], agent_name, params)
        return PBTAgent(agent_name, params, save_path=load_folder, start_logs=logs, model=model)

    @staticmethod
    def from_dir_train(load_folder, agent_name):
        try:
            logs = load_dict_from_file(load_folder + "logs.txt")
        except:
            logs = None
        params = load_dict_from_file(load_folder + "params.txt")
        model = load_trained_model(load_folder[0:-1], agent_name, params)
        return PBTAgent(agent_name, params, save_path=load_folder, start_logs=logs, model=model)

    @staticmethod
    def update_from_files(file0, file1, gym_env, save_dir):
        pbt_agent0 = PBTAgent.from_dir(file0)
        pbt_agent1 = PBTAgent.from_dir(file1)
        gym_env.other_agent = pbt_agent1
        pbt_agent0.update(gym_env)
        return pbt_agent0

    def save_predictor(self, save_folder):
        """Saves easy-to-load simple_save tensorflow predictor for agent"""
        simple_save(
            tf.get_default_session(),
            save_folder,
            inputs={"obs": self.model.act_model.X},
            outputs={
                "action": self.model.act_model.action,
                "value": self.model.act_model.vf,
                "action_probs": self.model.act_model.action_probs
            }
        )

    def update_pbt_iter_logs(self):
        for k, v in self.params.items():
            self.logs["params_hist"][k].append(v)
        self.logs["params_hist"] = dict(self.logs["params_hist"])

    def explore_from(self, best_training_agent):
        overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = self.mutate_params(best_training_agent.params)

    def copy_from(self, best_training_agent):
        overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = best_training_agent.params

    def mutate_params(self, params_to_mutate):
        params_to_mutate = params_to_mutate.copy()
        for k in self.params["HYPERPARAMS_TO_MUTATE"]:
            if np.random.random() < params_to_mutate["RESAMPLE_PROB"]:
                mutation = np.random.choice(self.params["MUTATION_FACTORS"])

                if k == "LAM":
                    # Move eps/2 in either direction
                    eps = min(
                        (1 - params_to_mutate[k]) / 2,  # If lam is > 0.5, avoid going over 1
                        params_to_mutate[k] / 2  # If lam is < 0.5, avoid going under 0
                    )
                    rnd_direction = (-1) ** np.random.randint(2)
                    mutation = rnd_direction * eps
                    params_to_mutate[k] = params_to_mutate[k] + mutation
                elif type(params_to_mutate[k]) is int:
                    params_to_mutate[k] = max(int(params_to_mutate[k] * mutation), 1)
                else:
                    params_to_mutate[k] = params_to_mutate[k] * mutation

                print("Mutated {} by a factor of {}".format(k, mutation))

        print("Old params", self.params)
        print("New params", params_to_mutate)
        return params_to_mutate


@ex.config
def my_config():
    ##################
    # GENERAL PARAMS #
    ##################

    TIMESTAMP_DIR = False
    EX_NAME = "undefined_name"

    if TIMESTAMP_DIR:
        SAVE_DIR = PBT_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + EX_NAME + "/"
    else:
        SAVE_DIR = PBT_DATA_DIR + EX_NAME + "/"

    print("Saving data to ", SAVE_DIR)

    RUN_TYPE = "pbt"
    LOAD_FOLDER_PATH = '/nas/zhangshao/DATA_DIR/'
    # for Apex Run
    # LOAD_FOLDER_PATH = '/NAS2020/Workspaces/DRLGroup/zhangshao/DATA_DIR'

    RESUME = False
    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = False

    # GPU id to use
    GPU_ID = 1

    # List of seeds to run
    SEEDS = [0]

    init_pop_size = 3

    # Number of parallel environments used for simulating rollouts
    sim_threads = 64 if not LOCAL_TESTING else 2

    ##############
    # PBT PARAMS #
    ##############

    # Entropy bonus coefficient for the model pool
    ENTROPY_POOL = 0

    ITER_PER_SELECTION = 100  # How many pairings and model training updates before the worst model is overwritten

    RESAMPLE_PROB = 0.33
    MUTATION_FACTORS = [0.75, 1.25]
    HYPERPARAMS_TO_MUTATE = ["LAM", "CLIPPING", "LR", "STEPS_PER_UPDATE", "ENTROPY", "VF_COEF"]

    ##############
    # PPO PARAMS #
    ##############

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 1000
    NUM_PBT_ITER = 100  # int(PPO_RUN_TOT_TIMESTEPS//ITER_PER_SELECTION)

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 1000

    # Number of minibatches we divide up each batch into before
    # performing gradient steps
    MINIBATCHES = 5 if not LOCAL_TESTING else 1

    BATCH_SIZE = int(TOTAL_BATCH_SIZE // sim_threads)

    print('++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++')
    print('==========================')
    print("PPO_RUN_TOT_TIMESTEPS: {}".format(PPO_RUN_TOT_TIMESTEPS))
    print("ITER_PER_SELECTION: {}".format(ITER_PER_SELECTION))
    print("NUM_PBT_ITER: {}".format(NUM_PBT_ITER))
    print("TOTAL_BATCH_SIZE: {}".format(TOTAL_BATCH_SIZE))
    print("MINIBATCHES: {}".format(MINIBATCHES))
    print("BATCH_SIZE: {}".format(BATCH_SIZE))
    print("sim_threads: {}".format(sim_threads))
    print('==========================')
    print('++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++')
    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1

    # Learning rate
    LR = 5e-3

    # Entropy bonus coefficient
    ENTROPY = 0.5

    # Value function coefficient
    VF_COEF = 0.1

    # Gamma discounting factor
    GAMMA = 0.99

    # Lambda advantage discounting factor
    LAM = 0.98

    # Max gradient norm
    MAX_GRAD_NORM = 0.1

    # PPO clipping factor
    CLIPPING = 0.05

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = 0

    ##################
    # NETWORK PARAMS #
    ##################

    # Network type used
    NETWORK_TYPE = "conv_and_mlp"

    # Network params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    ##################
    # MDP/ENV PARAMS #
    ##################

    # Mdp params
    layout_name = None
    start_order_list = None

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }

    # Env params
    horizon = 400

    #########
    # OTHER #
    #########

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6]
    }
    RATIO = "2:2"
    # Approximate info stats
    # GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE) * ITER_PER_SELECTION * NUM_PBT_ITER // POPULATION_SIZE
    # print("Grad updates per agent", GRAD_UPDATES_PER_AGENT  )
    MAX_NEG_POP_SIZE = 50
    params = {
        "RATIO": RATIO,
        "MAX_NEG_POP_SIZE": MAX_NEG_POP_SIZE,
        "init_pop_size": init_pop_size,
        "RESUME": RESUME,
        "LOAD_FOLDER_PATH": LOAD_FOLDER_PATH,
        "ENTROPY_POOL": ENTROPY_POOL,
        "LOCAL_TESTING": LOCAL_TESTING,
        "RUN_TYPE": RUN_TYPE,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        },
        "env_params": {
            "horizon": horizon
        },
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "NUM_PBT_ITER": NUM_PBT_ITER,
        "ITER_PER_SELECTION": ITER_PER_SELECTION,
        "RESAMPLE_PROB": RESAMPLE_PROB,
        "MUTATION_FACTORS": MUTATION_FACTORS,
        "mdp_generation_params": mdp_generation_params,  # NOTE: currently not used
        "HYPERPARAMS_TO_MUTATE": HYPERPARAMS_TO_MUTATE,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "ENTROPY": ENTROPY,
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "NETWORK_TYPE": NETWORK_TYPE,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "SEEDS": SEEDS,
        # "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
    }


@ex.named_config
def fixed_mdp():
    LOCAL_TESTING = False
    # fixed_mdp = True
    layout_name = "simple"


    sim_threads = 30 if not LOCAL_TESTING else 2
    PPO_RUN_TOT_TIMESTEPS = 36000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 5
    MINIBATCHES = 6 if not LOCAL_TESTING else 2

    LR = 5e-4


def play(agent1, agent2, env, num_rounds=3):
    #################
    # Added by Yang Li
    # Aim to play with agents and return mean rewards
    #
    #################
    agent_pair = AgentPair(agent1, agent2, allow_duplicate_agents=True)
    trajs = env.get_rollouts(agent_pair, num_rounds)
    sparse_rews1 = trajs["ep_returns_sparse"]
    return np.mean(sparse_rews1)


def update_payoffs(population, ego_agent, payoffs, env):
    #################
    # Added by Yang Li
    # Aim to update payoffs by playing with each agent
    #
    #################
    N = len(population)

    if N != 0:
        for i in range(N):
            teammate = population[i].get_agent()
            payoffs[i, N] = play(teammate, ego_agent.get_agent(), env)
            payoffs[N, i] = play(ego_agent.get_agent(), teammate, env)
    payoffs[N, N] = play(ego_agent.get_agent(), ego_agent.get_agent(), env)

    return payoffs


def save_agent(path, agent):
    delete_dir_if_exists(path, verbose=True)
    agent.save_predictor(path)
    agent.save(path)
    print('neg agent saved to {}'.format(path))


def sym_matrix(matrix):
    new_m = np.zeros_like(matrix)
    for i in range(50):
        for j in range(i, 50):
            new_m[i, j] = new_m[j, i] = (matrix[i][j] + matrix[j][i]) / 2
    return new_m


def update_main_pop(main_population, agent0, main_payoffs, overcooked_env, params, pbt_iter, TIME):
    # test is nta ranking last, if yes train it again
    # the base population size is 15
    history_len = len(main_population)
    main_payoffs_tmp = update_payoffs(main_population, agent0, main_payoffs, overcooked_env)

    his_payoffs = copy.deepcopy(main_payoffs_tmp)
    his_payoffs = sym_matrix(his_payoffs)
    PG = GameGraph(his_payoffs, list(range(history_len+1)))
    eta = PG.eta()
    min_index = np.argmin(eta)
    if pbt_iter > 50:
        percent = 0.2
    else:
        percent = 0.1
    threshold = int(history_len * percent) + 1
    if not (min_index < history_len - threshold) or TIME == 2 or pbt_iter < params['init_pop_size']:
        flag = True
        save_neg_pop_path = params["SAVE_DIR"] + 'main_pop/{}/'.format(pbt_iter)
        save_agent(save_neg_pop_path, agent0)
        agent = PBTAgent.from_dir(save_neg_pop_path, 'agent{}'.format(pbt_iter))
        main_population.append(agent)
        print("main_population size is {}".format(len(main_population)))
        return main_payoffs_tmp, main_population, flag
    else:
        flag = False
        return main_payoffs, main_population, flag



def pop_dequeue(main_population, main_payoffs, params):
    #################
    # Added by Yang Li
    # Aim to maintain a fixed size neg population
    #
    #################
    while not len(main_population) < params["MAX_NEG_POP_SIZE"] + 1:
        # pop latest generated model
        # index = random.choices([i for i in range(start, start+10)], k=1)[0]
        # 09 02 2023 modified: deleted model randomly in the past population
        # index = random.choices([i for i in range(0, len(main_population)-1)], k=1)[0]
        start = 0
        index = random.choices([i for i in range(start, start+10)], k=1)[0]
        main_population.pop(index)
        main_payoffs = np.delete(main_payoffs, index, 0)
        main_payoffs = np.delete(main_payoffs, index, 1)
        print("neg population is filled, pop {} th agent, and now neg population has {} agents".format(index, len(main_population)))

    assert len(main_population) < params["MAX_NEG_POP_SIZE"] + 1

    return main_population, main_payoffs


def resume(params, main_population):
    #################
    # Added by Yang Li
    # Aim to resume training from checkpoint
    # To completed
    #################
    save_dir = params["SAVE_DIR"]
    ego_path = os.path.join(save_dir, "ego_agent")
    ego_names = [int(i.replace("pbt_iter", "")) for i in os.listdir(ego_path) if "pbt_iter" in i]
    ego_names = sorted(ego_names)

    main_path = os.path.join(save_dir, "main_pop")
    main_names = [int(i) for i in os.listdir(main_path)]
    main_names = sorted(main_names)

    max_iter = min(ego_names[-1], main_names[-1])
    print(ego_names[-1], main_names[-1], max_iter)
    while main_names[-1] != max_iter:
        agent_path = os.path.join(save_dir, "main_pop", str(main_names[-1])) + '/'
        delete_dir_if_exists(agent_path)
        main_names.pop()
        print("{} is aborted.".format(agent_path))
    while ego_names[-1] != max_iter:
        ego_name = "pbt_iter" + str(ego_names[-1])
        ego_path = os.path.join(ego_path, str(ego_name)) + '/'
        delete_dir_if_exists(ego_path)
        ego_names.pop()
        print("{} is aborted.".format(ego_path))
    assert ego_names[-1] == main_names[-1] == max_iter

    ego_name = "pbt_iter" + str(ego_names[-1])
    ego_path = os.path.join(ego_path, str(ego_name)) + '/'
    agent_path = os.path.join(save_dir, "main_pop", str(main_names[-1])) + '/'
    if not os.path.isfile(os.path.join(ego_path, "path_info.txt")) or not os.path.isfile(os.path.join(agent_path, "params.txt")):
        print('path_info.txt is not exist in {} and delete it and its copy in {}'.format(ego_path, agent_path))
        delete_dir_if_exists(ego_path)
        delete_dir_if_exists(agent_path)
        ego_names.pop()
        main_names.pop()
        ego_path = os.path.join(save_dir, "ego_agent")
        ego_name = "pbt_iter" + str(ego_names[-1])
        ego_path = os.path.join(ego_path, str(ego_name)) + '/'
        agent_path = os.path.join(save_dir, "main_pop", str(ego_names[-1])) + '/'

    print('now the path is in {}'.format(ego_path))
    dec_main_names = sorted(main_names, reverse=True)

    if len(dec_main_names) > params['MAX_NEG_POP_SIZE']:
        print("now the main pop size {} is bigger than {}".format(len(dec_main_names), params['MAX_NEG_POP_SIZE']))
        dec_main_names = dec_main_names[:params['MAX_NEG_POP_SIZE']]
        print("we only need load newest {} agents:\n\t {}".format(len(dec_main_names), dec_main_names))

    main_names = sorted(dec_main_names)
    for name in main_names:
        p = os.path.join(save_dir, "main_pop", str(name) + '/')
        print('reload main population from {}'.format(p))
        agent = PBTAgent.from_dir(p, 'agent{}'.format(name))
        main_population.append(agent)
    print('we load {} main agents'.format(len(main_population)))

    iter = ego_names[-1]
    ego_agent = PBTAgent.from_dir_train(agent_path, 'ego_agent')

    neg_payoffs = np.array(pd.read_csv(os.path.join(ego_path, "payoffs_neg.csv"), header=None))
    main_payoffs = np.array(pd.read_csv(os.path.join(ego_path, "main_payoffs.csv"), header=None))

    #expand main_payoffs to 1000*1000
    full_main_payoffs = np.zeros((1000, 1000))
    for i in range(main_payoffs.shape[0]):
        for j in range(main_payoffs.shape[1]):
            full_main_payoffs[i, j] = main_payoffs[i, j]
    main_payoffs = full_main_payoffs

    print(main_payoffs.shape, neg_payoffs.shape)

    neg_population = []

    pop_info_path = os.path.join(ego_path, "path_info.txt")
    with open(pop_info_path, 'r') as f:
        data = f.readlines()[0]
        data = ast.literal_eval(data)
        neg_paths = data['neg_path']

    for i, neg_pop_path in enumerate(neg_paths):
        print('reload neg population 0 from {}'.format(neg_pop_path))
        # neg_pop_path = neg_pop_path.replace("god_game_shapley", "half_god_game_shapley")
        neg_population.append(PBTAgent.from_dir(neg_pop_path, 'neg_agent{}'.format(i)))

    print('we load {} neg agent'.format(len(neg_population)))

    pop_info_path = os.path.join(ego_path, "shapley_info.txt")
    with open(pop_info_path, 'r') as f:
        data = f.readlines()[0]
        data = ast.literal_eval(data)
        neg_sv = data["neg_sv"]
        main_sv = data["main_sv"]

    best_info_path = os.path.join(save_dir, "ego_agent", 'best', 'best_info.txt')
    with open(best_info_path, 'r') as f:
        data = f.readlines()[0]
        data = ast.literal_eval(data)

        best_0 = data['best_sparse_rew_avg_0']
        best_1 = data['best_sparse_rew_avg_1']

    return iter, (main_population, main_payoffs, main_sv), ego_agent, \
           (neg_population, neg_payoffs, neg_sv), (best_0, best_1)



def main_train(params, seed):
    # Iterating noptepochs over same batch data but shuffled differently
    # dividing each batch in `nminibatches` and doing a gradient step for each one
    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")

    # 0:4 1:3, 2:2,3:1 4:0
    in_nums, neg_nums = int(params["RATIO"].split(":")[0]), int(params["RATIO"].split(":")[1])
    print("The training ratio is {}".format(params["RATIO"]))
    print("Each time we sample {} sp, and {} neg".format(in_nums, neg_nums))

    mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
    overcooked_env = OvercookedEnv(mdp, **params["env_params"])

    print("Sample training environments:")
    for _ in range(5):
        overcooked_env.reset()
        print(overcooked_env)

    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
    )
    gym_env.update_reward_shaping_param(1.0)  # Start reward shaping from 1

    REW_SHAPING_HORIZON = params["init_pop_size"] * params["PPO_RUN_TOT_TIMESTEPS"] * (params["REW_SHAPING_HORIZON"] - params["init_pop_size"]) - params["init_pop_size"]*params["PPO_RUN_TOT_TIMESTEPS"]

    annealer = LinearAnnealer(horizon=REW_SHAPING_HORIZON)

    # AGENT POPULATION INITIALIZATION

    if params["RESUME"] == True:
        print("now we resume from path....")
        main_population = []
        start_iter, main_info, agent_train, neg_info, best_info = resume(params, main_population)
        main_population, main_payoffs, main_sv_value = main_info
        neg_population, neg_payoffs, neg_sv_value = neg_info
        best_sparse_rew_avg_0, best_sparse_rew_avg_1 = best_info
        start_iter += 1
    else:
        print("start training from scratch")

        save_neg_pop_path = params["SAVE_DIR"] + 'main_pop/{}/'.format(0)

        agent_train = PBTAgent('ego_agent', params, gym_env=gym_env, save_path=save_neg_pop_path)
        save_agent(save_neg_pop_path, agent_train)

        main_payoffs = np.zeros((1000,1000))


        main_population = [PBTAgent.from_dir(save_neg_pop_path, 'agent{}'.format(0))]
        neg_sv_value = [1]
        main_payoffs[0, 0] = play(agent_train.get_agent(), agent_train.get_agent(), overcooked_env)
        start_iter = 1
        best_sparse_rew_avg_0 = -np.Inf
        best_sparse_rew_avg_1 = -np.Inf

    # MAIN LOOP

    print(params)
    TIME = 0
    for pbt_iter in range(start_iter, 1+params["NUM_PBT_ITER"]):
        print("\n\n\nPBT ITERATION NUM {}/{}".format(pbt_iter, params["NUM_PBT_ITER"]))
        train_start_time = time.time()
        # TRAINING PHASE
        neg_visitation_info = defaultdict(lambda: 0)
        if len(neg_sv_value) < params["init_pop_size"] + 1:
            iner_times = 1
        else:
            iner_times = params["init_pop_size"]
        for i in range(iner_times): #debug
            start_time = time.time()
            # other_agents is tuple list, each element in it includes agent_indx and agent
            other_agents = []

            # sp and our
            if iner_times == 1:
                print("len(neg_sv_value)==1")
                index = "1" if random.random() > 0.5 else "0"
                other_agents += [(index, agent_train.get_agent())]
                index = "1" if random.random()>0.5 else "0"
                other_agents += [(index, agent_train.get_agent())]
                index = "1" if random.random()>0.5 else "0"
                other_agents += [(index, agent_train.get_agent())]
                index = "1" if random.random()>0.5 else "0"
                other_agents += [(index, agent_train.get_agent())]
            else:
                for i_iter in range(in_nums):
                    index = "1" if random.random() > 0.5 else "0"
                    other_agents += [(index, agent_train.get_agent())]
                for neg_iter in range(neg_nums):
                    neg_weight_r = inversed_ucb_shapley_value(neg_sv_value, visits=neg_visitation_info)
                    other_idx = random.choices(list(range(len(neg_weight_r))), weights=neg_weight_r, k=1)[0]
                    neg_visitation_info[str(other_idx)] += 1
                    index = "1" if random.random() > 0.5 else "0"
                    other_agents += [(index, main_population[other_idx].get_agent())]


            agent_env_steps = agent_train.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
            reward_shaping_param = annealer.param_value(agent_env_steps)
            # print("Current reward shaping:", reward_shaping_param, "\t Save_dir", params["SAVE_DIR"])
            agent_train.logs["reward_shaping"].append(reward_shaping_param)
            gym_env.update_reward_shaping_param(reward_shaping_param)

            if len(other_agents) == 0:
                other_partners = None
            else:
                other_partners = other_agents

            # print(other_partners)
            agent_train.update(gym_env, other_partners=other_partners)

            print("Iter {} trainning costs {:.3f} seconds.".format(i, time.time() - start_time))

        print("stage 1: Training this iter costs {:.3f} seconds.".format(time.time() - train_start_time))
        # update pos payoffs
        start_time = time.time()

        # update neg population main_population, agent0, main_payoffs, overcooked_env, params, pbt_iter, TIME
        main_payoffs, main_population, flag = update_main_pop(main_population, agent_train, main_payoffs, overcooked_env, params, pbt_iter, TIME)

        print("stage 2: Update payoffs and neg info costs {:.3f} seconds.".format(time.time() - start_time))

        if flag == False:
            TIME += 1
            print("ETA is not satisfied, train it again. Repeats {}/3 times.".format(TIME+1))
            continue

        # update main population info like population, payoffs and sv
        start_time = time.time()
        main_population, main_payoffs = pop_dequeue(main_population, main_payoffs, params)
        neg_sv_value = Shapley_Value(main_payoffs, list(range(len(main_population))))

        print("Stage 2: update populations costs {:.3f} seconds.".format(time.time() - start_time))

        TIME = 0
        # save
        save_folder = params["SAVE_DIR"] + agent_train.agent_name + '/'
        os.makedirs(save_folder + "pbt_iter{}/".format(pbt_iter), exist_ok=True)
        main_path = [i.id for i in main_population]

        save_dict_to_file({
                           'main_path': list(main_path),
                           }, save_folder + "pbt_iter{}/path_info".format(pbt_iter))


        save_dict_to_file({
                           'main_sv': list(neg_sv_value),
                           }, save_folder + "pbt_iter{}/shapley_info".format(pbt_iter))

        save_dict_to_file(dict(agent_train.logs), save_folder + "logs")

        pd.DataFrame(main_payoffs[:len(main_population), : len(main_population)]).to_csv(save_folder + 'pbt_iter{}/main_payoffs.csv'.format(pbt_iter), header=None, index=None)
        print("Stage 3: Save model info costs {:.3f} seconds.".format(time.time() - start_time))

        print("\nEVALUATION PHASE\n")

        start_time = time.time()
        # Dictionary with average returns for each agent when matched with each other agent
        avg_ep_returns_sparse_0 = []
        avg_ep_returns_sparse_1 = []

        ego_agent = agent_train

        # Saving each agent model at the end of the pbt iteration
        if iner_times > 1:
            ego_agent.update_pbt_iter_logs()
            eval_len = 10 if len(main_population) > 10 else len(main_population)
            for j in range(eval_len):
                print("Evaluating agent {} and {}".format(0, j))
                pbt_agent_other = main_population[j]
                agent_pair = AgentPair(ego_agent.get_agent(), pbt_agent_other.get_agent())
                trajs = overcooked_env.get_rollouts(agent_pair, 5, reward_shaping=0)
                dense_rews, sparse_rews, lens = trajs["ep_returns"], trajs["ep_returns_sparse"], trajs["ep_lengths"]
                avg_ep_returns_sparse_0.append(sparse_rews)
                # switch the agent pair
                print("Evaluating agent {} and {}".format(j, 0))

                agent_pair = AgentPair(pbt_agent_other.get_agent(), ego_agent.get_agent())
                trajs = overcooked_env.get_rollouts(agent_pair, 5, reward_shaping=0)
                dense_rews, sparse_rews, lens = trajs["ep_returns"], trajs["ep_returns_sparse"], trajs["ep_lengths"]
                avg_ep_returns_sparse_1.append(sparse_rews)

            print("AVG sparse rewards", avg_ep_returns_sparse_0)
            print("AVG sparse rewards", avg_ep_returns_sparse_1)

            ego_agent.update_avg_rew_logs(avg_ep_returns_sparse_0, avg_ep_returns_sparse_1)

            avg_sparse_rew_0 = np.mean(avg_ep_returns_sparse_0)
            avg_sparse_rew_1 = np.mean(avg_ep_returns_sparse_1)
            if (avg_sparse_rew_0 > best_sparse_rew_avg_0) and (avg_sparse_rew_1 > best_sparse_rew_avg_1):
                best_sparse_rew_avg_0 = avg_sparse_rew_0
                best_sparse_rew_avg_1 = avg_sparse_rew_1
                agent_name = ego_agent.agent_name
                print("New best avg sparse rews {} and {} for agent {}, saving...".format(avg_ep_returns_sparse_0, avg_ep_returns_sparse_1, agent_name))

            print("Stage 4: Evaluation model costs {:.3f} seconds.".format(time.time() - start_time))
        print("stage all : Training this iteration costs {:.3f} seconds.".format(time.time() - train_start_time))


    reset_tf()
    print(params["SAVE_DIR"])

    end_sentences = "Now, the training for seed {} is done.".format(seed)
    split_sentence = "=" * len(end_sentences)
    print(split_sentence)
    print(end_sentences)
    print(split_sentence)


@ex.automain
def run_pbt(params):
    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")
    # save_dict_to_json(params, params["SAVE_DIR"] + "config")
    for seed in params["SEEDS"]:
        set_global_seed(seed)
        curr_seed_params = params.copy()
        curr_seed_params["SAVE_DIR"] += "seed_{}/".format(seed)
        main_train(curr_seed_params, seed)
