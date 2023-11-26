import shutil
import time, gym, copy, seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict

import tqdm
from overcooked_ai_py.utils import save_pickle, load_pickle, load_dict_from_txt,save_dict_to_file
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from human_aware_rl.utils import set_global_seed, prepare_nested_default_dict_for_pickle, common_keys_equal
from human_aware_rl.baselines_utils import get_agent_from_saved_model
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved,get_part_bc_agent_from_saved
from human_aware_rl.utils import get_max_iter
import os
import ast
import random
import argparse

PBT_DATA_DIR = "./data/cfs"
# Visualization

def plot_pbt_runs(pbt_model_paths, seeds, single=False, save=False, show=False):
    """Plots sparse rewards"""
    for layout in pbt_model_paths.keys():
        logs_and_cfgs = get_logs(pbt_model_paths[layout], seeds=seeds)
        log, cfg = logs_and_cfgs[0]

        ep_rew_means = []
        for l, cfg in logs_and_cfgs:
            rews = np.array(l['ep_sparse_rew_mean'])
            ep_rew_means.append(rews)
        ep_rew_means = np.array(ep_rew_means)

        ppo_updates_per_pairing = int(cfg['PPO_RUN_TOT_TIMESTEPS'] / cfg['TOTAL_BATCH_SIZE'])
        x_axis = list(range(0, log['num_ppo_runs'] * cfg['PPO_RUN_TOT_TIMESTEPS'], cfg['PPO_RUN_TOT_TIMESTEPS'] // ppo_updates_per_pairing))
        plt.figure(figsize=(7,4.5))
        if single:
            for i in range(len(logs_and_cfgs)):
                plt.plot(x_axis, ep_rew_means[i], label=str(i))
            plt.legend()
        else:
            seaborn.tsplot(time=x_axis, data=ep_rew_means)
        plt.xlabel("Environment timesteps")
        plt.ylabel("Mean episode reward")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.tight_layout()
        if save: plt.savefig("rew_pbt_" + layout, bbox_inches='tight')
        if show: plt.show()

def get_logs(save_dir, seeds, agent_array=None):
    """
    Get training logs across seeds for all PBT runs. By default take the logs for
    agent0, but can specify custom agents through the `agent_array` parameter.
    """
    save_dir = PBT_DATA_DIR + save_dir + "/"
    logs_across_seeds = []
    if agent_array is None:
        agent_array = [0] * len(seeds)
    for seed_idx, seed in enumerate(seeds):
        seed_log = load_dict_from_txt(save_dir + "seed_{}/agent{}/logs".format(seed, agent_array[seed_idx]))
        seed_cfg = load_dict_from_txt(save_dir + "config")
        logs_across_seeds.append((seed_log, seed_cfg))
    return logs_across_seeds

def get_pbt_agent_from_config(save_dir, sim_threads, seed, agent_idx=0, best=False, mode='mix'):
    agent_folder = save_dir + '/seed_{}/'.format(seed)
    if best:
        agent_to_load_path = agent_folder + "/best"
    else:
        agent_to_load_path = agent_folder + str(get_max_iter(agent_folder))
    print("agent_to_load_path : {}".format(agent_to_load_path))
    return agent_to_load_path

# Evaluation

def evaluate_all_pbt_models(pbt_model_paths, best_bc_model_paths, num_rounds, seeds, best=False, mode='mix'):
    pbt_performance = [defaultdict(lambda: defaultdict(list)),defaultdict(lambda: defaultdict(list))]
    for layout in pbt_model_paths.keys():
        print(layout)
        pbt_performance = evaluate_pbt_for_layout(layout, num_rounds, pbt_performance, pbt_model_paths, best_bc_model_paths['test'], seeds=seeds, best=best, mode=mode)
    return prepare_nested_default_dict_for_pickle(pbt_performance[0]),prepare_nested_default_dict_for_pickle(pbt_performance[1])

def evaluate_pbt_for_layout(layout_name, num_rounds, pbt_performance, pbt_model_paths, best_test_bc_models, seeds,mode, best=False):
    # bc_agent, bc_params = get_bc_agent_from_saved(model_name=best_test_bc_models[layout_name])
    bc_agent, bc_params = get_part_bc_agent_from_saved(model_name=best_test_bc_models[layout_name])
    ae = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    pbt_save_dir = os.path.join(PBT_DATA_DIR, pbt_model_paths[layout_name])
    print(pbt_save_dir)
    pbt_config = load_dict_from_txt(os.path.join(pbt_save_dir, "config"))
    assert common_keys_equal(bc_params["mdp_params"], pbt_config["mdp_params"]), "Mdp params differed between PBT and BC models training"
    assert common_keys_equal(bc_params["env_params"], pbt_config["env_params"]), "Env params differed between PBT and BC models training"

    pbt_agents = [get_pbt_agent_from_config(pbt_save_dir, pbt_config["sim_threads"], seed=seed, agent_idx=0, best=best, mode=mode) for seed in seeds]
    eval_pbt_over_seeds(pbt_agents, bc_agent, layout_name, num_rounds, pbt_performance, ae, mode=mode)
    return pbt_performance

def eval_pbt_over_seeds(pbt_agents, bc_agent, layout_name, num_rounds, pbt_performance, agent_evaluator, mode):
    ae = agent_evaluator
    for i in range(len(pbt_agents)):
        agent_to_load_path = pbt_agents[i]
        with open(os.path.join(agent_to_load_path, "shapley_info.txt"), 'r') as f:
            data = f.readlines()[0]
            prob_i = list(ast.literal_eval(data)['pos_sv'])
        if mode == 'mix' or mode == 'random':
            k = 5
        else:
            k = 1

        max1 = 0
        max2 = 0
        max3 = 0

        for i in tqdm.tqdm(range(k)):
            if mode == 'mix':
                i_idx = random.choices(list(range(len(prob_i))), weights=prob_i, k=1)[0]
                print("select {}th from {}".format(i_idx, prob_i))
            elif mode == 'pure':
                i_idx = np.argmax(prob_i)
            elif mode == 'random':
                i_idx = random.choices(list(range(len(prob_i))), k=1)[0]
                print("select {}th from random".format(i_idx))
            agent = get_agent_from_saved_model(os.path.join(agent_to_load_path,"pos_population", str(i_idx)), 50)
            pbt_and_pbt = ae.evaluate_agent_pair(AgentPair(agent, agent, allow_duplicate_agents=True), num_games=num_rounds)
            avg_pbt_and_pbt = np.mean(pbt_and_pbt['ep_returns'])
            pbt_performance[0][layout_name]["PBT+PBT"].append(avg_pbt_and_pbt)
            if avg_pbt_and_pbt > max1:
                max1 = avg_pbt_and_pbt

            pbt_and_bc = ae.evaluate_agent_pair(AgentPair(agent, bc_agent), num_games=num_rounds)
            avg_pbt_and_bc = np.mean(pbt_and_bc['ep_returns'])
            pbt_performance[0][layout_name]["PBT+BC_0"].append(avg_pbt_and_bc)
            if avg_pbt_and_bc > max2:
                max2 = avg_pbt_and_bc

            bc_and_pbt = ae.evaluate_agent_pair(AgentPair(bc_agent, agent), num_games=num_rounds)
            avg_bc_and_pbt = np.mean(bc_and_pbt['ep_returns'])
            pbt_performance[0][layout_name]["PBT+BC_1"].append(avg_bc_and_pbt)
            if avg_bc_and_pbt > max3:
                max3 = avg_bc_and_pbt

        pbt_performance[1][layout_name]["PBT+BC_1"].append(max3)
        pbt_performance[1][layout_name]["PBT+BC_0"].append(max2)
        pbt_performance[1][layout_name]["PBT+PBT"].append(max1)

    print(pbt_performance)
    return pbt_performance

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--node', type=int, required=True)
    args = parser.parse_args()
    print(args)
    if args.node == 7:
        seeds = [8015, 3554]
    elif args.node == 8:
        seeds = [5608, 581]
    else:
        seeds = [4221]
    num_rounds = args.num
    print(seeds)
    print(num_rounds)


    source_path = '/home/liyang/HARL/human_aware_rl/data/mixed_shapley_pbt_runs'
    target_path = '/home/liyang/HARL/human_aware_rl/data/cfs'
    for dir_name in os.listdir(source_path):
        if '2022' in dir_name:
            if 'unident_s' in dir_name:
                name = 'unident_s'
            else:
                name = dir_name.split('_')[-1]
            layout_path = os.path.join(source_path, dir_name)
            for seed_name in os.listdir(layout_path):
                if 'config' in seed_name:
                    copy_path = os.path.join(layout_path, seed_name)
                    to_path = os.path.join(target_path, name, seed_name)
                    os.makedirs(os.path.join(target_path, name), exist_ok=True)
                    if not os.path.exists(to_path):
                        shutil.copy(copy_path, to_path)
                if 'seed' in seed_name:
                    seed_path = os.path.join(layout_path, seed_name)
                    names = []
                    for model_name in os.listdir(seed_path):
                        if os.path.isdir(os.path.join(seed_path,model_name)):
                            if 'neg' not in model_name:
                                names += [int(model_name)]
                    names = sorted(names)
                    copy_name = str(names[-1])

                    copy_path = os.path.join(seed_path, copy_name)
                    test_dirs = os.listdir(copy_path)
                    if 'shapley_info.txt' not in test_dirs:
                        copy_name = str(names[-2])
                        copy_path = os.path.join(seed_path, copy_name)

                    to_path = os.path.join(target_path, name, seed_name,copy_name)
                    os.makedirs(to_path,exist_ok=True)
                    if os.path.exists(to_path):
                        shutil.rmtree(to_path)

                    shutil.copytree(copy_path, to_path)
                    print('copy from {} to {}'.format(copy_path, to_path))

    pbt_model_paths = {
        "unident_s": "unident_s",
        "simple": "simple",
        "random1": "random1",
        "random3": "random3",
        "random0": "random0"
    }
    modes = ['mix']
    for mode in modes:

        best_bc_model_paths = load_pickle("./data/bc_runs/best_bc_model_paths")

        # Evaluating
        set_global_seed(512)
        pbt_performance1, pbt_performance2 = evaluate_all_pbt_models(pbt_model_paths, best_bc_model_paths, num_rounds, seeds, best=False,
                                                  mode=mode)
        save_dict_to_file(pbt_performance1, PBT_DATA_DIR + "/mix_pbt_performance_" + mode + "_7")
        save_dict_to_file(pbt_performance2, PBT_DATA_DIR + "/max_pbt_performance_" + mode + "_7")
