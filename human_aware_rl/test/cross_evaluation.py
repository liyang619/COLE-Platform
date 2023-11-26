import time, gym, copy, seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from overcooked_ai_py.utils import save_pickle, load_pickle, load_dict_from_txt
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.utils import set_global_seed, prepare_nested_default_dict_for_pickle, common_keys_equal
from human_aware_rl.baselines_utils import get_pbt_agent_from_config, get_agent_from_saved_model

from overcooked_ai_py.utils import save_pickle, load_pickle, save_dict_to_file
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, StayAgent, RandomMoveAgent

from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved, get_part_bc_agent_from_saved, \
    get_none_bc_agent_from_saved
from human_aware_rl.utils import reset_tf, set_global_seed, prepare_nested_default_dict_for_pickle
from human_aware_rl.ppo.ppo import get_ppo_agent, plot_ppo_run, PPO_DATA_DIR
# Visualization
from tqdm import tqdm
import os
import argparse

print("mixed_agents_experiments.py: done import")
FCP_LOAD_IGNORE_IDX = True

PBT_DATA_DIR = "PATH_TO_YOUR_OTHER_MODEL"
PBT_DATA_DIR_OUR = "PATH_TO_YOUR_MODEL"


def get_shapley_agent_from_config(save_dir, sim_threads, seed,name="ego_agent", best=False):
    agent_folder = save_dir + 'seed_{}'.format(seed)
    agent_to_load_path = agent_folder + "/best"
    agent = get_agent_from_saved_model(agent_to_load_path, sim_threads)
    return agent

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
        x_axis = list(range(0, log['num_ppo_runs'] * cfg['PPO_RUN_TOT_TIMESTEPS'],
                            cfg['PPO_RUN_TOT_TIMESTEPS'] // ppo_updates_per_pairing))
        plt.figure(figsize=(7, 4.5))

        if single:
            for i in range(len(logs_and_cfgs)):
                plt.plot(x_axis, ep_rew_means[i], label=str(i))
            plt.legend()
        else:
            seaborn.tsplot(time=x_axis, data=ep_rew_means)
        plt.xlabel("Environment timesteps")
        plt.ylabel("Mean episode reward")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
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


# Evaluation

def evaluate_all_pbt_models(layouts, ego_algo, model_paths, best_bc_model_paths, num_rounds, seeds, best=False):
    performance = defaultdict(lambda: defaultdict(list))
    for layout in layouts:
        print(layout)
        performance = evaluate_pbt_for_layout(layout, ego_algo, num_rounds, performance, model_paths,
                                              best_bc_model_paths['test'], seeds=seeds, best=best)
    return prepare_nested_default_dict_for_pickle(performance)


def evaluate_pbt_for_layout(layout_name, ego_algo, num_rounds, performance, model_paths, best_test_bc_models, seeds,
                            best=False):
    # ego_agents = load_egos(layout_name, ego_algo, model_paths, seeds, best)
    print("load partners")
    partner_agents, ae = load_partners(layout_name, model_paths, best_test_bc_models, seeds, best)

    ego_agents = partner_agents.pop("{}_agents".format(ego_algo))
    performance = eval_pbt_over_seeds(ego_agents, partner_agents, layout_name, num_rounds, performance, ae, seeds[ego_algo])
    return performance


def load_pbt_based_agents(layout_name, algo_name, model_paths, seeds, best=False):
    agents = []
    if algo_name == "shapley":
        save_dir = PBT_DATA_DIR_OUR + model_paths[algo_name][layout_name] + "/"
    else:
        save_dir = PBT_DATA_DIR + model_paths[algo_name][layout_name] + "/"
    config = load_dict_from_txt(save_dir + "config")
    agent_idx = -1 if algo_name == 'shapley' and FCP_LOAD_IGNORE_IDX else 0  # -1: no agent_idx given in directory. (default is 0)
    print(f"Loading {algo_name} agents {agent_idx}...")
    time.sleep(5)
    if algo_name == "shapley":
        for seed in tqdm(seeds):
            agents.append(
                get_shapley_agent_from_config(save_dir, config["sim_threads"], seed=seed, best=best))
    else:
        for seed in tqdm(seeds):
            agents.append(
                    get_pbt_agent_from_config(save_dir, config["sim_threads"], seed=seed, agent_idx=agent_idx, best=best))
    return agents


def load_egos(layout_name, ego_algo, model_paths, seeds, best=False):
    # load our algorithm
    ego_agents = []
    if ego_algo == 'sp':
        for seed in tqdm(seeds['sp']):
            sp_agent, _ = get_ppo_agent(model_paths['sp'][layout_name], seed, best=best)
            ego_agents.append(sp_agent)
    elif ego_algo in ["pbt", "mep", "fcp", "shapley"]:
        ego_agents = load_pbt_based_agents(layout_name, ego_algo, model_paths, seeds[ego_algo], best)
    else:
        raise NotImplementedError()

    return ego_agents


def load_partners(layout_name, model_paths, best_test_bc_models, seeds: dict, best=False):
    agents = {
        "bc_agents": dict(),
        "sp_agents": [],
        "pbt_agents": [],
        "mep_agents": [],
        "fcp_agents": [],
        "shapley_agents": [],
    }
    # BC
    agent_bc_test, bc_params = get_bc_agent_from_saved(model_name=best_test_bc_models[layout_name])
    ae = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])
    agents['bc_agents'] = {
        'bc_best': agent_bc_test,
    }
    # shapley
    agents['shapley_agents'] = load_pbt_based_agents(layout_name, 'shapley', model_paths, seeds['shapley'], best)

    # PPO SP
    for seed in tqdm(seeds['sp']):
        sp_agent, _ = get_ppo_agent(model_paths['sp'][layout_name], seed, best=best)
        agents['sp_agents'].append(sp_agent)

    # PBT
    agents['pbt_agents'] = load_pbt_based_agents(layout_name, 'pbt', model_paths, seeds['pbt'], best)

    # MEP
    agents['mep_agents'] = load_pbt_based_agents(layout_name, 'mep', model_paths, seeds['mep'], best)

    # FCP
    agents['fcp_agents'] = load_pbt_based_agents(layout_name, 'fcp', model_paths, seeds['fcp'], best)

    return agents, ae


def eval_pbt_over_seeds(eval_agents, partner_agents, layout_name, num_rounds, performance, agent_evaluator, seeds,
                        display=False):
    ae = agent_evaluator
    partner_algos = partner_agents.keys()
    for i in range(len(eval_agents)):
        seed_res = defaultdict(lambda: defaultdict(list))
        for algo in partner_algos:
            partners = partner_agents[algo]
            if algo == "bc_agents":
                for bc_agent_name, bc_agent in partners.items():
                    for position_swapped in (0, 1):  # 0: don't swap, 1: swap
                        print(
                            f'Evaluate ego agent (seed={seeds[i]}) with {bc_agent_name}, {"swapped" if position_swapped else "not swapped"}.')
                        agent_pair = AgentPair(bc_agent, eval_agents[i], allow_duplicate_agents=True) if position_swapped else AgentPair(
                            eval_agents[i], bc_agent, allow_duplicate_agents=True)
                        rollouts = ae.evaluate_agent_pair(agent_pair, num_games=num_rounds, display=display)
                        avg_rew = np.mean(rollouts['ep_returns'])
                        performance[layout_name][f"{algo}({bc_agent_name})_{position_swapped}"].append(avg_rew)
                        seed_res[seeds[i]][f"{algo}({bc_agent_name})_{position_swapped}"].append(avg_rew)
            else:
                for j, partner_agent in enumerate(partners):
                    for position_swapped in (0, 1):  # 0: don't swap, 1: swap
                        print(
                            f'Evaluate ego agent (seed={seeds[i]}) with {algo}_{j}, {"swapped" if position_swapped else "not swapped"}.')
                        agent_pair = AgentPair(partner_agent, eval_agents[i], allow_duplicate_agents=True) if position_swapped else AgentPair(
                            eval_agents[i], partner_agent, allow_duplicate_agents=True)
                        rollouts = ae.evaluate_agent_pair(agent_pair, num_games=num_rounds, display=display)
                        avg_rew = np.mean(rollouts['ep_returns'])
                        performance[layout_name][f"{algo}_{position_swapped}"].append(avg_rew)
                        seed_res[seeds[i]][f"{algo}({bc_agent_name})_{position_swapped}"].append(avg_rew)
        save_dict_to_file(prepare_nested_default_dict_for_pickle(seed_res), PBT_DATA_DIR_OUR + f"{layout_name}_{seeds[i]}_cross_play")
        print(seeds[i],seed_res)

    return performance


def run_all_pbt_experiments(best_bc_model_paths, layouts, shapley_seeds):
    # layouts = ["random1"]
    ego_algo = 'shapley'
    exp_name = f"Mixed_{'_'.join(layouts)}_{ego_algo}"

    # NAME of MODELs
    shapley_model_paths = {
        "simple": "pbt_simple",
        "unident_s": "pbt_unident_s",
        "random1": "pbt_random1",
        "random3": "pbt_random3",
        "random0": "pbt_random0"
    }

    sp_model_paths = {
        "simple": "ppo_sp_simple",
        "unident_s": "ppo_sp_unident_s",
        "random1": "ppo_sp_random1",
        "random3": "ppo_sp_random3",
        "random0": "ppo_sp_random0"
    }
    sp_seeds = [TBD]

    pbt_model_paths = {
        "simple": "pbt_runs/pbt_simple",
        "unident_s": "pbt_runs/pbt_unident_s",
        "random1": "pbt_runs/pbt_random1",
        "random3": "pbt_runs/pbt_random3",
        "random0": "pbt_runs/pbt_random0"
    }

    pbt_seeds = [TBD]
    if 'simple' in layouts:
        print("simple!!")
        pbt_seeds = [TBD]
    # MEP
    mep_model_paths = {
        "simple": "mep_runs/pbt_simple",
        "unident_s": "mep_runs/pbt_unident_s",
        "random1": "mep_runs/pbt_random1",
        "random3": "mep_runs/pbt_random3",
        "random0": "mep_runs/pbt_random0"
    }
    mep_seeds = [TBD]

    # ----- FCP ------ #
    fcp_model_paths = {
        "simple": "fcp_runs/FCP_simple_5_new",
        "unident_s": "fcp_runs/FCP_unident_s_5_new",
        "random1": "fcp_runs/FCP_random1_5_new",
        "random3": "fcp_runs/FCP_random3_5_new",
        "random0": "fcp_runs/FCP_random0_5_new"
    }
    fcp_seeds = [TBD]

    model_paths = {
        "sp": sp_model_paths,
        "pbt": pbt_model_paths,
        "mep": mep_model_paths,
        "fcp": fcp_model_paths,
        "shapley": shapley_model_paths,
    }
    seeds_all = {
        "sp": sp_seeds,
        "pbt": pbt_seeds,
        "mep": mep_seeds,
        "fcp": fcp_seeds,
        "shapley": shapley_seeds,
    }

    use_best = True

    # Plotting
    # plot_pbt_runs(pbt_model_paths, seeds, save=True)

    # Evaluating
    set_global_seed(512)
    num_rounds = 5

    performance = evaluate_all_pbt_models(layouts, ego_algo, model_paths, best_bc_model_paths, num_rounds, seeds_all,
                                          best=use_best)
    performance['seeds'] = seeds_all
    performance['num_rounds'] = num_rounds
    # save_pickle(performance, PBT_DATA_DIR + f"{exp_name}_{num_rounds}rounds_pbt_performance")
    save_dict_to_file(performance, PBT_DATA_DIR_OUR + f"{exp_name}_{num_rounds}rounds_pbt_performance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env', type=str, default=None)
    args = parser.parse_args()
    print(args)
    '''
    The seed and env codes are based on assumption that all layouts use same seeds.
    Please Replace it if you use different optimal seeds.
    '''
    if args.env == None:
        layouts = ["simple", "unident_s","random1","random0","random3"]

        path = os.path.join("{}{}".format(PBT_DATA_DIR_OUR, "pbt_simple"))
    elif args.env == "simple" or args.env == "unident_s":
        layouts = [args.env]
        path = os.path.join("{}{}".format(PBT_DATA_DIR_OUR, "pbt_{}".format(args.env)))
    else:
        print("not implement")
        exit()

    seeds = []
    for seed in os.listdir(path):
        if os.path.isdir(os.path.join(path, seed)):
            seeds += [int(seed.split('_')[1])]
    print(seeds)
    print(layouts)

    start_time = time.time()
    best_bc_models_paths = load_pickle("data/bc_runs/best_bc_model_paths.pickle")
    print(best_bc_models_paths)
    run_all_pbt_experiments(best_bc_models_paths, layouts, seeds)
    print(f"took {(time.time() - start_time) / 60:.2f} mins.")
