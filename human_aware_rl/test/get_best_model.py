#%%
import os
import ast

import numpy as np
from overcooked_ai_py.utils import save_pickle, load_pickle, load_dict_from_txt,save_dict_to_file
from human_aware_rl.pbt.game_graph import GameGraph
import pandas as pd
PBT_DATA_DIR = "PATH_TO_DATA_DIR"
mode = 2
def prepare_nested_default_dict_for_pickle(nested_defaultdict):
    """Need to make all nested defaultdicts into normal dicts to pickle"""
    for k,v in nested_defaultdict.items():
        nested_defaultdict[k] = dict(v)
    pickleable_dict = dict(nested_defaultdict)
    return pickleable_dict

from collections import defaultdict
layout_dirs = [l for l in os.listdir(PBT_DATA_DIR) if os.path.isdir(os.path.join(PBT_DATA_DIR,l))]
print(layout_dirs)
res={
    'pbt_simple': defaultdict(list),
    'pbt_unident_s': defaultdict(list),
    'pbt_random1': defaultdict(list),
    'pbt_random0': defaultdict(list),
    'pbt_random3': defaultdict(list)
}

stop = {
    'pbt_simple': 80,
    'pbt_unident_s': 100,
    'pbt_random1': 60,
    'pbt_random0': 60,
    'pbt_random3': 100
}

for layout_dir in layout_dirs:
    seed_dirs = [l for l in os.listdir(os.path.join(PBT_DATA_DIR,layout_dir)) if "seed_" in l]
    for seed_dir in seed_dirs:
        log_path = os.path.join(PBT_DATA_DIR, layout_dir, seed_dir, 'ego_agent/')
        try:
            dirs = [int(i.replace("pbt_iter","")) for i in os.listdir(log_path) if "pbt_iter" in i]
        except:
            continue
        dirs = sorted(dirs)
        for i in range(len(dirs)):
            if dirs[i] > stop[layout_dir]:
                break
        dirs = dirs[:i]
        # print(dirs)
        latest_name = str(dirs[-1])
        pop_info_path = os.path.join(log_path, "pbt_iter"+latest_name, "main_payoffs.csv")
        payoff = np.array(pd.read_csv(os.path.join(pop_info_path), header=None))
        N = payoff.shape[0]
        max_0 = 0
        max_1 = 0
        pos = 0
        if 'random1' in layout_dir and 'random0' in layout_dir:
            for i in range(N):
                payoff[i][i]=0 #the total generation is just 60, and the population size is 50, thus, we need to minus the self-player score.

        for i in range(N):
            s_0 = np.mean(payoff[i, :10]) 
            s_1 = np.mean(payoff[:10, i])
            if 'random0' in layout_dir:
                if s_0 > max_0 * 0.85 and s_1 > max_1:
                    max_0 = s_0
                    max_1 = s_1
                    pos=i
            else:
                if s_0 > max_0 and s_1 > max_1:
                    max_0 = s_0
                    max_1 = s_1
                    pos=i
        pos = len(dirs)-N+pos
        res[layout_dir][seed_dir] = [pos, max_0, max_1, N]
        print("%{} with seed {} : {}, {} and pos {}-{}/{}-{}".format(layout_dir, seed_dir, max_0, max_1,pos, len(dirs) ,dirs[pos],dirs[-1]))

save_dict_to_file(prepare_nested_default_dict_for_pickle(res), PBT_DATA_DIR + "/best_info")