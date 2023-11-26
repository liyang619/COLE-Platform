import os
from torch.utils.tensorboard import SummaryWriter
import ast
import shutil
import tqdm
import numpy as np

mode = "remote"

names = ['avg_rew_per_step', 'reward_shaping', 'eprewmean', 'ep_dense_rew_mean',
                             'ep_sparse_rew_mean', 'eplenmean', 'explained_variance', 'policy_loss', 'value_loss',
                             'policy_entropy', 'approxkl', 'clipfrac']
name = "events"
if mode == "remote":
    root_path = "./data/pbt_runs/"

    layout_dirs = [l for l in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,l))]
    print(layout_dirs)
    # if os.path.exists('./data/{}'.format(name)):
    #     shutil.rmtree('./data/{}'.format(name))
    #     print("rm folder {}".format('./data/{}'.format(name)))

    for layout_dir in tqdm.tqdm(layout_dirs):
        seed_dirs = [l for l in os.listdir(os.path.join(root_path,layout_dir)) if "seed_" in l]
        for seed_dir in seed_dirs:
            log_path = os.path.join(root_path, layout_dir, seed_dir, 'agent0')

            log_dir = os.path.join(log_path, 'logs.txt')
            log_writer = SummaryWriter(os.path.join(log_path, 'events'))
            try:
                with open(log_dir, 'r') as f:
                    data = f.readlines()[0]
                    if "defaultdict(<class 'list'>, {})" in data:
                        data = data.replace("defaultdict(<class 'list'>, {})", "[]")
                    if "nan" in data:
                        data = data.replace("nan", "65536")
                    data = data.replace("array(","")
                    data = data.replace(")","")
                    data = ast.literal_eval(data)

                    for k, v in data.items():
                        if k != "agent_name" and k != 'ent_pool_coef' and isinstance(v, list):
                            count_dict = 0
                            for num in v:
                                if isinstance(num, list):
                                    log_writer.add_scalar('{}'.format(k), float(np.mean(num)), count_dict)
                                else:
                                    log_writer.add_scalar('{}'.format(k), float(num), count_dict)
                                count_dict += 1
            except Exception as e:
                print('sdfadsfda')
                print(log_path)
                print(e)
                # print(k)
            log_writer.close()
