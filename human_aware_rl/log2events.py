    """
    THE code convert the log info to tensorboard event.
    """
import os
from torch.utils.tensorboard import SummaryWriter
import ast
import shutil
import tqdm
import numpy as np

model_name = "MODE_NAME_OF_YOU"
PBT_DATA_DIR = f"PATH_TO_YOUR_SAVE/{model_name}/" 
mode = "remote"

name = "events"
if mode == "remote":
    layout_dirs = [l for l in os.listdir(PBT_DATA_DIR) if os.path.isdir(os.path.join(PBT_DATA_DIR,l))]
    print(layout_dirs)
    if os.path.exists('./data/{}'.format(name)):
        shutil.rmtree('./data/{}'.format(name))
        print("rm folder {}".format('./data/{}'.format(name)))

    for layout_dir in tqdm.tqdm(layout_dirs):
        seed_dirs = [l for l in os.listdir(os.path.join(PBT_DATA_DIR,layout_dir)) if "seed_" in l]
        for seed_dir in seed_dirs:
            log_path = os.path.join(PBT_DATA_DIR, layout_dir, seed_dir, 'ego_agent/')
            log_path = os.path.join(log_path, "logs.txt")
            log_dir = log_path.replace(model_name, '{}'.format(name))
            log_writer = SummaryWriter(log_dir)
            try:
                with open(log_path, 'r') as f:
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
