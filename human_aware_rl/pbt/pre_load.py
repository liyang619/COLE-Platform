import os


def pre_load(params):
    init_population_size = params['init_pop_size']

    load_root_dirs = ['pop_entropy', 'sp']
    assert init_population_size % (3+2) == 0
    each_load_num = 3
    LOAD_FOLDER_LST = []

    load_root_dir = load_root_dirs[0]
    print("Now we load agents name from {}".format(load_root_dir))
    layout_path = os.path.join(params['LOAD_FOLDER_PATH'], load_root_dir, params["mdp_params"]['layout_name'])
    seed_dirs = [i for i in os.listdir(layout_path) if os.path.isdir(os.path.join(layout_path, i))]
    seed_dir = seed_dirs[0]
    seed_path = os.path.join(layout_path, seed_dir)
    for i in range(each_load_num):
        agent_path = os.path.join(seed_path, "agent" +str(i))
        LOAD_FOLDER_LST.append(os.path.join(agent_path, "best/"))
        LOAD_FOLDER_LST.append(os.path.join(agent_path, "pbt_iter1/"))
        # LOAD_FOLDER_LST.append(os.path.join(agent_path, "pbt_iter152/"))
    print("Now we have loaded {} agents name from {}".format(len(LOAD_FOLDER_LST), load_root_dir))

    load_root_dir = load_root_dirs[1]
    print("Now we load agents name from {}".format(load_root_dir))
    layout_path = os.path.join(params['LOAD_FOLDER_PATH'], load_root_dir, params["mdp_params"]['layout_name'])
    seed_dirs = [i for i in os.listdir(layout_path) if os.path.isdir(os.path.join(layout_path, i))][:each_load_num]
    for seed_dir in seed_dirs:
        seed_path = os.path.join(layout_path, seed_dir)

        best_rew = 0
        best_ckpt_path = None
        # print(seed_path)
        for ckpt_path in os.listdir(seed_path):
            # print(ckpt_path)
            pos = ckpt_path.find("reward=")
            if pos < 0:
                continue
            ckpt_rew = float(ckpt_path[pos + len("reward="):])
            if ckpt_rew > best_rew:
                best_rew = ckpt_rew
                best_ckpt_path = ckpt_path + '/'
        LOAD_FOLDER_LST.append(os.path.join(seed_path, best_ckpt_path))

        # find mid-tier agent
        min_diff = 1e10
        mid_rew = best_rew / 2
        mid_ckpt_path = None
        for ckpt_path in os.listdir(seed_path):
            pos = ckpt_path.find("reward=")
            if pos < 0:
                continue
            ckpt_rew = float(ckpt_path[pos + len("reward="):])
            if abs(ckpt_rew - mid_rew) < min_diff:
                min_diff = abs(ckpt_rew - mid_rew)
                mid_ckpt_path = ckpt_path + '/'
        LOAD_FOLDER_LST.append(os.path.join(seed_path, mid_ckpt_path))

        worst_ckpt_path = None
        for ckpt_path in os.listdir(seed_path):
            if ckpt_path.startswith("1_"):
                worst_ckpt_path = ckpt_path + '/'
                break
        LOAD_FOLDER_LST.append(os.path.join(seed_path, worst_ckpt_path))


    print(f"population_size {init_population_size} len(params['LOAD_FOLDER_PATH']) {len(params['LOAD_FOLDER_PATH'])}")
    for i ,j in enumerate(LOAD_FOLDER_LST):
        print('{} agent from {} .......'.format(i +1, j))

    return LOAD_FOLDER_LST
