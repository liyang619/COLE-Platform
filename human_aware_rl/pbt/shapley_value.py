import numpy as np
import time
from human_aware_rl.pbt.game_graph import GameGraph
from human_aware_rl.pbt.utils import normalization


def c_value(coalition, payoffs):
    if len(coalition) == 0:
        return 0
    values = 0

    G = GameGraph(payoffs, coalition)
    _, _, sigma = G.inv_WPR()

    for i, agent_i in enumerate(coalition):
        for j, agent_j in enumerate(coalition):
            value = (1/len(coalition))*sigma[j]*payoffs[agent_i, agent_j]
            values += value
    if values < 0:
        return 0
    else:
        return values


def Shapley_Value(payoffs, grand_coalition, mc_times=200):
    #################
    # Added by Yang Li
    # Aim to calculate shapley value
    # input : grand_coalition is the index list
    # return normalized shapley value
    #################
    start_time = time.time()
    N = len(grand_coalition) 
    SV = [0] * N

    for k in range(mc_times):
        pi = list(np.random.choice(a=N, size=N, replace=False, p=None))
        for i in grand_coalition:
            j = pi.index(i)
            S_pi = pi[:j]
            S_pi_i = pi[:j + 1]
            tmp = c_value(S_pi_i, payoffs) - c_value(S_pi, payoffs)
            if tmp < 0:
                tmp = 0
            SV[i] += 1 / mc_times * (tmp)

    print("MC based Shapley value algorithm costs {:.3f} seconds.".format(time.time() - start_time))
    return SV


if __name__=="__main__":
    import pandas as pd
    payoffs = np.array(pd.read_csv("/home/liyang/HARL/human_aware_rl/data/game_shapley/pbt_random1/seed_4221/pos_agent0/pbt_iter17/neg_payoffs.csv", header=None))
    coalitions = list(range(10))
    a = payoffs[34-10:34,34-10:34]
    print(a)
    x = Shapley_Value(a, coalitions)
    print(x)
    from utils import exp_prob
    x = normalization(x)
    x = exp_prob(-x, t=10)
    print(x)