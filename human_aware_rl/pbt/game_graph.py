import networkx as nx
import numpy as np
from human_aware_rl.pbt.utils import normalization, exp_prob

class GameGraph():
    """
    Game Graph class for conducting,visulizing,evaluating
    """
    def __init__(self, payoffs, coalitions):
        self.PG_adj_m = None #adjacent matrix of preference game graph
        self.PG = None
        self.build_graph(payoffs, coalitions)
        self.G_adj_m = payoffs
        self.G = None
        self.build_full_graph(payoffs, coalitions)

    def build_full_graph(self, payoffs, coalitions):
        N = len(coalitions)
        data = np.zeros((N, N))
        for i, index_1 in enumerate(coalitions):
            for j, index_2 in enumerate(coalitions):
                data[i, j] = payoffs[index_1, index_2]

        G = nx.DiGraph()
        for i in range(data.shape[0]):
            G.add_node(str(i))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                G.add_edge(str(i), str(j), weight=data[i, j])
        self.G = G

    def build_graph(self, payoffs, coalitions):
        N = len(coalitions)
        data = np.zeros((N, N))
        for i, index_1 in enumerate(coalitions):
            for j, index_2 in enumerate(coalitions):
                data[i, j] = payoffs[index_1, index_2]

        PG = np.zeros_like(data)
        for i in range(N):
            max_val = data[i].max()
            indexs = np.where(data[i] == max_val)
            j = indexs[0][-1]
            PG[i, j] = data[i, j]
        self.PG_adj_m = PG

        G = nx.DiGraph()
        for i in range(self.PG_adj_m.shape[0]):
            G.add_node(str(i))
        for i in range(self.PG_adj_m.shape[0]):
            for j in range(self.PG_adj_m.shape[0]):
                if self.PG_adj_m[i, j] > 0:
                    G.add_edge(str(i), str(j), weight=self.PG_adj_m[i, j])
        self.PG = G

        # print("==>conduct PG sucessfully !!! The adjacent matrix of preference game graph is")
        # print(self.PG_adj_m)

    def WPR(self):
        wpg_value = nx.pagerank(self.G)
        wpg_value = np.array(list(wpg_value.values()))

        return wpg_value

    def inv_WPR(self):
        wpg_value = nx.pagerank(self.G)
        wpg_value = np.array(list(wpg_value.values()))

        wpg_prob = exp_prob(wpg_value)

        in_wpg_value = -wpg_value
        in_wpg_prob = exp_prob(in_wpg_value, t=10)
        return wpg_value, wpg_prob, in_wpg_prob

    def eta(self):
        #################
        # Added by Yang Li
        # Aim to calculate preference centrality eta
        #
        #################
        eta = np.array(list(nx.in_degree_centrality(self.PG).values()))
        eta = 1 - eta
        return eta

