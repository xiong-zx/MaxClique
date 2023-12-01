import random
import numpy as np
import networkx as nx
from copy import deepcopy
from DBK import mc_upper_bound, mc_lower_bound, k_core_reduction, \
    edge_k_core, is_clique, remove_zero_degree_nodes, ch_partitioning

random.seed(392)
np.random.seed(483)


class GraphMDP:

    def __init__(self, graph, LIMIT, solver_function):

        assert type(graph) is nx.Graph
        assert type(LIMIT) is int
        assert len(graph) != 0

        self.G0 = graph.copy()

        graph = remove_zero_degree_nodes(graph)
        self.k = mc_lower_bound(graph)
        graph = k_core_reduction(graph, len(self.k))

        if len(graph) != 0 and len(graph) <= LIMIT:
            self.k = solver_function(graph)

        self.G_init = graph  # .copy()

        self.decomposition_cnt = 0
        self.LIMIT = LIMIT
        self.solver_function = solver_function

        self.vertex_removal = {graph: []}
        self.subgraphs = [graph]
        self.last_subgraph = graph

    """ Return all actions with non-zero probability from this state """

    def get_actions(self):
        if len(self.subgraphs) == 0:
            return []
        state = self.subgraphs[-1]  # because we use pop() in execute()
        return list(state.nodes)

    def is_terminal(self):
        for graph in self.subgraphs:
            if len(graph) > self.LIMIT:
                return False
        return True
        # Todo: remove the following code
        # if len(state) == 0:
        #     return True
        # elif 0 < len(state) <= self.LIMIT:
        #     self.k = max(self.solver_function(state), self.k)
        #     return True
        # return False

    """ Return the discount factor for this MDP """

    def get_discount_factor(self):
        return 0.9

    """ Return the initial state of this MDP """

    def get_initial_state(self):
        return self.G_init

    def copy(self):
        return deepcopy(self)

    """ Return a new state and a reward for executing action in state
    """

    def execute(self, action):

        SG = self.subgraphs.pop()
        SG = remove_zero_degree_nodes(SG)
        assert len(SG) != 0
        vcount = self.vertex_removal[SG]
        del self.vertex_removal[SG]
        vertex = action

        self.last_subgraph = SG.copy()
        n = len(SG)
        bi = 0  # bound improvement
        parent = SG.copy()

        # Partitioning Subgraph
        SSG, SG = ch_partitioning(vertex, SG)
        self.decomposition_cnt += 1
        SG = remove_zero_degree_nodes(SG)
        SSG = remove_zero_degree_nodes(SSG)
        SG = k_core_reduction(SG, len(self.k) - len(vcount))
        SSG = k_core_reduction(SSG, len(self.k) - len(vcount + [vertex]))
        self.vertex_removal[SSG] = vcount + [vertex]
        self.vertex_removal[SG] = vcount

        #####################################################################################################
        if len(SSG) != 0:
            SSG_lower = mc_lower_bound(SSG) + self.vertex_removal[SSG]
            assert is_clique(self.G0.subgraph(SSG_lower)) == True
            if len(SSG_lower) > len(self.k):
                bi += len(SSG_lower) - len(self.k)
                vcount = self.vertex_removal[SSG]
                del self.vertex_removal[SSG]
                self.k = SSG_lower
                SSG = k_core_reduction(SSG, len(self.k) - len(vcount))
                SSG = remove_zero_degree_nodes(SSG)
                self.vertex_removal[SSG] = vcount
            if len(SSG) != 0:
                SSG_upper = mc_upper_bound(SSG) + len(self.vertex_removal[SSG])
                if SSG_upper > len(self.k):
                    if len(SSG) <= self.LIMIT:
                        # print("=== Calling Solver Function ===")
                        sub_solution_SSG = self.solver_function(SSG) + self.vertex_removal[SSG]
                        del self.vertex_removal[SSG]
                        assert is_clique(self.G0.subgraph(sub_solution_SSG)) == True
                        if len(sub_solution_SSG) > len(self.k):
                            bi += len(sub_solution_SSG) - len(self.k)
                            self.k = sub_solution_SSG
                    else:
                        self.subgraphs.append(SSG)
                else:
                    del self.vertex_removal[SSG]
        if len(SSG) == 0:
            if SSG in list(self.vertex_removal.keys()):
                sub_solution_SSG = self.vertex_removal[SSG]
                del self.vertex_removal[SSG]
                assert is_clique(self.G0.subgraph(sub_solution_SSG)) == True
                if len(sub_solution_SSG) > len(self.k):
                    bi += len(sub_solution_SSG) - len(self.k)
                    self.k = sub_solution_SSG
        #####################################################################################################
        if len(SG) != 0:
            SG_lower = mc_lower_bound(SG) + self.vertex_removal[SG]
            assert is_clique(self.G0.subgraph(SG_lower)) == True
            if len(SG_lower) > len(self.k):
                bi += len(SG_lower) - len(self.k)
                vcount = self.vertex_removal[SG]
                del self.vertex_removal[SG]
                self.k = SG_lower
                SG = k_core_reduction(SG, len(self.k) - len(vcount))
                SG = remove_zero_degree_nodes(SG)
                self.vertex_removal[SG] = vcount
            if len(SG) != 0:
                SG_upper = mc_upper_bound(SG) + len(self.vertex_removal[SG])
                if SG_upper > len(self.k):
                    if len(SG) <= self.LIMIT:
                        # print("=== Calling Solver Function ===")
                        sub_solution_SG = self.solver_function(SG) + self.vertex_removal[SG]
                        del self.vertex_removal[SG]
                        assert is_clique(self.G0.subgraph(sub_solution_SG)) == True
                        if len(sub_solution_SG) > len(self.k):
                            bi += len(sub_solution_SG) - len(self.k)
                            self.k = sub_solution_SG
                    else:
                        self.subgraphs.append(SG)
                else:
                    del self.vertex_removal[SG]
        if len(SG) == 0:
            if SG in list(self.vertex_removal.keys()):
                sub_solution_SG = self.vertex_removal[SG]
                del self.vertex_removal[SG]
                assert is_clique(self.G0.subgraph(sub_solution_SG)) == True
                if len(sub_solution_SG) > len(self.k):
                    bi += len(sub_solution_SG) - len(self.k)
                    self.k = sub_solution_SG

        x = len(SSG)
        y = len(SG)
        l = self.LIMIT
        # Todo: make sure to delete the next line
        new_states = [(graph, parent) for graph in [SG, SSG]]  # if graph in self.vertex_removal.keys()]
        # reward = np.exp((2 * n - (x + y)) / (2 * n)) - np.exp((x + y - 2 * self.LIMIT) / (2 * self.LIMIT))
        reward = np.exp(1 - (x / n)) + np.exp(1 - (y / n)) - np.exp((x / l) - 1) - \
                 np.exp((y / l) - 1) - np.exp((n / l) - 1) - self.decomposition_cnt  # + np.exp(bi / n)
        # Some questions:
        # Is this the only reward function? No!
        # Is this the best reward function? Idk!
        # Will it work? I think so!

        return reward  # Todo: make sure this is compatible with rest of the code

    # """ Execute a policy on this mdp for a number of episodes """
    #
    # def execute_policy(self, policy, episodes=100):
    #     for _ in range(episodes):
    #         state = self.get_initial_state()
    #         while not self.is_terminal(state):
    #             action = policy.select_action(state)  # Todo: select max action in GCN
    #             # Todo: in your mdp policy execution chanages because you have two next states
    #             (next_state, reward) = self.execute(state, action)
    #             state = next_state
