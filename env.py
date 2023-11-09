# %%
import networkx as nx
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, MultiBinary, Dict
import numpy as np

from utils import mc_lower_bound, decompose_graph,k_core_reduction

class GraphObservation(gym.Space):
    """
    A space of networkx graphs.
    """
    def __init__(
        self,
        node_space,
        edge_space,
        seed=None,
):
        self.node_space = node_space
        self.edge_space = edge_space
        self.seed(seed)
        
# %%
class VertexSeparatorEnv(gym.Env):
    """
    Description:
    A graph is picked randomly from a collection of graphs at each episode. The agent is asked to pick a vertex at each step. The episode ends when the picked vertices form a vertex separator of the graph. The reward is the number of nodes in the original graph minus that in the pruned subgraphs.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, graphs, **kwargs):
        super().__init__(**kwargs)
        self.graphs = graphs
        self.current_graph = graphs.pop()
        self.action_space = Discrete(n=len(self.current_graph))
        self.observation_space = Dict({
            'nodes': Box(0, 1, shape=(len(self.current_graph),), dtype=np.float32),
            'edges': Box(0, 1, shape=(len(self.current_graph),len(self.current_graph)), dtype=np.float32),
            'label': MultiBinary(len(graphs[0].nodes))
        })
        self.done = False
        self.truncated = False
        self.separator = set()
        
    def _is_separator(self):
        separator_list = list(self.separator)
        subgraph = self.current_graph.copy()
        subgraph.remove_nodes_from(separator_list)
        return not nx.is_connected(subgraph)
        
    def _get_obs(self):
        obs = np.zeros(len(self.current_graph))
        for node in self.separator:
            obs[node] = 1
        return obs
    
    def _calc_reward(self):
        if not self._is_separator():
            return -1
        else:
            self.done = True
            G1,G2 = decompose_graph(self.current_graph, self.separator)
            k = mc_lower_bound(self.current_graph)
            G1 = k_core_reduction(G1,len(k))
            G2 = k_core_reduction(G2,len(k))
            return len(self.current_graph)-len(G1)-len(G2)
            
        
    def reset(self):
        self.current_graph = np.random.choice(self.graphs)
        self.action_space = Discrete(n=len(self.current_graph))
        self.observation_space = Box(0, 1, shape=(len(self.current_graph),), dtype=np.float32)
        self.separator = set()
        self.done = False
        return self._get_obs()
        
    def step(self, action):
        if self.done:
            raise Exception('Episode is done. Call reset() to start a new episode.')
        node = list(self.current_graph.nodes)[action]
        self.separator.add(node)
        reward = self._calc_reward()
        return self._get_obs(), reward, self.done, self.truncated, {}
    
# create a random networkx graph
