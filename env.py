# %%
import networkx as nx
import gymnasium as gym
from gymnasium.spaces import Box, MultiBinary, Dict
import numpy as np
import os
from node2vec import Node2Vec

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from utils import mc_lower_bound, decompose_graph, k_core_reduction

MAX_NODES = 200
COMPLETION_REWARD = 10
STEP_PENALTY = -0.01
SEPARATOR_PENALTY = -0.01
MASK_PENALTY = -10

# %%
class VertexSeparatorEnv(gym.Env):
    """
    Description:
    A graph is picked randomly from a collection of graphs at each episode. The agent is asked to pick a vertex at each step. The episode ends when the picked vertices form a vertex separator of the graph. The reward is the number of nodes in the original graph minus that in the pruned subgraphs.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            graphs: list[str],
            node_embedding_dim: int = 4,
            **kwargs):
        super().__init__(**kwargs)
        self.graphs = graphs
        np.random.shuffle(self.graphs)
        self.current_graph = nx.read_edgelist(graphs.pop())
        self.node_embedding_dim = node_embedding_dim
        self.action_space = MultiBinary(MAX_NODES)

        self.observation_space = Dict({
            'node_embeddings': Box(low=-np.inf, high=np.inf, shape=(MAX_NODES, node_embedding_dim), dtype=np.float32),
            'adjacency_matrix': Box(low=0, high=1, shape=(MAX_NODES, MAX_NODES), dtype=np.int8),
            'node_labels': MultiBinary(MAX_NODES)
        })

        self.done = False
        self.truncated = False
        self.separator = set()
        self.steps = 0

    def _is_separator(self):
        separator_list = list(self.separator)
        subgraph = self.current_graph.copy()
        subgraph.remove_nodes_from(separator_list)
        return not nx.is_connected(subgraph)

    def _get_obs(self):
        adjacency_matrix = self.current_graph.graph['adjacency_matrix']

        node_embeddings = np.zeros(
            (MAX_NODES, self.node_embedding_dim), dtype=np.float32)
        num_nodes = len(self.current_graph)
        node_embeddings[:num_nodes, :] = [
            self.current_graph.nodes[node]['node_embeddings'] for node in self.current_graph]

        node_labels = np.zeros(MAX_NODES, dtype=np.int8)
        node_labels[:num_nodes] = [
            1 if node in self.separator else 0 for node in self.current_graph.nodes()]

        return {
            'node_embeddings': node_embeddings,
            'adjacency_matrix': adjacency_matrix,
            'node_labels': node_labels
        }

    def _calc_reward(self, action):
        reward = 0
        
        if self.action_mask[action] == 0:
            reward += MASK_PENALTY
            # self.truncated = True
        else:
            node = list(self.current_graph.nodes)[action]
            self.separator.add(node)
            self.action_mask[action] = 0
            
        if not self._is_separator():
            reward += STEP_PENALTY
        else:
            self.done = True
            print(f'Episode done in {self.steps} steps.')  # todo: remove
            print(f'Separator size: {len(self.separator)}')  # todo: remove
            self.steps = 0
            k = mc_lower_bound(self.current_graph)
            decomposed_graphs = decompose_graph(
                self.current_graph, self.separator)
            decomposed_graphs = [k_core_reduction(
                g, len(k)) for g in decomposed_graphs]
            node_reduced = len(self.current_graph) - sum([len(g) for g in decomposed_graphs])
            print(f'Number of node reduced: {node_reduced}')  # todo: remove
            reward += COMPLETION_REWARD + node_reduced

        reward += SEPARATOR_PENALTY * len(self.separator)
        return reward

    def reset(self, seed=None):
        self.current_graph = nx.read_edgelist(self.graphs.pop())
        num_nodes = len(self.current_graph)

        actual_nodelist = list(self.current_graph.nodes())
        edges = nx.to_numpy_array(self.current_graph, nodelist=actual_nodelist)
        adjacency_matrix = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        adjacency_matrix[:num_nodes, :num_nodes] = edges
        self.current_graph.graph['adjacency_matrix'] = adjacency_matrix

        node2vec = Node2Vec(self.current_graph, dimensions=self.node_embedding_dim,
                            walk_length=80, num_walks=10, workers=4, quiet=True)
        model = node2vec.fit()
        embeddings = {node: model.wv[str(node)]
                      for node in self.current_graph.nodes()}
        nx.set_node_attributes(
            self.current_graph, embeddings, 'node_embeddings')
        self.separator = set()
        self.action_mask = np.ones((MAX_NODES,), dtype=np.float32)
        self.done = False

        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            raise Exception(
                'Episode is done. Call reset() to start a new episode.')
        self.steps += 1
        reward = self._calc_reward(action)
        obs = self._get_obs()
        return obs, reward, self.done, self.truncated, {}


# %%
if __name__ == "__main__":
    _root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(_root, 'data', '200', '0.1')
    graphs = [os.path.join(data_dir, f) for f in os.listdir(
        data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    env = VertexSeparatorEnv(graphs, node_embedding_dim=4)

    env = make_vec_env(lambda: VertexSeparatorEnv(
        graphs, node_embedding_dim=4), n_envs=1)

    model = A2C("MultiInputPolicy", env, verbose=0)

    model.learn(
        total_timesteps=10000,
        progress_bar=False,
    )

    print('Finished training.')

    model.save("a2c_vertex_separator")

    env.close()
# %%
