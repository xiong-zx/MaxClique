# %%
import networkx as nx
import gymnasium as gym
from gymnasium.spaces import Box, MultiBinary, Dict
import numpy as np
import pandas as pd
import os
import random
import torch_geometric
from torch_geometric.datasets import GNNBenchmarkDataset
import matplotlib.pyplot as plt

import stable_baselines3
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env, VecEnv


from utils import mc_lower_bound, decompose_graph, k_core_reduction

MAX_NODES = 300
STEP_PENALTY = -0.1
SEPARATOR_PENALTY = -1
MASK_PENALTY = -1

# %%


class VertexSeparatorEnv(gym.Env):
    """
    Description:
    A graph is picked randomly from a collection of graphs at each episode. The agent is asked to pick a vertex at each step. The episode ends when the picked vertices form a vertex separator of the graph. The reward is the number of nodes in the original graph minus that in the pruned subgraphs.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            dataset: torch_geometric.data.dataset.Dataset,
            node_embedding: bool = False,
            **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset.shuffle()
        self.node_embedding = node_embedding
        self.action_space = MultiBinary(MAX_NODES)
        if self.node_embedding:
            if isinstance(self.dataset[0], torch_geometric.data.Data):
                node_embedding_dim = self.dataset[0].num_node_features
            elif isinstance(self.dataset[0], nx.Graph):
                node_embedding_dim = 1
            self.observation_space = Dict({
                'node_embeddings': Box(low=-np.inf, high=np.inf, shape=(MAX_NODES, node_embedding_dim), dtype=np.float32),
                'adjacency_matrix': Box(low=0, high=1, shape=(MAX_NODES, MAX_NODES), dtype=np.int8),
                'node_labels': MultiBinary(MAX_NODES)
            })
        else:
            self.observation_space = Dict({
                'adjacency_matrix': Box(low=0, high=1, shape=(MAX_NODES, MAX_NODES), dtype=np.int8),
                'node_labels': MultiBinary(MAX_NODES)
            })

        self.done = False
        self.truncated = False
        self.next_graph = True
        self.separator = set()
        self.steps = 0
        self.graph_index = 0
        self.epoch = 0
        self.info = {}

    def _is_separator(self):
        separator_list = list(self.separator)
        subgraph = self.current_graph.copy()
        subgraph.remove_nodes_from(separator_list)
        if len(subgraph) == 0:
            return True
        else:
            return not nx.is_connected(subgraph)

    def _get_obs(self):
        num_nodes = len(self.current_graph)
        adjacency_matrix = self.current_graph.graph['adjacency_matrix']

        node_labels = np.zeros(MAX_NODES, dtype=np.int8)
        node_labels[:num_nodes] = [
            1 if node in self.separator else 0 for node in self.current_graph.nodes()]

        if self.node_embedding:
            return {
                'node_embeddings': self.current_graph.graph['node_embeddings'],
                'adjacency_matrix': adjacency_matrix,
                'node_labels': node_labels
            }
        else:
            return {
                'adjacency_matrix': adjacency_matrix,
                'node_labels': node_labels
            }

    def _calc_reward(self, action):
        reward = 0
        invalid_action_count = np.sum((action == 1) & (self.action_mask == 0))
        valid_actions = np.where((action == 1) & (self.action_mask == 1))[0]
        reward += invalid_action_count * MASK_PENALTY

        valid_nodes = np.array(list(self.current_graph.nodes))[valid_actions]
        self.separator.update(valid_nodes)
        self.action_mask[valid_actions] = 0
        # check if the separator is valid
        if not self._is_separator():
            reward += STEP_PENALTY
            return reward, {
                'nodes_picked': np.sum((action == 1)),
                'invalid_nodes_picked': invalid_action_count
            }
        else:
            self.done = True
            k = mc_lower_bound(self.current_graph)
            decomposed_graphs = decompose_graph(
                self.current_graph, self.separator)
            decomposed_graphs = [k_core_reduction(
                g, len(k)) for g in decomposed_graphs]
            node_reduced = len(self.current_graph) - \
                sum([len(g) for g in decomposed_graphs])
            # if len(decomposed_graphs) != 2:
            #     print(f"The number of subgraphs is {len(decomposed_graphs)}.")
            #     print(f"The separator size is {len(self.separator)}.")
            reward += node_reduced
            reward += SEPARATOR_PENALTY * len(self.separator)
            return reward, {
                'epoch': self.epoch,
                'nodes_picked': np.sum((action == 1)),
                'invalid_nodes_picked': invalid_action_count,
                'separator_size': len(self.separator),
                'nodes': len(self.current_graph.nodes),
                'node_reduced': node_reduced,
                'steps': self.steps
            }

    def reset(self, seed=None):
        # set seed
        if seed is not None:
            np.random.seed(seed)
        if self.next_graph:
            self.graph_index += 1
            if self.graph_index >= len(self.dataset):
                print(f"Epoch {self.epoch} finished.")
                self.epoch += 1
                self.graph_index = 0
                self.dataset.shuffle()

            data = self.dataset[self.graph_index]
            if isinstance(data, torch_geometric.data.Data):
                edge_index = data.edge_index.to('cpu').numpy()
                self.current_graph = nx.from_edgelist(
                    edge_index.transpose().tolist())
                num_nodes = len(self.current_graph)
                self.current_graph.graph['name'] = data.name  # todo
                if self.node_embedding and data.x is not None:
                    node_embeddings = np.zeros(
                        (MAX_NODES, data.num_node_features), dtype=np.float32)
                    node_embeddings[:num_nodes, :] = data.x.to('cpu').numpy()
                    self.current_graph.graph['node_embeddings'] = node_embeddings
            elif isinstance(data, nx.Graph):
                self.current_graph = self.dataset[self.graph_index]
                num_nodes = len(self.current_graph)

            edges = nx.to_numpy_array(self.current_graph)
            adjacency_matrix = np.zeros(
                (MAX_NODES, MAX_NODES), dtype=np.float32)
            adjacency_matrix[:num_nodes, :num_nodes] = edges
            self.current_graph.graph['adjacency_matrix'] = adjacency_matrix

        self.separator = set()
        self.action_mask = np.zeros((MAX_NODES,), dtype=np.int8)
        self.action_mask[:len(self.current_graph)] = 1
        self.done = False

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward, info = self._calc_reward(action)
        obs = self._get_obs()
        if self.done:
            self.steps = 0
            self.next_graph = True
            key = (self.current_graph.graph['name'], self.epoch)
            self.info[key] = info
        return obs, reward, self.done, self.truncated, info


class NamedGNNBenchmarkDataset(torch_geometric.data.Dataset):
    """Wrapper for GNNBenchmarkDataset that 
    adds a name attribute to each graph.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__(root=None, transform=None, pre_transform=None)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        data.name = f"graph_{idx}"
        return data


class Dataset(torch_geometric.data.Dataset):
    def __init__(self, data_list, root, transform=None, pre_transform=None):
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        G = nx.read_edgelist(self.data_list[idx])
        G.graph['name'] = self.data_list[idx].split('/')[-1].rsplit('.')[0]
        node_features = {node: [1.0] for node in G.nodes()}
        nx.set_node_attributes(G, node_features, "x")
        return G

# def plot_test(info: dict, figure_dir: str):
#     """
#     Visualize the testing process.
#     Generate the following figures:
#     1. Average Node Reduced by Epoch: the average number of nodes reduced in each epoch.
#     2. Average Separator Size by Epoch: the average size of the separator in each epoch.
#     3. Histogram of Node Reduced: the distribution of the number of nodes reduced in the best epoch.
#     4. Histogram of Separator Size: the distribution of the size of the separator in the best epoch (most nodes are reduced).
#     """
#     graph_name_to_index = {}
#     index_counter = 0
#     total_epochs = 0
#     for (graph_name, e), _ in info.items():
#         if e >= total_epochs:
#             total_epochs = e+1
#         if graph_name not in graph_name_to_index:
#             graph_name_to_index[graph_name] = index_counter
#             index_counter += 1

#     total_graphs = len(graph_name_to_index)
#     node_reduced_matrix = np.zeros((total_graphs, total_epochs))
#     steps_matrix = np.zeros((total_graphs, total_epochs))
#     separator_size = np.zeros((total_graphs, total_epochs))

#     for (graph_name, epoch), v in info.items():
#         graph_index = graph_name_to_index[graph_name]
#         node_reduced_matrix[graph_index, epoch] = v['node_reduced']
#         steps_matrix[graph_index, epoch] = v['steps']
#         separator_size[graph_index, epoch] = v['separator_size']

#     avg_node_reduced = np.mean(node_reduced_matrix[:, :total_epochs-1], axis=0)
#     avg_separator_size = np.mean(separator_size[:, :total_epochs-1], axis=0)

#     best_epoch = np.argmax(avg_node_reduced)
#     best_node_reduced = node_reduced_matrix[:, best_epoch]
#     mean_best_node_reduced = np.mean(best_node_reduced)

#     best_separator_size = separator_size[:, best_epoch]
#     mean_best_separator_size = np.mean(best_separator_size)

#     plt.figure(figsize=(10, 6))
#     plt.plot(avg_node_reduced, label='Average Node Reduced')
#     plt.xlabel('Epoch')
#     plt.ylabel('Number of Node Reduced')
#     plt.title('Average Node Reduced by Epoch')
#     plt.legend()
#     plt.savefig(os.path.join(figure_dir, 'test_line_node_reduced.png'))
#     plt.close()



def plot_train(info: dict, figure_dir: str):
    """
    Visualize the training process.
    Generate the following figures:
    1. Average Node Reduced by Epoch: the average number of nodes reduced in each epoch.
    2. Average Separator Size by Epoch: the average size of the separator in each epoch.
    3. Histogram of Node Reduced: the distribution of the number of nodes reduced in the best epoch.
    4. Histogram of Separator Size: the distribution of the size of the separator in the best epoch (most nodes are reduced).
    """
    graph_name_to_index = {}
    index_counter = 0
    total_epochs = 0
    for (graph_name, e), _ in info.items():
        if e >= total_epochs:
            total_epochs = e+1
        if graph_name not in graph_name_to_index:
            graph_name_to_index[graph_name] = index_counter
            index_counter += 1

    total_graphs = len(graph_name_to_index)
    node_reduced_matrix = np.zeros((total_graphs, total_epochs))
    steps_matrix = np.zeros((total_graphs, total_epochs))
    separator_size = np.zeros((total_graphs, total_epochs))

    for (graph_name, epoch), v in info.items():
        graph_index = graph_name_to_index[graph_name]
        node_reduced_matrix[graph_index, epoch] = v['node_reduced']
        steps_matrix[graph_index, epoch] = v['steps']
        separator_size[graph_index, epoch] = v['separator_size']

    avg_node_reduced = np.mean(node_reduced_matrix[:, :total_epochs-1], axis=0)
    avg_separator_size = np.mean(separator_size[:, :total_epochs-1], axis=0)

    best_epoch = np.argmax(avg_node_reduced)
    best_node_reduced = node_reduced_matrix[:, best_epoch]
    mean_best_node_reduced = np.mean(best_node_reduced)

    best_separator_size = separator_size[:, best_epoch]
    mean_best_separator_size = np.mean(best_separator_size)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_node_reduced, label='Average Node Reduced')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Node Reduced')
    plt.title('Average Node Reduced by Epoch')
    plt.legend()
    plt.savefig(os.path.join(figure_dir, 'train_line_node_reduced.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_separator_size, label='Average Separator Size')
    plt.xlabel('Epoch')
    plt.ylabel('Separator Size')
    plt.title('Average Separator Size by Epoch')
    plt.legend()
    plt.savefig(os.path.join(figure_dir, 'train_line_separator_size.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(best_node_reduced, bins=30, align='mid', rwidth=0.8)
    plt.xlabel('Number of Node Reduced')
    plt.ylabel('Number of Graphs')
    plt.title('Histogram of Node Reduced')
    plt.savefig(os.path.join(figure_dir, 'train_hist_node_reduced.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(best_separator_size, bins=30, align='mid', rwidth=0.8)
    plt.xlabel('Separator Size')
    plt.ylabel('Number of Graphs')
    plt.title('Histogram of Separator Size')
    plt.savefig(os.path.join(figure_dir, 'train_hist_separator_size.png'))
    plt.close()

    # the average number of nodes reduced in the BEST epoch
    print(
        f"Training Best Avg Node Reduced (epoch {best_epoch}): {mean_best_node_reduced}")
    print(
        f"Training Best Avg Separator Size (epoch {best_epoch}): {mean_best_separator_size}")


def evaluate_test(info: dict):
    """Evaluate the testing results.
    """
    best_results = {}

    # collect the best results for each graph
    for (graph_name, _), v in info.items():
        if graph_name not in best_results or v['node_reduced'] > best_results[graph_name]['node_reduced']:
            best_results[graph_name] = v

    total_graphs = len(best_results)
    node_reduced_matrix = np.zeros((total_graphs, ))
    steps_matrix = np.zeros((total_graphs, ))
    separator_size = np.zeros((total_graphs, ))

    for i, result in enumerate(best_results.values()):
        node_reduced_matrix[i] = result['node_reduced']
        steps_matrix[i] = result['steps']
        separator_size[i] = result['separator_size']

    mean_node_reduced = np.mean(node_reduced_matrix)
    mean_separator_size = np.mean(separator_size)
    mean_steps = np.mean(steps_matrix)

    return mean_node_reduced, mean_separator_size, mean_steps


def train_model(
    model: stable_baselines3.A2C,
    env: VecEnv,
    total_timesteps: int
) -> dict:

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model.get_env().get_attr('info')[0]


def test_model(
    model: stable_baselines3.A2C,
    env: VecEnv,
    val_steps: int,
    epoch: int = 1
) -> dict:
    obs = env.reset()
    done = False
    if val_steps is None:
        while env.get_attr('epoch')[0] < epoch:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, done, info = env.step(action)
            if done:
                obs = env.reset()
    else:
        for _ in range(val_steps):
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, done, info = env.step(action)
            if done:
                obs = env.reset()

    return env.get_attr('info')[0]


# %%
if __name__ == "__main__":

    # hyperparameters
    total_steps = int(5e5)  # total number of training steps
    interval = int(5e2)  # number of steps between each test
    val_steps = None  # number of steps for each test; None for validating for 1 epoch
    test_steps = None # number of steps for each test; None for testing for 1 epoch
    random.seed(42)
    np.random.seed(42)
    
    _root = os.path.abspath(os.path.dirname(__file__))

    # data_dir = os.path.join(_root, 'data', 'CLUSTER')
    # _train_dataset = GNNBenchmarkDataset(root=os.path.join(
    #     _root, 'data'), name='CLUSTER', split='train')
    # _test_dataset = GNNBenchmarkDataset(root=os.path.join(
    #     _root, 'data'), name='CLUSTER', split='test')
    # train_dataset = NamedGNNBenchmarkDataset(_train_dataset)
    # val_dataset = NamedGNNBenchmarkDataset(_test_dataset)

    data_dir = os.path.join(_root, 'data', '300', '0.1')
    data_list = [os.path.join(data_dir, f) for f in os.listdir(
        data_dir) if f.endswith('.edgelist')][:100]
    dataset = Dataset(data_list=data_list, root=None)
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):]
    test_dataset = val_dataset
    
    # By now, the train_dataset and val_dataset should be ready.

    train_env = make_vec_env(lambda: VertexSeparatorEnv(
        dataset=train_dataset, node_embedding=False), n_envs=1)
    test_env = make_vec_env(lambda: VertexSeparatorEnv(
        dataset=test_dataset, node_embedding=False), n_envs=1)
    if os.path.exists(os.path.join(data_dir, "a2c_vertex_separator.zip")):
        msg = input("Model exists. Press Y/y to load the model.")
        if msg.lower() == 'y':
            print("Loading model.")
            model = A2C.load(path=os.path.join(
                data_dir, "a2c_vertex_separator"), verbose=0)
            model.env = train_env
            print("Model loaded.")
        else:
            model = A2C("MultiInputPolicy", train_env, verbose=0)
            print("Created a new model.")
    else:
        model = A2C("MultiInputPolicy", train_env, verbose=0)
    best_node_reduced = -np.inf
    val_node_reduced_list = []
    val_separator_size_list = []
    for i in range(0, total_steps, interval):
        print(f"\n{i}")
        print(f"Training for {interval} steps.")
        _ = train_model(model, train_env, interval)

        print(f"Testing.")
        val_env = make_vec_env(lambda: VertexSeparatorEnv(
            dataset=val_dataset.shuffle(), node_embedding=False), n_envs=1)
        val_info = test_model(model, val_env, val_steps)
        val_env.close()
        node_reduced, separator_size, _ = evaluate_test(val_info)
        # tracks the number of nodes reduced during the training process on validation dataset
        val_node_reduced_list.append(node_reduced)
        # tracks the size of the separator during the training process on validation dataset
        val_separator_size_list.append(separator_size)
        print(f"Validating Node Reduced: {node_reduced}")
        print(f"Validating Separator Size: {separator_size}")
        if node_reduced >= best_node_reduced:
            best_node_reduced = node_reduced
            best_separator_size = separator_size
            model.save(path=os.path.join(data_dir, "a2c_vertex_separator"))
            print("Model saved.")

    print('\nFinished training.')

    # plot the training data
    train_info = train_env.get_attr('info')[0]
    plot_train(info=train_info, figure_dir=data_dir)

    # plot the validating data
    plt.figure(figsize=(10, 6))
    plt.plot(val_node_reduced_list)
    plt.xlabel('Round')
    plt.ylabel('Node Reduced')
    plt.title('Node Reduced on Test Set')
    plt.savefig(os.path.join(data_dir, 'val_line_node_reduced.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(val_separator_size_list)
    plt.xlabel('Round')
    plt.ylabel('Separator Size')
    plt.title('Separator Size on Test Set')
    plt.savefig(os.path.join(data_dir, 'val_line_separator_size.png'))
    plt.close()
    # test the model on the test dataset
    print(f"Testing on test dataset.")
    test_info = test_model(model, test_env, val_steps)
    test_node_reduced, test_separator_size, _ = evaluate_test(test_info)
    print(f"Testing Node Reduced: {test_node_reduced}")
    print(f"Testing Separator Size: {test_separator_size}")
    
    train_env.close()
    val_env.close()
    test_env.close()
    
    val_node_reduced_list =  pd.DataFrame(val_node_reduced_list)
    val_separator_size_list = pd.DataFrame(val_separator_size_list)
    test_node_reduced = pd.DataFrame([test_node_reduced])
    test_separator_size = pd.DataFrame([test_separator_size])
    val_node_reduced_list.to_csv(os.path.join(data_dir, 'val_node_reduced.csv'))
    val_separator_size_list.to_csv(os.path.join(data_dir, 'val_separator_size.csv'))
    test_node_reduced.to_csv(os.path.join(data_dir, 'test_node_reduced.csv'))
    test_separator_size.to_csv(os.path.join(data_dir, 'test_separator_size.csv'))
    
# %%
