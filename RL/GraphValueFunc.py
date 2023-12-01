import random, torch, numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F


np.random.seed(483)
random.seed(392)
torch.manual_seed(1234567)


class GCN(torch.nn.Module): # Q(s,a) => s graph, a node
    def __init__(self, hidden_channels, num_features,dropout=None):
        super().__init__()
        # Todo: GAT vs GCN
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, nx_graph):
        pyg_graph = from_networkx(nx_graph) # Todo: check if pyg_graph is ok after decomposition and conversion from networkx
        x = self.conv1(pyg_graph.x, pyg_graph.edge_index)
        x = x.relu()
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, pyg_graph.edge_index)
        return x

if __name__ == '__main__':
    dataset = GNNBenchmarkDataset(root='data/MNIST', name='MNIST', transform=OneHotDegree(75))

    print(f'Dataset: {dataset}:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')

    # data = dataset[0]  # Get the first graph object.
    #
    # # Gather some statistics about the graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(data.x)
    # print(data.y)

    # max_nodes = -1
    # for data in dataset:
    #     if data.num_nodes > max_nodes:
    #         max_nodes = data.num_nodes
    # print(max_nodes)

    # min_nodes = float('inf')
    # for data in dataset:
    #     if data.num_nodes < min_nodes:
    #         min_nodes = data.num_nodes
    # print(min_nodes) # 40
    # exit(0)

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    indices = np.arange(0,len(dataset))
    np.random.shuffle(indices)
    train_indices = indices[:int(train_ratio*len(dataset))]
    val_indices = indices[int(train_ratio*len(dataset)):int((train_ratio+val_ratio)*len(dataset))]
    test_indices = indices[int((train_ratio+val_ratio)*len(dataset)):]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    data = dataset[0]  # Get the first graph object.
    nx_graph = data.to_networkx()
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    model = GCN(hidden_channels=16, num_features=dataset.num_features)
    print(model(nx_graph).size())




