# %%
import networkx as nx
import torch_geometric
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F

import gymnasium as gym

# %%

# step 1: load graphs

_root = os.path.dirname(os.path.abspath(__file__))
_data_dir = os.path.join(_root, "data", "200", "0.01")
_sample_files = [os.path.join(_data_dir, f) for f in os.listdir(_data_dir) if os.path.isfile(os.path.join(_data_dir, f))]

class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, data_list, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        G = nx.read_edgelist(self.data_list[idx])

        # create node fatures
        node_features = {node: [1.0] for node in G.nodes()}
        nx.set_node_attributes(G, node_features, "x")

        return torch_geometric.utils.from_networkx(G)
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, normalize=True):
        super(GraphSAGE, self).__init__()
        Conv = torch_geometric.nn.SAGEConv
        self.convs = torch.nn.ModuleList(
            [Conv(in_channels, hidden_channels, normalize=normalize)] +
            [Conv(hidden_channels, hidden_channels, normalize=normalize) for i in range(num_layers-2)] +
            [Conv(hidden_channels, out_channels, normalize=normalize)]
        )
        
    def forward(self, data: torch_geometric.data.Data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(in_channels=1,
                  hidden_channels=16,
                  num_layers=3,
                  out_channels=8,
                  normalize=True)
model.train()
model = model.to(device)
dataset = GraphDataset(data_list=_sample_files, root=None)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

count = 0
for data in data_loader:
    data = data.to(device)
    out = model(data)
    print(out)
    count += 1
    if count > 2:
        break
# %%
# step 2: let the agent choose a vertex
