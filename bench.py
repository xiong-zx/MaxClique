# =============================================================================
# Benchmark against the average node reduction by DBK
# =============================================================================
# %%
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import itertools
import pandas as pd

from utils import *
# %%
def decompose_one_vertex(graph: nx.Graph, select_vertex) -> int:
    """
    Decompose the graph and return the average node reduction
    INPUT:
        graph: nx.Graph
        LIMIT: int
    OUTPUT:
        int
    """
    graph = remove_zero_degree_nodes(graph.copy())
    lb,ub = mc_lower_bound(graph), mc_upper_bound(graph)
    v = select_vertex(graph)
    G1, G2 = ch_partitioning(vertex=v,G=graph)
    G1, G2 = remove_zero_degree_nodes(G1), remove_zero_degree_nodes(G2)
    G1, G2 = k_core_reduction(G1, len(lb) - 1), k_core_reduction(G2, len(lb))
    node_reduced = len(G) - len(G1) - len(G2)
    return node_reduced

def get_vertex_select_func(method):
    if method == 'random':
        return random_vertex
    elif method == 'highest':
        return highest_degree_vertex
    elif method == 'lowest':
        return lowest_degree_vertex
    else:
        raise ValueError(f'Unknown method: {method}')
# %%
if __name__ == "__main__":
    _root = os.path.dirname(os.path.abspath(__file__))

    size_list = [50,100,150,200,250,300,400,500,600,700,800,900,1000,1500,2000] 
    density_list = [0.1,0.5]
    vertex_select_strategy = ['random','highest','lowest']
    N = 100
    if os.path.exists(os.path.join(_root,'bench_stats.csv')):
        stats = pd.read_csv(os.path.join(_root,'bench_stats.csv'))
        stats = stats.values.tolist()
        existing = [tuple(row[:3]) for row in stats]
    else:
        stats = []
        existing = []
    random.seed(42)
    np.random.seed(42)
    
    for size, density, vertex_select_strategy in itertools.product(size_list, density_list, vertex_select_strategy):
        if (size, density, vertex_select_strategy) in existing:
            print(f'\nSkipping:\nSize: {size}\nDensity: {density}\nVertex select strategy: {vertex_select_strategy}')
            continue
        data_dir = os.path.join(_root, 'data',str(size),str(density))
        if not os.path.exists(data_dir):
            continue
        data_list = [os.path.join(data_dir, f) for f in os.listdir(
            data_dir) if f.endswith('.edgelist')]
        
        print(f'\nSize: {size}\nDensity: {density}\nVertex select strategy: {vertex_select_strategy}')
        
        node_reduced_list = []
        for graph in tqdm.tqdm(data_list[:N]):
            G = nx.read_edgelist(graph, nodetype=int)
            G = remove_zero_degree_nodes(G)
            node_reduced = decompose_one_vertex(G, get_vertex_select_func(vertex_select_strategy))
            node_reduced_list.append(node_reduced)
            
        stats.append((size, density, vertex_select_strategy, np.mean(node_reduced_list)))
        print(f'Average node reduced: {np.mean(node_reduced_list)}')
        
    stats = pd.DataFrame(stats, columns=['size','density','vertex_select_strategy','node_reduced'])
    stats.to_csv(os.path.join(_root,'bench_stats.csv'),index=False)
# %%
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import itertools
    
    _root = os.path.dirname(os.path.abspath(__file__))
    stats = pd.read_csv(os.path.join(_root,'bench_stats.csv'),index_col=0)
    
    for density in [0.1,0.5]:
        plt.figure(figsize=(8,6))
        for strategy in ['random','highest','lowest']:
            df = stats[(stats['density']==density) & (stats['vertex_select_strategy']==strategy)]
            plt.plot(df['size'],df['node_reduced'],label=strategy)
        plt.legend()
        plt.xlabel('Number of nodes',fontsize=14)
        plt.ylabel('Average node reduced',fontsize=14)
        plt.title(f'Average Number of Node Reduced (density: {density})',fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(_root,'figures',f'bench_{density}_{strategy}.png'))
        plt.close()
    
# %%
