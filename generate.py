# %%
import networkx as nx
import itertools
import os
# %%
density = [0.01,0.1,0.5]
nodes = [1000]
num = 100

root_dir = os.path.dirname(os.path.abspath(__file__))

for n, d in itertools.product(nodes, density):
    dir_name = os.path.join(root_dir,f"data/{n}/{d}")
    os.makedirs(os.path.join(root_dir,f"data/{n}/{d}"), exist_ok=True)
    for i in range(num):
        G = nx.gnp_random_graph(n, d, directed=False)
        nx.write_edgelist(G, os.path.join(dir_name,f"{i}.edgelist"), data=False)
# %%
import networkx as nx
import itertools
import os
from networkx.algorithms.approximation import max_clique
import pandas as pd
import matplotlib.pyplot as plt

density = [0.01,0.1,0.5]
nodes = [200]
num = 10

root_dir = os.path.dirname(os.path.abspath(__file__))
for n, d in itertools.product(nodes, density):
    if os.path.exists(os.path.join(root_dir,f"examples/{n}/max_cliques.csv")):
        max_cliques = pd.read_csv(os.path.join(root_dir,f"examples/{n}/max_cliques.csv"), index_col=0)
    else:
        max_cliques = {}
        dir_name = os.path.join(root_dir,f"examples/{n}/{d}")
        os.makedirs(os.path.join(root_dir,f"examples/{n}/{d}"), exist_ok=True)
        for i in range(num):
            if os.path.exists(os.path.join(dir_name,f"{i}.edgelist")):
                G = nx.read_edgelist(os.path.join(dir_name,f"{i}.edgelist"))
            else:
                G = nx.gnp_random_graph(n, d, directed=False)
                nx.write_edgelist(G, os.path.join(dir_name,f"{i}.edgelist"), data=False)
            max_cliques[f"{n}_{d}_{i}"] = len(max_clique(G))

        max_cliques = pd.DataFrame.from_dict(max_cliques, orient='index', columns=['max_clique'])
        max_cliques.to_csv(os.path.join(root_dir,f"examples/{n}/max_cliques.csv"))
