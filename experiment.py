# experiment with the DBK algorithm
# %%
import os
import networkx as nx
from utils import DBK

# %%
if __name__ == '__main__':
    _root = os.path.abspath(os.path.dirname(__file__))
    _data_dir = os.path.join(_root, 'data')

    for d1 in os.listdir(_data_dir):
        scale = d1.split('/')[-1]
        print(scale)
        for d2 in os.listdir(os.path.join(_data_dir, d1)):
            d2 = os.path.join(_data_dir, d1, d2)
            if os.path.isdir(d2):
                if float(d2.split('/')[-1]) < 0.1:
                    continue
                for g in os.listdir(d2):
                    g = os.path.join(d2, g)
                    if os.path.isfile(g) and g.endswith('.edgelist'):
                        G = nx.read_edgelist(g)
                        print(G)
                        break
                
        
# %%
