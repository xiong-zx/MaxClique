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

from utils import *

# %%
class SeparatorTestEnv(gym.Env):
    def __init__(self, n=10, m=10, seed=0):
        super().__init__()
        self.n = n
        self.m = m
        self.seed(seed)
        self.reset()
        
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    _root = os.path.dirname(os.path.dirname(__file__))
    
    data_dir = os.path.join(_root, 'data','200','0.1')
    data_list = [os.path.join(data_dir, f) for f in os.listdir(
        data_dir) if f.endswith('.edgelist')]