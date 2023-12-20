from NNet import NeuralNet
from GraphMDP import GraphMDP
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import numpy as np
import networkx as nx
from MCTS import MCTS
import random
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from copy import deepcopy
from DBK import DBK, get_features
import torch

random.seed(392)
np.random.seed(483)

def maximum_clique_exact_solve_np_hard(G):
    max_clique_number = nx.graph_clique_number(G)
    cliques = nx.find_cliques(G)
    for cl in cliques:
        if len(cl) == max_clique_number:
            return cl


def get_avg_decomposition_cnt_from_val(val_dataset, limit, nnet):
    print("==Validation==")
    decomposition_cnt_lst = []
    for i, data in enumerate(val_dataset):
        nx_graph = to_networkx(data, ['x'], None, ['y'], True)
        _, decomposition_cnt = DBK(nx_graph, limit, maximum_clique_exact_solve_np_hard, nnet)
        decomposition_cnt_lst.append(decomposition_cnt)
    return np.mean(decomposition_cnt_lst)


def prepare_testing_plot_data(test_dataset, limit, nnet):
    test_dataset_graph_size = []
    test_dataset_graph_density = []
    smart_decomposition_cnt_lst = []
    smart_leaf_subgraph_lst = []
    smart_times = []
    # min_huristic_decomposition_cnt_lst = []
    # min_huristic_leaf_subgraph_lst = []
    # min_huristic_times = []
    # max_huristic_decomposition_cnt_lst = []
    # max_huristic_leaf_subgraph_lst = []
    # max_huristic_times = []
    # rand_huristic_decomposition_cnt_lst = []
    # rand_huristic_leaf_subgraph_lst = []
    # rand_huristic_times = []
    for i, data in tqdm(enumerate(test_dataset)):

        nx_graph = to_networkx(data, ['x'], None, ['y'], True)
        test_dataset_graph_size.append(len(nx_graph.nodes))
        test_dataset_graph_density.append(nx.density(nx_graph))

        # start_time = time.time()
        # _, decomposition_cnt,leaf_subgraphs = DBK(nx_graph.copy(), limit, maximum_clique_exact_solve_np_hard,'MIN')
        # end_time = time.time()
        # min_huristic_times.append(end_time - start_time)
        # min_huristic_decomposition_cnt_lst.append(decomposition_cnt)
        # min_huristic_leaf_subgraph_lst.append(leaf_subgraphs)
        # print("min done")
        #
        # start_time = time.time()
        # _, decomposition_cnt, leaf_subgraphs = DBK(nx_graph.copy(), limit, maximum_clique_exact_solve_np_hard, 'MAX')
        # end_time = time.time()
        # max_huristic_times.append(end_time - start_time)
        # max_huristic_decomposition_cnt_lst.append(decomposition_cnt)
        # max_huristic_leaf_subgraph_lst.append(leaf_subgraphs)
        # print("max done")


        start_time = time.time()
        _, decomposition_cnt,leaf_subgraphs = DBK(nx_graph.copy(), limit, maximum_clique_exact_solve_np_hard, nnet)
        end_time = time.time()
        smart_times.append(end_time - start_time)
        smart_decomposition_cnt_lst.append(decomposition_cnt)
        smart_leaf_subgraph_lst.append(leaf_subgraphs)
        # print("smart done")

        # start_time = time.time()
        # _, decomposition_cnt, leaf_subgraphs = DBK(nx_graph.copy(), limit, maximum_clique_exact_solve_np_hard, 'RAND')
        # end_time = time.time()
        # rand_huristic_times.append(end_time - start_time)
        # rand_huristic_decomposition_cnt_lst.append(decomposition_cnt)
        # rand_huristic_leaf_subgraph_lst.append(leaf_subgraphs)
        # print("rand done")

    # save plot data
    np.save('test_plots_data3/test_dataset_graph_size.npy', test_dataset_graph_size)
    np.save('test_plots_data3/test_dataset_graph_density.npy', test_dataset_graph_density)
    np.save('test_plots_data3/smart_decomposition_cnt_lst.npy', smart_decomposition_cnt_lst)
    np.save('test_plots_data3/smart_leaf_subgraph_lst.npy', smart_leaf_subgraph_lst)
    # np.save('test_plots_data2/min_huristic_decomposition_cnt_lst.npy', min_huristic_decomposition_cnt_lst)
    # np.save('test_plots_data2/min_huristic_leaf_subgraph_lst.npy', min_huristic_leaf_subgraph_lst)
    # np.save('test_plots_data2/max_huristic_decomposition_cnt_lst.npy', max_huristic_decomposition_cnt_lst)
    # np.save('test_plots_data2/max_huristic_leaf_subgraph_lst.npy', max_huristic_leaf_subgraph_lst)
    # np.save('test_plots_data2/rand_huristic_decomposition_cnt_lst.npy', rand_huristic_decomposition_cnt_lst)
    # np.save('test_plots_data2/rand_huristic_leaf_subgraph_lst.npy', rand_huristic_leaf_subgraph_lst)
    np.save('test_plots_data3/smart_times.npy', smart_times)
    # np.save('test_plots_data2/min_huristic_times.npy', min_huristic_times)
    # np.save('test_plots_data2/max_huristic_times.npy', max_huristic_times)
    # np.save('test_plots_data2/rand_huristic_times.npy', rand_huristic_times)

def decomposition_cnt_boxplot(headless=True):
    smart_decomposition_cnt_lst = np.load('test_plots_data3/smart_decomposition_cnt_lst.npy')
    # min_huristic_decomposition_cnt_lst = np.load('test_plots_data/min_huristic_decomposition_cnt_lst.npy')
    # max_huristic_decomposition_cnt_lst = np.load('test_plots_data/max_huristic_decomposition_cnt_lst.npy')
    # rand_huristic_decomposition_cnt_lst = np.load('test_plots_data/rand_huristic_decomposition_cnt_lst.npy')

    my_dict = {'GNN+MCTS': smart_decomposition_cnt_lst}#}, 'MIN': min_huristic_decomposition_cnt_lst,
               #'MAX': max_huristic_decomposition_cnt_lst}#, 'RAND': rand_huristic_decomposition_cnt_lst}

    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    ax.set_ylabel('Number of Decompositions')
    ax.set_title('Decomposition Count Boxplot')
    fig.savefig(f'plots/decomposition_cnt_boxplot.png')
    if not headless:
        plt.show()

def time_boxplot(headless=True):
    smart_times = np.load('test_plots_data/smart_times.npy')
    min_huristic_times = np.load('test_plots_data/min_huristic_times.npy')
    max_huristic_times = np.load('test_plots_data/max_huristic_times.npy')
    rand_huristic_times = np.load('test_plots_data/rand_huristic_times.npy')

    my_dict = {'GNN+MCTS': smart_times, 'MIN': min_huristic_times,
               'MAX': max_huristic_times, 'RAND': rand_huristic_times}

    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    ax.set_ylabel('Time (sec)')
    ax.set_title('Speed Boxplot')
    fig.savefig(f'plots/speed_boxplot.png')
    if not headless:
        plt.show()

def subgraph_cnt_boxplot(headless=True):
    smart_subgraph_cnt_lst = np.load('test_plots_data/smart_leaf_subgraph_lst.npy', allow_pickle=True)
    smart_subgraph_cnt_lst = list(map(lambda x: len(x), smart_subgraph_cnt_lst))
    min_huristic_subgraph_cnt_lst = np.load('test_plots_data/min_huristic_leaf_subgraph_lst.npy', allow_pickle=True)
    min_huristic_subgraph_cnt_lst = list(map(lambda x: len(x), min_huristic_subgraph_cnt_lst))
    max_huristic_subgraph_cnt_lst = np.load('test_plots_data/max_huristic_leaf_subgraph_lst.npy', allow_pickle=True)
    max_huristic_subgraph_cnt_lst = list(map(lambda x: len(x), max_huristic_subgraph_cnt_lst))
    rand_huristic_subgraph_cnt_lst = np.load('test_plots_data/rand_huristic_leaf_subgraph_lst.npy', allow_pickle=True)
    rand_huristic_subgraph_cnt_lst = list(map(lambda x: len(x), rand_huristic_subgraph_cnt_lst))

    my_dict = {'GNN+MCTS': smart_subgraph_cnt_lst, 'MIN': min_huristic_subgraph_cnt_lst,
               'MAX': max_huristic_subgraph_cnt_lst, 'RAND': rand_huristic_subgraph_cnt_lst}

    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    ax.set_ylabel('Number of Solved Subgraphs by Solver')
    ax.set_title('Subgraph Count Boxplot')
    fig.savefig(f'plots/subgraph_cnt_boxplot.png')
    if not headless:
        plt.show()

def graph_size_decomposition_cnt_errorbar(headless=True):

    graph_sizes = np.load('test_plots_data/test_dataset_graph_size.npy')
    graph_sizes = list(map(lambda x: int(round(x/10,0)*10), graph_sizes))
    smart_decomposition_cnt_lst = np.load('test_plots_data/smart_decomposition_cnt_lst.npy')
    min_huristic_decomposition_cnt_lst = np.load('test_plots_data/min_huristic_decomposition_cnt_lst.npy')
    max_huristic_decomposition_cnt_lst = np.load('test_plots_data/max_huristic_decomposition_cnt_lst.npy')
    rand_huristic_decomposition_cnt_lst = np.load('test_plots_data/rand_huristic_decomposition_cnt_lst.npy')

    data_by_size = {'MIN':{}, 'MAX':{}, 'RAND':{}, 'GNN+MCTS':{}}
    for method, sizes, decomposition_cnt_lst in zip(['GNN+MCTS', 'MIN', 'MAX', 'RAND'],[graph_sizes,graph_sizes,graph_sizes,graph_sizes],
                                            [smart_decomposition_cnt_lst, min_huristic_decomposition_cnt_lst,
                                                max_huristic_decomposition_cnt_lst, rand_huristic_decomposition_cnt_lst]):
        for size, num_decompositions in zip(sizes, decomposition_cnt_lst):
            if size not in data_by_size[method]:
                data_by_size[method][size] = []

            data_by_size[method][size].append(num_decompositions)

    sorted_size = list(sorted(data_by_size['GNN+MCTS'].keys()))
    l = len(sorted_size)
    means = {'MIN':[0]*l, 'MAX':[0]*l, 'RAND':[0]*l, 'GNN+MCTS':[0]*l}
    stds = {'MIN':[0]*l, 'MAX':[0]*l, 'RAND':[0]*l, 'GNN+MCTS':[0]*l}

    for method in data_by_size:
        for size, decompositions in data_by_size[method].items():
            idx = sorted_size.index(size)
            means[method][idx] = np.mean(decompositions)
            stds[method][idx] = np.std(decompositions)

    for method in ['MIN', 'MAX', 'RAND', 'GNN+MCTS']:
        eb1 = plt.errorbar(sorted_size, means[method], yerr=stds[method], label=method)
        eb1[-1][0].set_linestyle(':')

    plt.xlabel('Graph Size')
    plt.ylabel('Number of Decompositions')
    plt.title('Variation in Number of Decompositions by Graph Size')
    plt.legend()
    plt.savefig(f'plots/graph_size_decomposition_cnt_errorbar.png')
    if not headless:
        plt.show()

def graph_size_subgraph_cnt_errorbar(headless=True):
    # Todo: How to count subgraphs?
    graph_sizes = np.load('test_plots_data/test_dataset_graph_size.npy')
    graph_sizes = list(map(lambda x: int(round(x/10,0)*10), graph_sizes))
    smart_subgraph_cnt_lst = np.load('test_plots_data/smart_leaf_subgraph_lst.npy', allow_pickle=True)
    smart_subgraph_cnt_lst = list(map(lambda x: len(x), smart_subgraph_cnt_lst))
    min_huristic_subgraph_cnt_lst = np.load('test_plots_data/min_huristic_leaf_subgraph_lst.npy', allow_pickle=True)
    min_huristic_subgraph_cnt_lst = list(map(lambda x: len(x), min_huristic_subgraph_cnt_lst))
    max_huristic_subgraph_cnt_lst = np.load('test_plots_data/max_huristic_leaf_subgraph_lst.npy', allow_pickle=True)
    max_huristic_subgraph_cnt_lst = list(map(lambda x: len(x), max_huristic_subgraph_cnt_lst))
    rand_huristic_subgraph_cnt_lst = np.load('test_plots_data/rand_huristic_leaf_subgraph_lst.npy', allow_pickle=True)
    rand_huristic_subgraph_cnt_lst = list(map(lambda x: len(x), rand_huristic_subgraph_cnt_lst))

    data_by_size = {'MIN':{}, 'MAX':{}, 'RAND':{}, 'GNN+MCTS':{}}
    for method, sizes, decomposition_cnt_lst in zip(['GNN+MCTS', 'MIN', 'MAX', 'RAND'],[graph_sizes,graph_sizes,graph_sizes,graph_sizes],
                                            [smart_subgraph_cnt_lst, min_huristic_subgraph_cnt_lst,
                                                max_huristic_subgraph_cnt_lst, rand_huristic_subgraph_cnt_lst]):
        for size, num_decompositions in zip(sizes, decomposition_cnt_lst):
            if size not in data_by_size[method]:
                data_by_size[method][size] = []

            data_by_size[method][size].append(num_decompositions)

    sorted_size = list(sorted(data_by_size['GNN+MCTS'].keys()))
    l = len(sorted_size)
    means = {'MIN':[0]*l, 'MAX':[0]*l, 'RAND':[0]*l, 'GNN+MCTS':[0]*l}
    stds = {'MIN':[0]*l, 'MAX':[0]*l, 'RAND':[0]*l, 'GNN+MCTS':[0]*l}

    for method in data_by_size:
        for size, decompositions in data_by_size[method].items():
            idx = sorted_size.index(size)
            means[method][idx] = np.mean(decompositions)
            stds[method][idx] = np.std(decompositions)

    for method in ['MIN', 'MAX', 'RAND', 'GNN+MCTS']:
        eb1 = plt.errorbar(sorted_size, means[method], yerr=stds[method], label=method)
        eb1[-1][0].set_linestyle(':')

    plt.xlabel('Graph Size')
    plt.ylabel('Subgraph Count')
    plt.title('Variation in Number of Leaf Nodes by Graph Size')
    plt.legend()
    plt.savefig(f'plots/graph_size_subgraph_cnt_errorbar.png')
    if not headless:
        plt.show()

def graph_density_decomposition_cnt_errorbar(headless=True):

    graph_densities = np.load('test_plots_data/test_dataset_graph_density.npy')
    graph_densities = [round(d, 2) for d in graph_densities]
    smart_decomposition_cnt_lst = np.load('test_plots_data/smart_decomposition_cnt_lst.npy')
    min_huristic_decomposition_cnt_lst = np.load('test_plots_data/min_huristic_decomposition_cnt_lst.npy')
    max_huristic_decomposition_cnt_lst = np.load('test_plots_data/max_huristic_decomposition_cnt_lst.npy')
    rand_huristic_decomposition_cnt_lst = np.load('test_plots_data/rand_huristic_decomposition_cnt_lst.npy')

    data_by_density = {'MIN':{}, 'MAX':{}, 'RAND':{}, 'GNN+MCTS':{}}
    for method, densities, decomposition_cnt_lst in zip(['GNN+MCTS', 'MIN', 'MAX', 'RAND'],[graph_densities,graph_densities,graph_densities,graph_densities],
                                            [smart_decomposition_cnt_lst, min_huristic_decomposition_cnt_lst,
                                                max_huristic_decomposition_cnt_lst, rand_huristic_decomposition_cnt_lst]):
        for density, num_decompositions in zip(densities, decomposition_cnt_lst):
            if density not in data_by_density[method]:
                data_by_density[method][density] = []

            data_by_density[method][density].append(num_decompositions)

    sorted_density = list(sorted(data_by_density['GNN+MCTS'].keys()))
    l = len(sorted_density)
    means = {'MIN':[0]*l, 'MAX':[0]*l, 'RAND':[0]*l, 'GNN+MCTS':[0]*l}
    stds = {'MIN':[0]*l, 'MAX':[0]*l, 'RAND':[0]*l, 'GNN+MCTS':[0]*l}

    for method in data_by_density:
        for density, decompositions in data_by_density[method].items():
            idx = sorted_density.index(density)
            means[method][idx] = np.mean(decompositions)
            stds[method][idx] = np.std(decompositions)

    for method in ['MIN', 'MAX', 'RAND', 'GNN+MCTS']:
        eb1 = plt.errorbar(sorted_density, means[method], yerr=stds[method], label=method)
        eb1[-1][0].set_linestyle('dotted')


    plt.xlabel('Graph Density')
    plt.ylabel('Number of Decompositions')
    plt.title('Variation in Number of Decompositions by Graph Density')
    plt.legend()
    plt.savefig(f'plots/graph_density_decomposition_cnt_errorbar.png')
    if not headless:
        plt.show()



if __name__ == '__main__':

    # # time_boxplot(False)
    # decomposition_cnt_boxplot(False)
    # subgraph_cnt_boxplot(False)
    # # graph_size_decomposition_cnt_errorbar(False)
    # # graph_density_decomposition_cnt_errorbar(False)
    # # graph_size_subgraph_cnt_errorbar(False)
    # exit(0)

    # load final dataset
    start = time.time()
    i = 0
    # train_dataset = torch.load('data/train_200nodes_500graphs.pt')
    # train_dataset = train_dataset[:5]
    # for graph in tqdm(train_dataset):
    #     i += 1
    #     graph.x = get_features(to_networkx(graph, ['x'], None, None, True))

    # Todo: I'm skipping val to speed up for now
    # val_dataset = torch.load('data/val_200nodes_200graphs.pt')
    # for graph in tqdm(val_dataset):
    #     i += 1
    #     graph.x = get_features(to_networkx(graph, ['x'], None, None, True))

    test_dataset = torch.load('data/test_500nodes_10graphs.pt')
    # test_dataset = test_dataset[6:]
    for graph in tqdm(test_dataset):
        i += 1
        graph.x = get_features(to_networkx(graph, ['x'], None, None, True))

    end = time.time()
    print(f"Time to get features for each graph: {(end - start)/i}")

    # nnet = NeuralNet(train_dataset[0].x.shape[1])
    # limit = 40  # used in solver

    # timing DBK on 200-node graph with feature calculation
    # print("here!")
    # start = time.time()
    # for pyg_graph in tqdm(test_dataset[2:5]):
    #     nx_graph = to_networkx(pyg_graph, ['x'], None, None, True)
    #     # print(nx_graph._nodes)
    #     DBK(nx_graph.copy(), 190, maximum_clique_exact_solve_np_hard, 'MIN')
    # end = time.time()
    # print(f"Time to run DBK on 200-node graph: {(end - start)}")
    # exit(0)



    # Uncomment to test
    limit = 40
    nnet = NeuralNet(test_dataset[0].x.shape[1])
    nnet.load_checkpoint(filename='model499.pth.tar')
    prepare_testing_plot_data(test_dataset, limit, nnet)
    print("Done")
    exit(0)


    timeout = 120  # seconds # Todo: adaptive timeout based on nodes and limit
    lowest_val_avg_decomposition = float('inf')

    for i, data in tqdm(enumerate(train_dataset)):
        nx_graph = to_networkx(data, ['x'], None, None, True)

        mdp = GraphMDP(nx_graph, LIMIT=limit,
                       solver_function=maximum_clique_exact_solve_np_hard)
        mcts = MCTS(mdp, nnet)
        mcts.run(timeout=timeout, root_node=None)
        # print("Number of updates to the network: ", nnet.updates_no)
        nnet.write_avg_loss(i)

        if (i + 1) % 50 == 0:
            nnet.save_checkpoint(filename=f'model{i}.pth.tar')
            # new_val_avg_decomposition = get_avg_decomposition_cnt_from_val(val_dataset, limit, nnet)
            # if new_val_avg_decomposition < lowest_val_avg_decomposition:
            #     lowest_val_avg_decomposition = new_val_avg_decomposition
            #     nnet.save_checkpoint(filename=f'best_model{i}_{lowest_val_avg_decomposition}.pth.tar')
            # else:
            #     nnet.save_checkpoint(filename=f'rejected_model{i}_{new_val_avg_decomposition}.pth.tar')

# Todo: debugging + val_test protocol + plotting + let it run and evaluate policy
# validation criterion: how better wrt to the min vertex selection policy
