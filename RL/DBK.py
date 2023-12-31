import networkx as nx
import random
import torch
import numpy as np
import pickle
import uuid
from torch_geometric.utils.convert import from_networkx, to_networkx

# random.seed(11)
# np.random.seed(11)

decomposition_cnt = 0
leaf_subgraphs = []


def mc_upper_bound(G):
    """
	INPUT:
	 - "G" Networkx Undirected Graph
	OUTPUT:
	 - "chromatic_number" integer upper bound on the maximum clique number
	"""
    answ = nx.algorithms.coloring.greedy_color(G)
    chromatic_number = list(set(list(answ.values())))
    return len(chromatic_number)


def mc_lower_bound(G):
    """
	INPUT:
	 - "G" Networkx Undirected Graph
	OUTPUT:
	 - "lower bound" list of variables which form a clique in G
	"""
    return nx.maximal_independent_set(nx.complement(G))


def edge_k_core(G, k):
    """
	INPUT:
	 - "G" Networkx Undirected Graph
	 - "k" Integer that is at least one less than the global maximum clique number	
	OUTPUT:
	 - "G" Networkx Undirected Graph where edge k-core reduction has been applied
	"""
    for a in list(G.edges()):
        x = list(G.neighbors(a[0]))
        y = list(G.neighbors(a[1]))
        if len(list(set(x) & set(y))) <= (k - 2):
            G.remove_edge(a[0], a[1])
    return G


def k_core_reduction(graph, k):
    """
	INPUT:
	 - "graph" Networkx Undirected Graph
	 - "k" Integer that is at least one less than the global maximum clique number	
	OUTPUT:
	 - "graph" Networkx Undirected Graph where k-core reduction has been applied
	"""
    graph = nx.k_core(graph, k)
    ref1 = len(list(graph.edges()))
    graph = edge_k_core(graph, k)
    ref2 = len(list(graph.edges()))
    while ref1 != ref2:
        if len(graph) == 0:
            return graph
        graph = nx.k_core(graph, k)
        ref1 = len(list(graph.edges()))
        graph = edge_k_core(graph, k)
        ref2 = len(list(graph.edges()))
    return graph


def is_clique(G):
    """
	INPUT:
	 - "G" Networkx Undirected Graph
	OUTPUT:
	 - "True" if G is a clique, and "False" if G is not a clique
	"""
    n = len(list(G.nodes()))
    m = len(list(G.edges()))
    if int(m) == int((n * (n - 1)) / float(2)):
        return True
    else:
        return False


def ch_partitioning(vertex, G, vertex_selection_func=None):
    global decomposition_cnt
    """
	INPUT:
	 - "vertex" splitting vertex
	 - "G" Networkx Undirected Graph
	OUTPUT:
	 - "SSG" Left subgraph after partitioning
	 - "SG" Right subgraph after partitioning
	"""
    decomposition_cnt += 1
    n = list(G.neighbors(vertex))
    # original from DBK paper
    # Gp = []
    # for iter in list(G.edges()):
    #     if iter[0] in n:
    #         if iter[1] in n:
    #             Gp.append(iter)
    # G.remove_node(vertex)
    # return nx.Graph(Gp), G

    # This is meant to fix lack of node features in Gp => not computing node features again which is wrong
    # Gp = nx.Graph()
    # for iter in list(G.edges()):
    #     if iter[0] in n:
    #         if iter[1] in n:
    #             Gp.add_node(iter[0], x=G.nodes[iter[0]]['x'])
    #             Gp.add_node(iter[1], x=G.nodes[iter[1]]['x'])
    #             Gp.add_edge(iter[0], iter[1])
    # G.remove_node(vertex)

    # Computes node features again
    Gp = nx.Graph()
    for iter in list(G.edges()):
        if iter[0] in n:
            if iter[1] in n:
                Gp.add_node(iter[0], x=G.nodes[iter[0]]['x'])
                Gp.add_node(iter[1], x=G.nodes[iter[1]]['x'])
                Gp.add_edge(iter[0], iter[1])
    G.remove_node(vertex)

    # pyg_G = from_networkx(G)
    # pyg_Gp = from_networkx(Gp)
    # pyg_G.x = get_features(G)
    # pyg_Gp.x = get_features(Gp)
    # new_G = to_networkx(pyg_G, ['x'], None, None, True)
    # new_Gp = to_networkx(pyg_Gp, ['x'], None, None, True)

    if vertex_selection_func not in ['RAND','MIN','MAX']: # features are not recomputed for RAND, MIN, MAX
        G_features = get_features(G)
        for node in G.nodes():
            node_idx = list(G.nodes()).index(node)
            G.nodes[node]['x'] = G_features[node_idx]
        Gp_features = get_features(Gp)
        for node in Gp.nodes():
            node_idx = list(Gp.nodes()).index(node)
            Gp.nodes[node]['x'] = Gp_features[node_idx]


    return Gp, G


def lowest_degree_vertex(graph):
    """
	INPUT:
	 - "graph" Networkx Undirected Graph
	OUTPUT:
	 - "i" node that has the lowest degree in the graph
	"""
    degrees = [graph.degree(a) for a in list(graph.nodes())]
    minimum = min(degrees)
    for i in list(graph.nodes()):
        if graph.degree(i) == minimum:
            return i


def highest_degree_vertex(graph):
    """
	INPUT:
	 - "graph" Networkx Undirected Graph
	OUTPUT:
	 - "i" node that has the maximum degree in the graph
	"""
    degrees = [graph.degree(a) for a in list(graph.nodes())]
    maximum = max(degrees)
    for i in list(graph.nodes()):
        if graph.degree(i) == maximum:
            return i


def random_vertex_selection(graph):
    """
    INPUT:
     - "graph" Networkx Undirected Graph
    OUTPUT:
     - "i" node that is randomly selected from the graph
    """
    return random.choice(list(graph.nodes()))


def smart_vertex_selection(graph, nnet):
    action_idx = nnet.select_action(graph)
    return list(graph.nodes)[action_idx]


def remove_zero_degree_nodes(graph):
    """
	INPUT:
	 - "graph" Networkx Undirected Graph
	OUTPUT:
	 - "graph" Networkx Undirected Graph with no zero degree nodes
	"""
    nodes = list(graph.nodes())
    for n in nodes:
        if graph.degree(n) == 0:
            graph.remove_node(n)
    return graph


def get_features(nxgraph):
    # get features
    num_nodes = nxgraph.number_of_nodes()
    num_edges = nxgraph.number_of_edges()
    node_degrees = dict(nxgraph.degree())

    uid = uuid.uuid4()

    try:
        degree_centrality = nx.degree_centrality(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_degree_centrality_{uid}.pickle", "wb"))
        degree_centrality = [-1] * num_nodes

    try:
        betweenness_centrality = nx.betweenness_centrality(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_betweenness_centrality_{uid}.pickle", "wb"))
        betweenness_centrality = [-1] * num_nodes

    try:
        closeness_centrality = nx.closeness_centrality(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_closeness_centrality_{uid}.pickle", "wb"))
        closeness_centrality = [-1] * num_nodes

    try:
        eigenvector_centrality = nx.eigenvector_centrality(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_eigenvector_centrality_{uid}.pickle", "wb"))
        eigenvector_centrality = [-1] * num_nodes

    try:
        pagerank_centrality = nx.pagerank(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_pagerank_centrality_{uid}.pickle", "wb"))
        pagerank_centrality = [-1] * num_nodes

    try:
        harmonic_centrality = nx.harmonic_centrality(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_harmonic_centrality_{uid}.pickle", "wb"))
        harmonic_centrality = [-1] * num_nodes

    try:
        load_centrality = nx.load_centrality(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_load_centrality_{uid}.pickle", "wb"))
        load_centrality = [-1] * num_nodes

    try:
        clustering_coefficient = nx.clustering(nxgraph)
    except:
        pickle.dump(nxgraph, open(f"errors/error_clustering_coefficient_{uid}.pickle", "wb"))
        clustering_coefficient = [-1] * num_nodes
    # make it into an array
    features_array = np.array([
        [num_nodes] * num_nodes,
        [num_edges] * num_nodes,
        list(node_degrees.values()),
        list(degree_centrality.values()) if not isinstance(degree_centrality, list) else degree_centrality,
        list(betweenness_centrality.values()) if not isinstance(betweenness_centrality, list) else betweenness_centrality,
        list(closeness_centrality.values()) if not isinstance(closeness_centrality, list) else closeness_centrality,
        list(eigenvector_centrality.values()) if not isinstance(eigenvector_centrality, list) else eigenvector_centrality,
        list(pagerank_centrality.values()) if not isinstance(pagerank_centrality, list) else pagerank_centrality,
        list(harmonic_centrality.values()) if not isinstance(harmonic_centrality, list) else harmonic_centrality,
        list(load_centrality.values()) if not isinstance(load_centrality, list) else load_centrality,
        list(clustering_coefficient.values()) if not isinstance(clustering_coefficient, list) else clustering_coefficient])
    features_array = features_array.T  # feature for each node is a row
    return torch.tensor(features_array)


def DBK(graph, LIMIT, solver_function, vertex_selection_func=None):
    global decomposition_cnt, leaf_subgraphs
    decomposition_cnt = 0
    leaf_subgraphs = []
    """
	INPUT:
	 - "graph" must be a Networkx Undirected Graph
	 - "LIMIT" is an integer describing the largest size of graph which solver_func can solve; all subgraph sizes solved will be less than or equal to LIMIT
	 - "solver_function" takes a Networkx Graph, and outputs a list of nodes which are hopefully the Maximum Clique elements; it can be an approximate or exact solver function
	OUTPUT:
	 - "k" is a list of graph nodes which form a clique in the input graph. If the solver is exact, then k is the Maximum Clique
	NOTES:
	 - The central idea of using bounds is that we maintain a global lower bound on the Maximum Clique. Then, for each sub problem we calculate a fast upper bound. 
	   If any sub problem has an upper bound which is less than or equal to the global lower bound, we can remove that sub problem from consideration in the remaining iterations of the algorithm. 
	 - This algorithm does not necessarily enumerate all cliques nor all Maximum Cliques. In particular, it is designed to return a single maximum clique assuming the solver is exact. 
	   However, the algorithm could be modified to include all maximum cliques found from solving each sub-problem.
	 - There are many assert statements in this function. These all serve as "sanity checks"; if any of them are tripped, something went wrong or an input was incorrect
	"""
    assert type(graph) is nx.Graph
    assert type(LIMIT) is int
    assert len(graph) != 0
    # print("=== Starting DBK Algorithm ===")
    G = graph.copy()
    if len(graph) <= LIMIT:
        leaf_subgraphs.append(len(graph))
        # print("=== Input Graph Size is Smaller than LIMIT ===")
        # print("=== Calling Solver Function ===")
        k = solver_function(graph)
        # print("=== Finished DBK Algorithm ===")
        return k, decomposition_cnt, leaf_subgraphs
    graph = remove_zero_degree_nodes(graph)
    k = mc_lower_bound(graph)
    graph = k_core_reduction(graph, len(k))
    if len(graph) == 0:
        # leaf_subgraphs.append(len(graph))
        return k, decomposition_cnt, leaf_subgraphs
    if len(graph) <= LIMIT:
        leaf_subgraphs.append(len(graph))
        # print("=== After K-core Reduction the Graph Size is Smaller than LIMIT ===")
        # print("=== Calling Solver Function ===")
        k = solver_function(graph)
        # print("=== Finished DBK Algorithm ===")
        return k, decomposition_cnt, leaf_subgraphs
    vertex_removal = {graph: []}
    subgraphs = [graph]
    while len(subgraphs) != 0:

        # Todo: This timeout is to make runtime more predictable and realistic for the final report
        if decomposition_cnt >= 10000: # we consider more than this as outlier
            print("=== Timeout ===")
            return k, decomposition_cnt, leaf_subgraphs

        SG = subgraphs.pop()  # Todo: this is where your subgraph gets selected - smart selection can help pruning
        SG = remove_zero_degree_nodes(SG)
        assert len(SG) != 0
        vcount = vertex_removal[SG]
        del vertex_removal[SG]
        if vertex_selection_func == "MIN":
            vertex = lowest_degree_vertex(SG)
        elif vertex_selection_func == "MAX":
            vertex = highest_degree_vertex(SG)
        elif vertex_selection_func == "RAND":
            vertex = random_vertex_selection(SG)
        else:
            vertex = smart_vertex_selection(SG, vertex_selection_func)
        # print("=== Partitioning Subgraph ===")
        SSG, SG = ch_partitioning(vertex, SG, vertex_selection_func)
        SG = remove_zero_degree_nodes(SG)
        SSG = remove_zero_degree_nodes(SSG)
        SG = k_core_reduction(SG, len(k) - len(
            vcount))  # Todo: you can compute which graph had less reduction as a reward signal
        SSG = k_core_reduction(SSG, len(k) - len(vcount + [vertex]))
        vertex_removal[SSG] = vcount + [vertex]
        vertex_removal[SG] = vcount
        #####################################################################################################
        if is_clique(G.subgraph(list(SSG.nodes()))) == True:
            assert is_clique(G.subgraph(list(SSG.nodes()) + vertex_removal[SSG])) == True
            if len(SSG) + len(vertex_removal[SSG]) > len(k):
                k = list(SSG.nodes()) + vertex_removal[SSG]
            del vertex_removal[SSG]
            SSG = nx.Graph()
        if is_clique(G.subgraph(list(SG.nodes()))) == True:
            assert is_clique(G.subgraph(list(SG.nodes()) + vertex_removal[SG])) == True
            if len(SG) + len(vertex_removal[SG]) > len(k):
                k = list(SG.nodes()) + vertex_removal[SG]
            del vertex_removal[SG]
            SG = nx.Graph()
        #####################################################################################################
        if len(SSG) != 0:
            SSG_lower = mc_lower_bound(SSG) + vertex_removal[SSG]
            assert is_clique(G.subgraph(SSG_lower)) == True
            if len(SSG_lower) > len(k):
                vcount = vertex_removal[SSG]
                del vertex_removal[SSG]
                k = SSG_lower
                SSG = k_core_reduction(SSG, len(k) - len(vcount))
                SSG = remove_zero_degree_nodes(SSG)
                vertex_removal[SSG] = vcount
            if len(SSG) != 0:
                SSG_upper = mc_upper_bound(SSG) + len(vertex_removal[SSG])
                if SSG_upper > len(k):
                    if len(SSG) <= LIMIT:
                        leaf_subgraphs.append(len(SSG))
                        # print("=== Calling Solver Function ===")
                        sub_solution_SSG = solver_function(SSG) + vertex_removal[SSG]
                        del vertex_removal[SSG]
                        assert is_clique(G.subgraph(sub_solution_SSG)) == True
                        if len(sub_solution_SSG) > len(k):
                            k = sub_solution_SSG
                    else:
                        subgraphs.append(SSG)
                else:
                    del vertex_removal[SSG]
        if len(SSG) == 0:
            # leaf_subgraphs.append(len(SSG))
            if SSG in list(vertex_removal.keys()):
                sub_solution_SSG = vertex_removal[SSG]
                del vertex_removal[SSG]
                assert is_clique(G.subgraph(sub_solution_SSG)) == True
                if len(sub_solution_SSG) > len(k):
                    k = sub_solution_SSG
        #####################################################################################################
        if len(SG) != 0:
            SG_lower = mc_lower_bound(SG) + vertex_removal[SG]
            assert is_clique(G.subgraph(SG_lower)) == True
            if len(SG_lower) > len(k):
                vcount = vertex_removal[SG]
                del vertex_removal[SG]
                k = SG_lower
                SG = k_core_reduction(SG, len(k) - len(vcount))
                SG = remove_zero_degree_nodes(SG)
                vertex_removal[SG] = vcount
            if len(SG) != 0:
                SG_upper = mc_upper_bound(SG) + len(vertex_removal[SG])
                if SG_upper > len(k):
                    if len(SG) <= LIMIT:
                        leaf_subgraphs.append(len(SG))
                        # print("=== Calling Solver Function ===")
                        sub_solution_SG = solver_function(SG) + vertex_removal[SG]
                        del vertex_removal[SG]
                        assert is_clique(G.subgraph(sub_solution_SG)) == True
                        if len(sub_solution_SG) > len(k):
                            k = sub_solution_SG
                    else:
                        subgraphs.append(SG)
                else:
                    del vertex_removal[SG]
        if len(SG) == 0:
            # leaf_subgraphs.append(len(SG))
            if SG in list(vertex_removal.keys()):
                sub_solution_SG = vertex_removal[SG]
                del vertex_removal[SG]
                assert is_clique(G.subgraph(sub_solution_SG)) == True
                if len(sub_solution_SG) > len(k):
                    k = sub_solution_SG
    assert len(vertex_removal) == 0
    # print("=== Finished DBK Algorithm ===")
    return k, decomposition_cnt, leaf_subgraphs
