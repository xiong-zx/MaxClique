# %%
import matplotlib.pyplot as plt
import numpy
import networkx as nx
import random
random.seed(392)
numpy.random.seed(483)


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
        if len(list(set(x) & set(y))) <= (k-2):
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
    if int(m) == int((n*(n-1))/float(2)):
        return True
    else:
        return False


def ch_partitioning(vertex, G):
    """
    INPUT:
     - "vertex" splitting vertex
     - "G" Networkx Undirected Graph
    OUTPUT:
     - "SSG" Left subgraph after partitioning
     - "SG" Right subgraph after partitioning
    """
    n = list(G.neighbors(vertex))
    Gp = []
    for iter in list(G.edges()):
        if iter[0] in n:
            if iter[1] in n:
                Gp.append(iter)
    G.remove_node(vertex)
    return nx.Graph(Gp), G


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
     - "i" node that has the highest degree in the graph
    """
    degrees = [graph.degree(a) for a in list(graph.nodes())]
    maximum = max(degrees)
    for i in list(graph.nodes()):
        if graph.degree(i) == maximum:
            return i
        
def random_vertex(graph):
    """
    INPUT:
     - "graph" Networkx Undirected Graph
    OUTPUT:
     - "i" node that has the highest degree in the graph
    """
    return random.choice(list(graph.nodes()))

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


def DBK(graph, LIMIT, solver_function):
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
    print("=== Starting DBK Algorithm ===")
    G = graph.copy()
    num_of_atoms = 0
    iteration = 0
    if len(graph) <= LIMIT:
        print("=== Input Graph Size is Smaller than LIMIT ===")
        num_of_atoms += 1
        print("=== Finished DBK Algorithm ===")
        return num_of_atoms, iteration
    graph = remove_zero_degree_nodes(graph)
    k = mc_lower_bound(graph)
    graph = k_core_reduction(graph, len(k))
    if len(graph) == 0:
        return k
    if len(graph) <= LIMIT:
        print("=== After K-core Reduction the Graph Size is Smaller than LIMIT ===")
        num_of_atoms += 1
        print("=== Finished DBK Algorithm ===")
        return num_of_atoms, iteration
    vertex_removal = {graph: []}
    subgraphs = [graph]
    while len(subgraphs) != 0:
        iteration += 1
        SG = subgraphs.pop()
        SG = remove_zero_degree_nodes(SG)
        assert len(SG) != 0
        vcount = vertex_removal[SG]
        del vertex_removal[SG]
        vertex = lowest_degree_vertex(SG)
        SSG, SG = ch_partitioning(vertex, SG)
        SG = remove_zero_degree_nodes(SG)
        SSG = remove_zero_degree_nodes(SSG)
        SG = k_core_reduction(SG, len(k)-len(vcount))
        SSG = k_core_reduction(SSG, len(k)-len(vcount+[vertex]))
        vertex_removal[SSG] = vcount+[vertex]
        vertex_removal[SG] = vcount
        #####################################################################################################
        if is_clique(G.subgraph(list(SSG.nodes()))) == True:
            assert is_clique(G.subgraph(
                list(SSG.nodes())+vertex_removal[SSG])) == True
            if len(SSG)+len(vertex_removal[SSG]) > len(k):
                k = list(SSG.nodes())+vertex_removal[SSG]
            del vertex_removal[SSG]
            SSG = nx.Graph()
        if is_clique(G.subgraph(list(SG.nodes()))) == True:
            assert is_clique(G.subgraph(
                list(SG.nodes())+vertex_removal[SG])) == True
            if len(SG)+len(vertex_removal[SG]) > len(k):
                k = list(SG.nodes())+vertex_removal[SG]
            del vertex_removal[SG]
            SG = nx.Graph()
        #####################################################################################################
        if len(SSG) != 0:
            SSG_lower = mc_lower_bound(SSG)+vertex_removal[SSG]
            assert is_clique(G.subgraph(SSG_lower)) == True
            if len(SSG_lower) > len(k):
                vcount = vertex_removal[SSG]
                del vertex_removal[SSG]
                k = SSG_lower
                SSG = k_core_reduction(SSG, len(k)-len(vcount))
                SSG = remove_zero_degree_nodes(SSG)
                vertex_removal[SSG] = vcount
            if len(SSG) != 0:
                SSG_upper = mc_upper_bound(SSG)+len(vertex_removal[SSG])
                if SSG_upper > len(k):
                    if len(SSG) <= LIMIT:
                        print("=== Terminal Subgraph Found ===")
                        num_of_atoms += 1
                        del vertex_removal[SSG]
                    else:
                        subgraphs.append(SSG)
                else:
                    del vertex_removal[SSG]
        if len(SSG) == 0:
            if SSG in list(vertex_removal.keys()):
                sub_solution_SSG = vertex_removal[SSG]
                del vertex_removal[SSG]
                assert is_clique(G.subgraph(sub_solution_SSG)) == True
                if len(sub_solution_SSG) > len(k):
                    k = sub_solution_SSG
        #####################################################################################################
        if len(SG) != 0:
            SG_lower = mc_lower_bound(SG)+vertex_removal[SG]
            assert is_clique(G.subgraph(SG_lower)) == True
            if len(SG_lower) > len(k):
                vcount = vertex_removal[SG]
                del vertex_removal[SG]
                k = SG_lower
                SG = k_core_reduction(SG, len(k)-len(vcount))
                SG = remove_zero_degree_nodes(SG)
                vertex_removal[SG] = vcount
            if len(SG) != 0:
                SG_upper = mc_upper_bound(SG)+len(vertex_removal[SG])
                if SG_upper > len(k):
                    if len(SG) <= LIMIT:
                        print("=== Terminal Subgraph Found ===")
                        num_of_atoms += 1
                        # sub_solution_SG = solver_function(SG)+vertex_removal[SG]
                        del vertex_removal[SG]
                    else:
                        subgraphs.append(SG)
                else:
                    del vertex_removal[SG]
        if len(SG) == 0:
            if SG in list(vertex_removal.keys()):
                sub_solution_SG = vertex_removal[SG]
                del vertex_removal[SG]
                assert is_clique(G.subgraph(sub_solution_SG)) == True
                if len(sub_solution_SG) > len(k):
                    k = sub_solution_SG
    assert len(vertex_removal) == 0
    print("=== Finished DBK Algorithm ===")
    return num_of_atoms, iteration

# %%

def find_separators_within_size(graph, size_limit):
    separators = []
    nodes = set(graph.nodes())
    for node in nodes:
        nodes.remove(node)
        for connected_node in nodes:
            subgraph = graph.subgraph(nodes - {connected_node})
            if not nx.is_connected(subgraph) and len(subgraph) <= size_limit:
                separators.append(list(nodes - {connected_node}))
        nodes.add(node)
    return separators


def remove_overlapping_sets(list_of_sets):
    non_overlapping_sets = []
    for s in list_of_sets:
        if not any(set(s).intersection(set(existing_set)) for existing_set in non_overlapping_sets):
            non_overlapping_sets.append(s)
    return non_overlapping_sets


def get_largest_component(graph):
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest_component = graph.subgraph(components[0])
    return largest_component


def decompose_graph(graph, nodes_to_remove):
    modified_graph = graph.copy()
    og_graph = graph.copy()
    modified_graph.remove_nodes_from(nodes_to_remove)
    connected_components = list(nx.connected_components(modified_graph))
    result_components = []
    for component in connected_components:
        component = component.union(nodes_to_remove)
        component_graph = og_graph.subgraph(component)
        component_graph = get_largest_component(component_graph)
        result_components.append(component_graph)
    return result_components


def maximum_clique_exact_solve_np_hard(G):
    max_clique_number = nx.graph_clique_number(G)
    cliques = nx.find_cliques(G)
    for cl in cliques:
        if len(cl) == max_clique_number:
            return cl

# %%
if __name__ == "__main__":
    G = nx.gnp_random_graph(60, 0.6)
    num_of_subgraphs, num_of_iterations = DBK(
        G, 20, maximum_clique_exact_solve_np_hard)
    print("Number of iterations to termination:", num_of_iterations)
    print("Resulting number of subgraphs:", num_of_subgraphs)