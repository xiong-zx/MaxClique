# from DBK import DBK as fastDBK
# from simpleDBK import DBK
from org_DBK import DBK
import networkx as nx
import random
import torch
from torch_geometric.utils.convert import to_networkx
import time
from tqdm import tqdm

random.seed(11)
import numpy
numpy.random.seed(11)

def maximum_clique_exact_solve_np_hard(G):
	max_clique_number = nx.graph_clique_number(G)
	cliques = nx.find_cliques(G)
	for cl in cliques:
		if len(cl) == max_clique_number:
			return cl

if __name__ == '__main__':
	# g = nx.Graph()
	# g.add_nodes_from([1,2,3,4])
	# g.add_edges_from([(1,2),(2,3),(3,4),(4,2)])
	#
	# print("DBK solution", DBK(g, 3, maximum_clique_exact_solve_np_hard))
	# print("Exact solution", nx.graph_clique_number(g))

	# for i in range(10):
	# 	G = nx.gnp_random_graph(random.randint(66, 80), random.uniform(0.01, 0.99),seed=11)
	# 	#G.nodes
	# 	G2 = G.copy()
	# 	print(len(G))
	# 	# DBK_solution,decompositionCnt = DBK(G, 65, maximum_clique_exact_solve_np_hard)
	# 	fastDBK_solution,fastDecompositionCnt = fastDBK(G2.copy(), 30, maximum_clique_exact_solve_np_hard)
	# 	# exact_solution = nx.graph_clique_number(G2)
	# 	# # print("DBK solution", len(DBK_solution), "decomposition count:", decompositionCnt)
	# 	# print("Fast DBK solution", len(fastDBK_solution), "decomposition count:", fastDecompositionCnt)
	# 	# print("Exact solution", exact_solution)
	# 	# # assert len(DBK_solution) == len(fastDBK_solution)
	# 	# assert len(fastDBK_solution) == exact_solution

	test_dataset = torch.load('data/test_200nodes_100graphs.pt')
	test_dataset = test_dataset[:10]

	start = time.time()
	for pyg_graph in tqdm(test_dataset[2:5]):
		nx_graph = to_networkx(pyg_graph, to_undirected=True)
		# print(nx_graph._nodes)
		DBK(nx_graph.copy(), 190, maximum_clique_exact_solve_np_hard)
	end = time.time()
	print(f"Time to run DBK on 200-node graph: {(end - start)}")



