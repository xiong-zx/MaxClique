# MaxClique
This is the Github Repo for the course project (MIE1666): A machine learning enhanced approach to decomposition for maximum clique.

xzx folder includes the vertex separator identification code.
ImitationLearning includes the imitation learning model code.
ReinforcementLearning includes the reinforcement learning model code.

Datasets are stored in `data/` directory while the `examples/` directory contains some graphs for experiments. 
Inside, the `200` is the number of nodes and the decimals `0.01,0.1,0.5` are density parameters.

Currently, the graphs in `examples/200/` have been solved using `max_clique()` function (an approximate algorithm provided by `networkx` package). 
Results are saved in `examples/200/max_cliques.csv`.
