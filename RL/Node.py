import math
import time
import random
from collections import defaultdict
from networkx import weisfeiler_lehman_graph_hash as graph_hash
from copy import deepcopy


class Node:
    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, nnet, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        # self.state = state
        self.id = id(mdp)  # graph_hash(state)

        # The Q function used to store state-action values
        self.nnet = nnet

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward

        # The action that generated this node
        self.action = action

    """ Select a node that is not fully expanded """

    def select(self):
        pass

    """ Expand a node if it is not a terminal node """

    def expand(self):
        pass

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child):
        pass

    """ Return the value of this node """

    def get_value(self):
        return self.nnet.get_max_q(self.mdp.last_subgraph)

    """ Get the number of visits to this state """

    def get_visits(self):
        # Todo: there may be a subtle bug here because hash is the same for isomorphic graphs thus sum_a N(s,a) != N(s)
        return Node.visits[self.id]
        # return sum([Node.visits[(self.id, action)] for action in self.mdp.get_actions(self.state)])

    """ Get the number of visits to this state-action pair """

    def get_sa_visits(self, action):
        return Node.visits[(self.id, action)]


class SingleAgentNode(Node):
    def __init__(
            self,
            mdp,  # see this as the state of your DBK algorithm
            parent,
            # state,
            nnet,
            reward=0.0,
            action=None,
    ):
        super().__init__(mdp, parent, nnet, reward, action)

        # A dictionary from actions to a set of node-probability pairs
        self.children = {}

    """ Return true if and only if all child actions have been expanded """

    def is_fully_expanded(self):  # Todo: maybe relax this condition
        valid_actions = self.mdp.get_actions()
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    """ Select a node that is not fully expanded """

    def select(self):
        if not self.is_fully_expanded() or self.mdp.is_terminal():
            return self
        else:
            actions = list(self.children.keys())  # actions that have been expanded

            # UCB
            Q_sa = self.nnet.predict(self.mdp.last_subgraph)  # (nodes,1)
            nodes_lst = list(self.mdp.last_subgraph.nodes)
            Cp = 1 / math.sqrt(2)
            Ns = self.get_visits()
            eps = 1e-8
            best_action = None
            curr_best_value = float("-inf")
            for action in actions:
                Nsa = self.get_sa_visits(action)
                action_value = Q_sa[nodes_lst.index(action), 0] + 2 * Cp * math.sqrt(math.log(Ns) / (Nsa + eps))
                if action_value > curr_best_value:
                    curr_best_value = action_value
                    best_action = action

            return self.children[best_action].select()  # Decomposition by one node is deterministic

    """ Expand a node if it is not a terminal node """

    def expand(self):
        # print(f"Expanded {len(self.children)}/{len(self.mdp.get_actions())} actions")
        if not self.mdp.is_terminal():
            # Randomly select an unexpanded action to expand
            assert len(self.children.keys() - self.mdp.get_actions()) == 0 # sanity check
            actions = self.mdp.get_actions() - self.children.keys()
            action = random.choice(list(actions))  # Todo: maybe I can use min vertex heuristic here

            self.children[action] = []

            return self.get_outcome_child(action)
        return self

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child):
        action = child.action

        Node.visits[self.id] = Node.visits[self.id] + 1
        Node.visits[(self.id, action)] = self.get_sa_visits(action) + 1

        # Todo: This update to qfunc can be noisy, maybe use a different update rule
        self.nnet.qfunction.train()
        action_idx = list(self.mdp.last_subgraph.nodes).index(action)
        q_value = self.nnet.qfunction(self.mdp.last_subgraph)[action_idx, 0]
        eps = 1e-8
        loss = (1 / (self.get_sa_visits(action) + eps)) * (
                reward - q_value
        )
        # print(f"Backpropagating loss {loss}")

        self.nnet.update(loss)

        if self.parent != None:
            self.parent.back_propagate(self.reward + reward, self)

    """ Simulate the outcome of an action, and return the child node """

    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities
        new_mdp = self.mdp.copy()  # I want mdp to be memory-less wrt to other node expansions
        reward = new_mdp.execute(action)
        # print("Executed action", action, " on", self.id, "with reward", reward)

        # This outcome has not occured from this state-action pair previously
        new_child = SingleAgentNode(
            new_mdp, self, self.nnet, reward, action
        )

        self.children[action] = new_child

        return self.children[action]  # resulting node which is state of DBK algorithm

    def copy(self):
        return deepcopy(self)
