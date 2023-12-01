import math
import time
import random
from collections import defaultdict
from Node import SingleAgentNode


class MCTS:
    def __init__(self, mdp, nnet):
        self.mdp = mdp
        self.nnet = nnet
        self.idx = 0

    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """

    def run(self, timeout, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()

        if self.mdp.is_terminal():
            return root_node

        start_time = time.time()
        current_time = time.time()
        while current_time - start_time < timeout:

            # Find a state node to expand
            selected_node = root_node.select()  # self if not fully expanded else best UCB node
            if not selected_node.mdp.is_terminal():
                child = selected_node.expand()
                child_mdp = child.mdp.copy()

                if child_mdp.is_terminal(): # Todo: experimental
                    reward = child.reward
                else:
                    reward = self.simulate(child_mdp)
                # reward = self.simulate(child_mdp)

                assert len(child_mdp.vertex_removal) == 0
                selected_node.back_propagate(reward, child)
            else:
                selected_node.parent.back_propagate(selected_node.reward, selected_node) # Todo: experimental

            current_time = time.time()

        return root_node

    """ Create a root node representing an initial state """

    def create_root_node(self):
        return SingleAgentNode(self.mdp, None, self.nnet)

    """ Choose a random action. Heustics can be used here to improve simulations. """

    def choose(self, mdp):
        return random.choice(mdp.get_actions())

    # """ Simulate until a terminal state """

    def simulate(self, child_mdp):
        cumulative_reward = 0.0
        depth = 0
        while not child_mdp.is_terminal():
            # Choose an action to execute
            action = self.choose(child_mdp)

            # Execute the action
            reward = child_mdp.execute(action)
            # Discount the reward
            cumulative_reward += pow(child_mdp.get_discount_factor(), depth) * reward
            depth += 1

        return cumulative_reward
