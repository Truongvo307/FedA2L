import random

import networkx as nx

from .base import Topology


class KConnected(Topology):
    def __init__(self, num_nodes, k):
        if k >= num_nodes:
            raise ValueError("k must be less than the total number of nodes.")
        if num_nodes * k % 2 != 0:
            raise ValueError("num_nodes * k must be even for a valid KConnected.")

        self.num_nodes = num_nodes
        self.k = k
        self.neighbors = self._gen()

    def _gen(self):
        G_undirected = nx.random_regular_graph(self.k, self.num_nodes)
        G = G_undirected.to_directed()

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            while len(neighbors) < self.k:
                target = random.choice(list(set(G.nodes()) - set(neighbors)))
                G.add_edge(node, target)
                neighbors.append(target)

        neighbors = {node: list(G.neighbors(node)) for node in sorted(G.nodes())}
        return neighbors


if __name__ == "__main__":
    num_nodes = 10
    topology = KConnected(num_nodes, 5)
    print(f"Node has neighbors: {topology.neighbors}")
