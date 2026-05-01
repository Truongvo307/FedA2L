import networkx as nx

from .base import Topology


class Star(Topology):
    def _gen(self):
        neighbors = {0: []}
        for node in range(self.num_nodes):
            if node == 0:
                pass
            else:
                neighbors[node] = [0]
                neighbors[0].append(node)
        return neighbors
