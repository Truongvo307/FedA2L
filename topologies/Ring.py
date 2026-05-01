from .base import Topology


class Ring(Topology):
    def _gen(self):
        neighbors = {}
        for node in range(self.num_nodes):
            # Connect the node to its left and right neighbors in a circular manner
            left_neighbor = (node - 1) % self.num_nodes
            right_neighbor = (node + 1) % self.num_nodes
            neighbors[node] = [left_neighbor, right_neighbor]
        return neighbors
