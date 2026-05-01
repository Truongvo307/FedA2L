from .base import Topology


class Tree(Topology):
    def _gen(self):
        neighbors = {}
        for node in range(self.num_nodes):
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            neighbors[node] = []

            if left_child < self.num_nodes:
                neighbors[node].append(left_child)
            if right_child < self.num_nodes:
                neighbors[node].append(right_child)

            if node != 0:
                parent = (node - 1) // 2
                neighbors[node].append(parent)

        return neighbors
