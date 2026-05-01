import math

from .base import Topology


class HyperCube(Topology):
    def _gen(self):
        neighbors = {}
        for node in range(self.num_nodes):
            assert (
                self.num_nodes & (self.num_nodes - 1)
            ) == 0, "self.num_nodes must be a power of 2."
            log = int(math.log2(self.num_nodes))
            neighbors[node] = [node ^ (2**j) for j in range(0, log)]

        return neighbors
