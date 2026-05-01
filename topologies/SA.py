from .base import Topology


class StandAlone(Topology):
    def _gen(self):
        return {node: [] for node in range(self.num_nodes)}
