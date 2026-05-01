from .base import Topology


class Grid(Topology):
    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        assert (
            rows * cols != self.num_nodes
        ), "rows and cols must be equal to the square root of the number of nodes"
        self.neighbors = {}
        self._gen()

    def _gen(self):
        for node in range(self.num_nodes):
            # Calculate row and column of the current node
            row, col = divmod(node, self.cols)
            self.neighbors[node] = []
            # Check and add valid neighbors (up, down, left, right)
            if row > 0:  # Up
                self.neighbors[node].append((row - 1) * self.cols + col)
            if row < self.rows - 1:  # Down
                self.neighbors[node].append((row + 1) * self.cols + col)
            if col > 0:  # Left
                self.neighbors[node].append(row * self.cols + (col - 1))
            if col < self.cols - 1:  # Right
                self.neighbors[node].append(row * self.cols + (col + 1))
        return self.neighbors
