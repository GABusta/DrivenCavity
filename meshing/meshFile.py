import matplotlib.pyplot as plt
import numpy as np


class TestMeshData:
    """
    Initial mesh data, for a square cavity \n
    available methods: \n
    - generation() \n
    - print_mesh()
    """

    def __init__(self, n=20, large=1.0):
        self.divisions = n
        self.nodes = n + 1
        self.dx = large / self.divisions
        self.distance = []
        self.coordinates = []
        self.connections = []
        self.totalNumberNodes = self.nodes ** 2
        self.totalNumberElements = self.divisions ** 2

    def generation(self):
        """
        Generation of coordinates and connections
        :return: self
        """

        # Generation of coordinates
        self.distance = [i * self.dx for i in range(self.nodes)]
        k = 1
        for i in range(self.nodes):
            for j in range(self.nodes):
                self.coordinates.append([self.distance[j], self.distance[i]])

        self.distance = np.array(self.distance[:])
        self.coordinates = np.array(self.coordinates[:])

        # Generation of connections
        cont = 0
        for i in range(self.nodes - 1):
            cont += 21
            for j in range(self.nodes - 1):
                self.connections.append(
                    [
                        cont + j + 2,
                        cont + j + 1,
                        cont - self.nodes + j + 1,
                        cont - self.nodes + j + 2,
                    ]
                )
        self.connections = np.array(self.connections[:])

        return self

    def print_mesh(self):
        plt.scatter(x=self.coordinates[:, 0], y=self.coordinates[:, 1], c="blue")
        plt.show()


# --- test for the meshing script ---
if __name__ == "__main__":
    mesh = TestMeshData().generation()
    mesh.print_mesh()
    a = 1
