import matplotlib.pyplot as plt
import numpy as np


class MeshData:
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

    def generation(self):

        # Generation of coordinates
        self.distance = [i * self.dx for i in range(self.nodes)]
        k = 1
        for i in range(self.nodes):
            for j in range(self.nodes):
                self.coordinates.append([self.distance[j], self.distance[i]])

        # Generation of connections
        cont = 0
        for i in range(self.nodes):
            cont += 20
            for j in range(self.nodes):
                self.connections.append(
                    [
                        cont + j + 1,
                        cont + j,
                        cont - self.nodes + j,
                        cont - self.nodes + j + 1,
                    ]
                )
        # Generate Dirichlet initial conditions

        return self

    def print_mesh(self):
        x, y = [], []
        for i in range(len(self.coordinates)):
            x.append(self.coordinates[i][0])
            y.append(self.coordinates[i][1])

        plt.scatter(x=x, y=y, c="blue")
        plt.show()


if __name__ == "__main__":
    # test for the meshing script
    mesh = MeshData().generation()
    mesh.print_mesh()
    a = 1
