import matplotlib.pyplot as plt
import numpy as np


class MeshData:
    def __init__(self, n=20, large=1.0):
        self.divisions = n
        self.dx = large / self.divisions
        self.distance = []
        self.coordinates = []
        self.connections = []
        self.numberNodes = self.divisions * self.divisions

    def generation(self):

        # Generation of coordinates
        self.distance = [i * self.dx for i in range(self.divisions)]
        k = 1
        for i in range(self.divisions):
            for j in range(self.divisions):
                self.coordinates.append([self.distance[j], self.distance[i]])

        # Generation of connections
        cont = 0
        for i in range(self.divisions):
            cont += 20
            for j in range(self.divisions):
                self.connections.append(
                    [
                        cont + j + 1,
                        cont + j,
                        cont - self.divisions + j,
                        cont - self.divisions + j + 1,
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
