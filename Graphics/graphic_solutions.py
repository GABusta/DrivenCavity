import matplotlib.pyplot as plt
import numpy as np


class PlotMatrices:
    @staticmethod
    def plot_streamline_quad_element(velocity, mesh):
        #  no est'an bien aplicadas las vel. de borde
        vy = np.array([vel for i, vel in enumerate(velocity) if (i + 1) % 2 == 0])
        vx = np.array([vel for i, vel in enumerate(velocity) if (i + 1) % 2 != 0])

        x = mesh.coordinates[:, 0].reshape(mesh.nodes, mesh.nodes)
        y = mesh.coordinates[:, 1].reshape(mesh.nodes, mesh.nodes)

        vx_new = vx.reshape(mesh.nodes, mesh.nodes)
        vy_new = vy.reshape(mesh.nodes, mesh.nodes)

        mag = np.sqrt(vx ** 2 + vy ** 2)
        mag_new = mag.reshape(mesh.nodes, mesh.nodes)

        fig, ax = plt.subplots(1)
        q = ax.quiver(x, y, vx_new, vy_new, mag, cmap="plasma")
        fig.colorbar(q, label="magnitude")

        # plt.contour(x=mesh.coordinates[:, 0], y=mesh.coordinates[:, 1], velocity)
        plt.show()
        a = 1
