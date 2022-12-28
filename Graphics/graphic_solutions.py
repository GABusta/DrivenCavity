import matplotlib.pyplot as plt
import numpy as np


class PlotMatrices:
    @staticmethod
    def plot_streamline_quad_element(velocity, mesh):
        vy = np.array([vel for i, vel in enumerate(velocity) if (i + 1) % 2 == 0])
        vx = np.array([vel for i, vel in enumerate(velocity) if (i + 1) % 2 != 0])

        x = mesh.coordinates[:, 0].reshape(mesh.nodes, mesh.nodes)
        y = mesh.coordinates[:, 1].reshape(mesh.nodes, mesh.nodes)

        vx_new = vx.reshape(mesh.nodes, mesh.nodes)
        vy_new = vy.reshape(mesh.nodes, mesh.nodes)

        mag = np.sqrt(vx ** 2 + vy ** 2)
        mag_new = mag.reshape(mesh.nodes, mesh.nodes)

        plt.figure(figsize=(50, 50))
        fig, ax = plt.subplots(1)
        q = ax.quiver(x, y, vx_new, vy_new, mag, cmap="plasma")
        fig.colorbar(q, label="Velocity vectors")
        ax.set_aspect("equal")
        plt.title("Driven cavity solution - Nonlinear Problem")
        plt.xlabel("Distance [m]")

        plt.savefig("images/solution.png")
