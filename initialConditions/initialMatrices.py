import numpy as np


class InitialMatrix:
    """
    Generation of the initial matrix, "x" and "y" coordinates
    """

    def __init__(self, mesh):
        self.KG = np.zeros((mesh.totalNumberNodes * 2, mesh.totalNumberNodes * 2))
        self.NG = np.zeros((mesh.totalNumberNodes * 2, mesh.totalNumberNodes * 2))
        self.RG = np.zeros((mesh.totalNumberNodes * 2, 1))
        self.V0 = np.zeros((mesh.totalNumberNodes * 2, 1))

    def assembly_quad_elements(self, elemental_matrix, connections):
        """
        Assembly of the Global matrices (KG, NG, RG)
        :param elemental_matrix:
        :param connections: "i" elemental connections
        :return:
        """
        for row_node, row in enumerate(connections):
            for column_node, column in enumerate(connections):

                # Global diffussion matrix - KG
                self.KG[row * 2 - 2, column * 2 - 2] += elemental_matrix.ke[
                    row_node * 2 - 2, column_node * 2 - 1
                ]

                self.KG[row * 2 - 2, column * 2 - 1] += elemental_matrix.ke[
                    row_node * 2 - 2, column_node * 2
                ]

                self.KG[row * 2 - 1, column * 2 - 1] += elemental_matrix.ke[
                    row_node * 2, column_node * 2 - 1
                ]

                self.KG[row * 2 - 1, column * 2 - 1] += elemental_matrix.ke[
                    row_node * 2, column_node * 2
                ]

                # Global convection matrix - NG
                self.NG[row * 2 - 2, column * 2 - 2] += elemental_matrix.ne[
                    row_node * 2 - 1, column_node * 2 - 1
                ]

                self.NG[row * 2 - 2, column * 2 - 1] += elemental_matrix.ne[
                    row_node * 2 - 1, column_node * 2
                ]

                self.NG[row * 2 - 1, column * 2 - 2] += elemental_matrix.ne[
                    row_node * 2, column_node * 2 - 1
                ]

                self.NG[row * 2 - 1, column * 2 - 1] += elemental_matrix.ne[
                    row_node * 2, column_node * 2
                ]

            # Global Results Array - RG
            self.RG[row * 2 - 2] += elemental_matrix.re[row_node * 2 - 1]
            self.RG[row * 2 - 1] += elemental_matrix.re[row_node * 2]

        return self

    def boundary_conditions(self, vi, mesh):
        """
        Apply the boundary conditions for the driven cavity problem \n
        :param vi: initial velocity, in this case in the "x" direction
        :param mesh: mesh object with information about the mesh
        :return: matrix object, with the BC's applied
        """

        # ---> nodes presented in matrix, with Dirichlet conditions
        velocity_initial_nodes = [
            node
            for node in range(mesh.totalNumberNodes - mesh.nodes, mesh.totalNumberNodes)
        ]

        # ---> BC's on sides and bottom
        velocity_zero_nodes = [node for node in range(0, mesh.nodes)]
        velocity_zero_nodes.extend(
            [node for node in range(0, mesh.totalNumberNodes - 1, mesh.nodes)]
        )
        velocity_zero_nodes.extend(
            [node for node in range(mesh.nodes - 1, mesh.totalNumberNodes, mesh.nodes)]
        )

        # ---> Replace initial and boundary conditions in matrix
        for node in velocity_initial_nodes:
            # Only velocity in "x" direction
            self.V0[node * 2] = vi
            for all_nodes in range(node, mesh.totalNumberNodes):
                if node != all_nodes:
                    self.RG[all_nodes * 2] += -self.KG[all_nodes * 2, node * 2] * vi - self.NG[all_nodes * 2, node * 2] * vi
                else:
                    self.RG[all_nodes * 2] = vi

            self.KG[node * 2, :] = 0.0
            self.KG[:, node * 2] = 0.0
            self.KG[node * 2, node * 2] = 1.0

            self.NG[node * 2, :] = 0.0
            self.NG[:, node * 2] = 0.0
            self.NG[node * 2, node * 2] = 1.0

        for node in velocity_zero_nodes:
            # Both directions
            self.RG[node * 2] = 0.0
            self.RG[node * 2 + 1] = 0.0

            self.KG[node * 2, :] = 0.0
            self.KG[:, node * 2] = 0.0
            self.KG[node * 2, node * 2] = 1.0

            self.NG[node * 2, :] = 0.0
            self.NG[:, node * 2] = 0.0
            self.NG[node * 2, node * 2] = 1.0

            self.KG[node * 2 + 1, :] = 0.0
            self.KG[:, node * 2 + 1] = 0.0
            self.KG[node * 2 + 1, node * 2 + 1] = 1.0

            self.NG[node * 2 + 1, :] = 0.0
            self.NG[:, node * 2 + 1] = 0.0
            self.NG[node * 2 + 1, node * 2 + 1] = 1.0

        return self


# --- test for initial matrix script ---
if __name__ == "__main__":
    from meshing.meshFile import MeshData

    mesh_test = MeshData()
    matrix_test = InitialMatrix(mesh_test)
    a = 1
