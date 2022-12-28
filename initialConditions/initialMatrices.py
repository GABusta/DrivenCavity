import numpy as np
from quadFiniteElement.quad_finite_elements import OneGaussPoint


class InitialMatrixQuadElement:
    """
    Generation of the initial matrix, "x" and "y" coordinates
    """

    def __init__(self, mesh):
        self.KG = np.zeros((mesh.totalNumberNodes * 2, mesh.totalNumberNodes * 2))
        self.NG = np.zeros((mesh.totalNumberNodes * 2, mesh.totalNumberNodes * 2))
        self.RG = np.zeros((mesh.totalNumberNodes * 2, 1))
        self.V0 = np.zeros((mesh.totalNumberNodes * 2, 1))

    def assembly_elemental_quad_matrix(self, ke, ne, re, connections):
        """
        Assembly of the elemental matrices (ke,ne,re) into Global (KG, NG, RG), one by one \n
        :param ke, ne, re:
        :param connections: "i" elemental connections
        :return: self
        """
        for row_node, row in enumerate(connections):
            for column_node, column in enumerate(connections):

                # Global diffusion matrix - KG
                self.KG[row * 2 - 2, column * 2 - 2] += ke[
                    row_node * 2, column_node * 2
                ]

                self.KG[row * 2 - 2, column * 2 - 1] += ke[
                    row_node * 2, column_node * 2 + 1
                ]

                self.KG[row * 2 - 1, column * 2 - 2] += ke[
                    row_node * 2 + 1, column_node * 2
                ]

                self.KG[row * 2 - 1, column * 2 - 1] += ke[
                    row_node * 2 + 1, column_node * 2 + 1
                ]

                # Global convection matrix - NG
                self.NG[row * 2 - 2, column * 2 - 2] += ne[
                    row_node * 2, column_node * 2
                ]

                self.NG[row * 2 - 2, column * 2 - 1] += ne[
                    row_node * 2, column_node * 2 + 1
                ]

                self.NG[row * 2 - 1, column * 2 - 2] += ne[
                    row_node * 2 + 1, column_node * 2
                ]

                self.NG[row * 2 - 1, column * 2 - 1] += ne[
                    row_node * 2 + 1, column_node * 2 + 1
                ]

            # Global Results Array - RG
            self.RG[row * 2 - 2] += re[row_node * 2]
            self.RG[row * 2 - 1] += re[row_node * 2 + 1]

        return self

    def apply_boundary_conditions(self, vi, mesh):
        """
        Apply the boundary conditions for the driven cavity problem \n
        :param vi: initial velocity, in this case in the "x" direction
        :param mesh: mesh object with information about the mesh
        :return: self
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
        for node in velocity_zero_nodes:
            # Both directions
            self.RG[node * 2] = 0.0
            self.RG[node * 2 + 1] = 0.0

            self.KG[node * 2, :] = 0.0
            self.KG[:, node * 2] = 0.0
            self.KG[node * 2, node * 2] = 1.0

            self.KG[node * 2 + 1, :] = 0.0
            self.KG[:, node * 2 + 1] = 0.0
            self.KG[node * 2 + 1, node * 2 + 1] = 1.0

            self.NG[node * 2, :] = 0.0
            self.NG[:, node * 2] = 0.0
            self.NG[node * 2, node * 2] = 1.0

            self.NG[node * 2 + 1, :] = 0.0
            self.NG[:, node * 2 + 1] = 0.0
            self.NG[node * 2 + 1, node * 2 + 1] = 1.0

            self.V0[node * 2] = 0.0
            self.V0[node * 2 + 1] = 0.0

        for node in velocity_initial_nodes:
            # Only velocity in "x" direction
            self.V0[node * 2] = vi
            self.V0[node * 2 + 1] = 0.0
            for all_nodes in range(node, mesh.totalNumberNodes):
                if node != all_nodes:
                    # x - coordinate
                    self.RG[all_nodes * 2] += (
                        -self.KG[all_nodes * 2, node * 2] * vi
                        - self.NG[all_nodes * 2, node * 2] * vi
                    )

                    # y - coordinate
                    self.RG[all_nodes * 2 + 1] += 0.0
                else:
                    self.RG[all_nodes * 2] = vi
                    self.RG[all_nodes * 2 + 1] = 0.0

            self.KG[node * 2, :] = 0.0
            self.KG[:, node * 2] = 0.0
            self.KG[node * 2, node * 2] = 1.0

            self.KG[node * 2 + 1, :] = 0.0
            self.KG[:, node * 2 + 1] = 0.0
            self.KG[node * 2 + 1, node * 2 + 1] = 1.0

            self.NG[node * 2, :] = 0.0
            self.NG[:, node * 2] = 0.0
            self.NG[node * 2, node * 2] = 1.0

            self.NG[node * 2 + 1, :] = 0.0
            self.NG[:, node * 2 + 1] = 0.0
            self.NG[node * 2 + 1, node * 2 + 1] = 1.0

        return self

    @staticmethod
    def quad_elemental_matrix_generation(
        elements, parameters, properties, matrix_g, v0_k
    ):
        """
        Generation of the Elemental matrices, for a QUAD element with one Gauss point \n
        :param elements: information about Jacobian, shape functions
        :param parameters: of run
        :param properties: of the material
        :param matrix_g:
        :param v0_k:
        :return: object with elemental matrices
        """
        # Initial matrices
        degrees_freedom = elements.degrees_freedom
        mmt = elements.mmt
        I = elements.I
        mmt_mmt = elements.mmt_mmt

        # C -> matrix
        dHv_dx = elements.DH[:2, :] * elements.inv_Je[0, 0]
        dHv_dy = elements.DH[2:, :] * elements.inv_Je[1, 1]
        C = v0_k[0] * dHv_dx + v0_k[1] * dHv_dy

        # G -> matrix
        # already calculated

        # ne -> matrix (convective term "N")
        dV = elements.det_Je * elements.w
        ne = (
            np.dot(np.transpose(elements.H), (C + np.dot(matrix_g, elements.H)))
            * properties.rho
            * dV
        )

        # K -> matrix (diffusive term)
        k_lan = (
            np.dot(
                np.transpose(np.dot(elements.inv_Je4, elements.DH4)),
                np.dot(mmt, np.dot(elements.inv_Je4, elements.DH4)),
            )
            * properties.ian
            * dV
        )

        k_vv = (
            np.dot(
                np.transpose(np.dot(elements.inv_Je4, elements.DH4)),
                np.dot(mmt_mmt, np.dot(elements.inv_Je4, elements.DH4)),
            )
            * dV
            * 2.0
            * properties.viscocity
        )

        # re -> results array (F)
        re = (
            np.dot(np.transpose(elements.H), np.dot(matrix_g, np.transpose(v0_k)))
            * properties.rho
            * dV
        )

        # ke -> return matrix
        ke = k_vv - k_lan

        return ke, ne, re

    def quad_global_matrix_assembly(self, properties, parameters, mesh):
        elements = OneGaussPoint().quad_elements()
        # global_matrix = deepcopy(matrices)

        for i in range(mesh.totalNumberElements):
            # Elemental connectivity --> element = [node1, node2, node3, node4]
            conn = [
                mesh.connections[i, 0],
                mesh.connections[i, 1],
                mesh.connections[i, 2],
                mesh.connections[i, 3],
            ]

            # Average velocity per Element   -->   vo_k = [Vx , Vy]
            v0_x = np.array(
                [
                    self.V0[conn[0] * 2 - 2],
                    self.V0[conn[1] * 2 - 2],
                    self.V0[conn[2] * 2 - 2],
                    self.V0[conn[3] * 2 - 2],
                ]
            )

            v0_y = np.array(
                [
                    self.V0[conn[0] * 2 - 1],
                    self.V0[conn[1] * 2 - 1],
                    self.V0[conn[2] * 2 - 1],
                    self.V0[conn[3] * 2 - 1],
                ]
            )

            v0_k = np.array([0.25 * np.sum(v0_x), 0.25 * np.sum(v0_y)])

            # Velocity gradient per Element "G"
            # G = [[ d(v0k)/dx  , d(v0k)/dxy ],
            #      [ d(v0k)/dyx , d(v0k)/dy  ]]

            d_v0k_x = float(
                elements.inv_Je[0][0]
                * 4
                * (
                    v0_x[0] * elements.DH[0][0]
                    + v0_x[1] * elements.DH[0][2]
                    + v0_x[2] * elements.DH[0][4]
                    + v0_x[3] * elements.DH[0][6]
                )
            )

            d_v0k_y = float(
                elements.inv_Je[1][1]
                * 4
                * (
                    v0_y[0] * elements.DH[2][1]
                    + v0_y[1] * elements.DH[2][3]
                    + v0_y[2] * elements.DH[2][5]
                    + v0_y[3] * elements.DH[2][7]
                )
            )

            d_v0k_xy = float(
                elements.inv_Je[1][1]
                * 4
                * (
                    v0_x[0] * elements.DH[2][0]
                    + v0_x[1] * elements.DH[2][2]
                    + v0_x[2] * elements.DH[2][4]
                    + v0_x[3] * elements.DH[2][6]
                )
            )

            d_v0k_yx = float(
                elements.inv_Je[0][0]
                * 4
                * (
                    v0_y[0] * elements.DH[0][1]
                    + v0_y[1] * elements.DH[0][3]
                    + v0_y[2] * elements.DH[0][5]
                    + v0_y[3] * elements.DH[0][7]
                )
            )

            matrix_g = np.array([[d_v0k_x, d_v0k_xy], [d_v0k_yx, d_v0k_y]])

            # Elemental Matrices
            ke, ne, re = self.quad_elemental_matrix_generation(
                elements=elements,
                parameters=parameters,
                properties=properties,
                matrix_g=matrix_g,
                v0_k=v0_k,
            )

            # Global Matrices
            self.assembly_elemental_quad_matrix(
                ke, ne, re, connections=mesh.connections[i]
            )

        return self


# --- test for initial matrix script ---
if __name__ == "__main__":
    from meshing.meshFile import TestMeshData

    mesh_test = TestMeshData()
    matrix_test = InitialMatrixQuadElement(mesh_test)
    a = 1
