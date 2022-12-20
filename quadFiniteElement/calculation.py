import numpy as np


def calculation(properties, parameters, mesh, matrices):
    elements = GaussPoint().quad_elements()
    for i in range(mesh.totalNumberElements):
        # Elemental connectivity --> element = [node1, node2, node3, node4]
        conn = [
            mesh.connections[i][0],
            mesh.connections[i][1],
            mesh.connections[i][2],
            mesh.connections[i][3],
        ]

        # Average velocity per Element   -->   vo_k = [Vx , Vy]
        v0_x = np.array(
            [
                matrices.V0[conn[0] * 2 - 1],
                matrices.V0[conn[1] * 2 - 1],
                matrices.V0[conn[2] * 2 - 1],
                matrices.V0[conn[3] * 2 - 1],
            ]
        )

        v0_y = np.array(
            [
                matrices.V0[conn[0] * 2],
                matrices.V0[conn[1] * 2],
                matrices.V0[conn[2] * 2],
                matrices.V0[conn[3] * 2],
            ]
        )

        v0_k = [0.25 * np.sum(v0_x), 0.25 * np.sum(v0_y)]

        # Velocity gradient per Element "G"
        # G = [[ d(v0k)/dx  , d(v0k)/dxy ],
        #      [ d(v0k)/dyx , d(v0k)/dy  ]]

        d_v0k_x = (
            elements.inv_Je[0][0]
            * 4
            * (
                v0_x[0] * elements.DH[0][0]
                + v0_x[1] * elements.DH[0][2]
                + v0_x[2] * elements.DH[0][4]
                + v0_x[3] * elements.DH[0][6]
            )
        )

        d_v0k_y = (
            elements.inv_Je[1][1]
            * 4
            * (
                v0_y[0] * elements.DH[2][1]
                + v0_y[1] * elements.DH[2][3]
                + v0_y[2] * elements.DH[2][5]
                + v0_y[3] * elements.DH[2][7]
            )
        )

        d_v0k_xy = (
            elements.inv_Je[1][1]
            * 4
            * (
                v0_x[0] * elements.DH[2][0]
                + v0_x[1] * elements.DH[2][2]
                + v0_x[2] * elements.DH[2][4]
                + v0_x[3] * elements.DH[2][6]
            )
        )

        d_v0k_yx = (
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
        elemental_matrices = QuadElement().matrix_generation(
            elements, parameters, properties, matrix_g, v0_k
        )

        # Global Matrices

    return elements


class QuadElement:
    def __init__(self):
        self.ne = []
        self.re = []
        self.ke = []

    def matrix_generation(self, elements, parameters, properties, matrix_g, v0_k):
        degrees_freedom = 8

        # generation of -->  m.transpose(m)
        mmt = np.array(
            [
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
            ]
        )

        # generation --> transpose( I - m.transpose(m)/3 ) . (I - m.transpose(m)/3 )
        I = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        mmt_mmt = np.matmul(np.transpose(I - mmt / 3), I - mmt / 3)

        # Initial matrices
        k_lan = np.zeros((degrees_freedom, degrees_freedom))
        k_vv = np.zeros((degrees_freedom, degrees_freedom))
        ne = np.zeros((degrees_freedom, degrees_freedom))
        re = np.zeros((degrees_freedom, 1))

        # C - matrix
        dHv_dx = elements.DH[:2, :] * elements.inv_Je[0, 0]
        dHv_dy = elements.DH[2:, :] * elements.inv_Je[1, 1]
        C = v0_k[0] * dHv_dx + v0_k[1] * dHv_dy

        # G - matrix
        # already calculated

        # N - matrix (convective term)
        dV = elements.det_Je * elements.w
        self.ne = (
            np.transpose(elements.H)
            * (C + np.matmul(matrix_g, elements.H))
            * properties.rho
            * dV
        )
        # K - matrix (diffusive term)
        k_lan = (
            np.transpose(np.matmul(elements.inv_Je4, elements.DH4))
            * mmt
            * elements.inv_Je4
            * elements.DH4
            * properties.ian
            * dV
        )
        k_vv = (
            np.transpose(np.matmul(elements.inv_Je4, elements.DH4))
            * 2
            * properties.viscocity
            * mmt_mmt
            * elements.inv_Je4
            * elements.DH4
            * dV
        )

        # F - results array
        self.re = 1

        # ke - return matrix

        return self


class GaussPoint:
    """
    Only one gauss point located at center of the QUAD element of 4 nodes. \n
    "quad_elements" method fills the object \n
    gauss points, Je, shape functions
    """

    def __init__(self):
        self.rg = 0.0
        self.sg = 0.0
        self.w = 2.0
        self.H = []
        self.DH = []
        self.DH4 = []
        self.Je = []
        self.inv_Je = []
        self.det_Je = []
        self.inv_Je4 = []

    def quad_elements(self):
        # Shape functions evaluated in Gauss point
        self.H = np.array(
            [
                [0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0],
                [0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25],
            ]
        )
        # Derivate of shape function, evaluated in gauss point
        self.DH = np.array(
            [
                [0.25, 0.0, -0.25, 0.0, -0.25, 0.0, 0.25, 0.0],
                [0.0, 0.25, 0.0, -0.25, 0.0, -0.25, 0.0, 0.25],
                [0.25, 0.0, 0.25, 0.0, -0.25, 0.0, -0.25, 0.0],
                [0.0, 0.25, 0.0, 0.25, 0.0, -0.25, 0.0, -0.25],
            ]
        )

        self.DH4 = np.array(
            [
                [0.25, 0.0, -0.25, 0.0, -0.25, 0.0, 0.25, 0.0],
                [0.25, 0.0, 0.25, 0.0, -0.25, 0.0, -0.25, 0.0],
                [0.0, 0.25, 0.0, -0.25, 0.0, -0.25, 0.0, 0.25],
                [0.0, 0.25, 0.0, 0.25, 0.0, -0.25, 0.0, -0.25],
            ]
        )
        # Jacobian of the QUAD element
        self.Je = np.array([[0.05 / 2.0, 0.0], [0.0, 0.05 / 2.0]])
        self.inv_Je = np.linalg.inv(self.Je)
        self.det_Je = np.linalg.det(self.Je)
        self.inv_Je4 = np.array(
            [
                [self.inv_Je[0, 0], self.inv_Je[0, 1], 0.0, 0.0],
                [0.0, 0.0, self.inv_Je[1, 0], self.inv_Je[1, 1]],
                [
                    self.inv_Je[1, 0],
                    self.inv_Je[1, 1],
                    self.inv_Je[0, 0],
                    self.inv_Je[0, 1],
                ],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        return self


# --- test for calculation script ---
if __name__ == "__main__":
    from initialConditions.initialMatrices import InitialMatrix
    from initialConditions.initialParameters import ParameterMaterial, PropertyMaterial
    from meshing.meshFile import MeshData

    properties_tes = PropertyMaterial()
    parameters_tes = ParameterMaterial()
    mesh_test = MeshData().generation()
    matrices_tes = InitialMatrix(mesh_test)

    result = calculation(properties_tes, parameters_tes, mesh_test, matrices_tes)
    a = 1
