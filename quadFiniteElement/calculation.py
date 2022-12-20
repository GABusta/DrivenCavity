from copy import deepcopy
import numpy as np
from initialConditions.initialMatrices import InitialMatrix


def global_matrix_calculation(properties, parameters, mesh, matrices):
    elements = GaussPoint().quad_elements()
    global_matrix = deepcopy(matrices)
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
                global_matrix.V0[conn[0] * 2 - 2],
                global_matrix.V0[conn[1] * 2 - 2],
                global_matrix.V0[conn[2] * 2 - 2],
                global_matrix.V0[conn[3] * 2 - 2],
            ]
        )

        v0_y = np.array(
            [
                global_matrix.V0[conn[0] * 2 - 1],
                global_matrix.V0[conn[1] * 2 - 1],
                global_matrix.V0[conn[2] * 2 - 1],
                global_matrix.V0[conn[3] * 2 - 1],
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
        elemental_matrices = QuadElement().matrix_generation(
            elements=elements,
            parameters=parameters,
            properties=properties,
            matrix_g=matrix_g,
            v0_k=v0_k,
        )

        # Global Matrices
        global_matrix.assembly_quad_elements(
            elemental_matrix=elemental_matrices, connections=mesh.connections[i]
        )

    return global_matrix


class QuadElement:
    def __init__(self):
        self.ne = []
        self.re = []
        self.ke = []

    def matrix_generation(self, elements, parameters, properties, matrix_g, v0_k):
        """
        Generation of the Elemental matrices, for a QUAD element \n
        :param elements: information about Jacobian, shape functions
        :param parameters: of run
        :param properties: of the material
        :param matrix_g:
        :param v0_k:
        :return: object with elemental matrices
        """
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

        # ne - matrix (convective term "N")
        dV = elements.det_Je * elements.w
        self.ne = (
            np.matmul(np.transpose(elements.H), (C + np.matmul(matrix_g, elements.H)))
            * properties.rho
            * dV
        )

        # K - matrix (diffusive term)
        k_lan = (
            np.matmul(
                np.transpose(np.matmul(elements.inv_Je4, elements.DH4)),
                np.matmul(mmt, np.matmul(elements.inv_Je4, elements.DH4)),
            )
            * properties.ian
            * dV
        )

        k_vv = (
            np.matmul(
                np.transpose(np.matmul(elements.inv_Je4, elements.DH4)),
                np.matmul(mmt_mmt, np.matmul(elements.inv_Je4, elements.DH4)),
            )
            * dV
            * 2.0
            * properties.viscocity
        )

        # re - results array (F)
        self.re = (
            np.matmul(np.transpose(elements.H), np.matmul(matrix_g, np.transpose(v0_k)))
            * properties.rho
            * dV
        )

        # ke - return matrix
        self.ke = k_vv - k_lan

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
        """
        Shape functions and Jacobians for a QUAD element, with one Gauss point \n
        :return: object with information
        """

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
    matrices_tes.V0[:] = 1.0

    result = global_matrix_calculation(
        properties=properties_tes,
        parameters=parameters_tes,
        mesh=mesh_test,
        matrices=matrices_tes,
    )
    a = 1
