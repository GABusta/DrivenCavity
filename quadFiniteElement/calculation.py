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
        # G = [[d(v0k)/dx  , d(v0k)/dxy ],
        #      [d(v0k)/dyx , d(v0k)/dy  ]]

        # d_v0k_x = elements.inv_Je[0][0] * 4 * (v0_k)

    return elements


class GaussPoint:
    """
    Only one gauss point located at center of the QUAD element of 4 nodes.
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
