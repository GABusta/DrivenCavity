import numpy as np


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
        self.degrees_freedom = 8
        self.mmt = []
        self.I = []
        self.mmt_mmt = []

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

        # generation of -->  m.transpose(m)
        self.mmt = np.array(
            [
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0],
            ]
        )

        # generation --> transpose( I - m.transpose(m)/3 ) . (I - m.transpose(m)/3 )
        self.I = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # generation --> mmt_mmt (matrix)
        self.mmt_mmt = np.dot(np.transpose(self.I - self.mmt / 3), self.I - self.mmt / 3)
        return self


# --- test for calculation script ---
if __name__ == "__main__":
    from initialConditions.initialMatrices import InitialMatrixQuadElement
    from initialConditions.initialParameters import ParameterMaterial, PropertyMaterial
    from meshing.meshFile import MeshData

    properties_tes = PropertyMaterial()
    parameters_tes = ParameterMaterial()
    mesh_test = MeshData().generation()
    matrices_tes = InitialMatrixQuadElement(mesh_test)
    matrices_tes.V0[:] = 1.0

    result = global_matrix_assembly(
        properties=properties_tes,
        parameters=parameters_tes,
        mesh=mesh_test,
        matrices=matrices_tes,
    )
    a = 1
