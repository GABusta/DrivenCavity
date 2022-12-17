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


# --- test for initial matrix script ---
if __name__ == "__main__":
    from meshing.meshFile import MeshData

    mesh_test = MeshData()
    matrix_test = InitialMatrix(mesh_test)
    a = 1