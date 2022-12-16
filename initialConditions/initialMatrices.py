import numpy as np


class InitialMatrix:
    """
    Generation of the initial matrix, "x" and "y" coordinates
    """
    def __init__(self, mesh):
        self.KG = np.zeros((mesh.nodes * 2, mesh.nodes * 2))
        self.NG = np.zeros((mesh.nodes * 2, mesh.nodes * 2))
        self.RG = np.zeros((mesh.nodes * 2, 1))
        self.V0 = np.zeros((mesh.nodes * 2, 1))
