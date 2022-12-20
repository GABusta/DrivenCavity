from copy import deepcopy
import numpy as np
from initialConditions.initialParameters import ParameterMaterial, PropertyMaterial
from initialConditions.initialMatrices import InitialMatrix
from meshing.meshFile import MeshData
from quadFiniteElement.calculation import global_matrix_calculation


def iterative_solutions(properties, parameters, mesh, initial_matrix):
    error = 1.0
    iterations = 0
    matrices = deepcopy(initial_matrix)
    # matrices = boundary_conditions(matrices)

    while (error >= parameters.tolerance) & (parameters.max_iterations > iterations):
        new_matrices = global_matrix_calculation(
            properties=properties, parameters=parameters, mesh=mesh, matrices=matrices,
        )

        # solution
        new_matrices.V0 = np.matmul(
            np.linalg.inv(new_matrices.KG + new_matrices.NG), new_matrices.RG
        )

        # Boundary conditions
        new_matrices.boundary_conditions(properties.initial_velocity, mesh)

        # Error

        # error = 1.0
        iterations += 1

        # Reassign new values and override the old ones


    return iterations


if __name__ == "__main__":
    properties = PropertyMaterial()
    parameters = ParameterMaterial()
    mesh = MeshData().generation()
    initial_matrix = InitialMatrix(mesh)

    solution = iterative_solutions(properties, parameters, mesh, initial_matrix)
    a = 1
