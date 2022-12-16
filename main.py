from initialConditions.initialParameters import ParameterMaterial, PropertyMaterial
from initialConditions.initialMatrices import InitialMatrix
from meshing.meshFile import MeshData


def iterative_solutions(properties, parameters, mesh, initial_matrix):
    error = 1.0
    iterations = 0
    matrices = initial_matrix

    while (error >= parameters.tolerance) & (parameters.max_iterations > iterations):
        # matrices = calculation(properties, parameters, mesh, matrices)
        # matrices = boundary_conditions(matrices)

        # error = 1.0
        iterations += 1

    return iterations


if __name__ == "__main__":
    properties = PropertyMaterial()
    parameters = ParameterMaterial()
    mesh = MeshData().generation()
    initial_matrix = InitialMatrix(mesh)

    solution = iterative_solutions(properties, parameters, mesh, initial_matrix)
    a = 1
