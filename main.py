from initialConditions.initialParameters import ParameterMaterial, PropertyMaterial
from initialConditions.initialMatrices import InitialMatrixQuadElement
from meshing.meshFile import MeshData
from solutions_algorithms.iterative_solution import SolutionIterative

if __name__ == "__main__":
    properties = PropertyMaterial()
    parameters = ParameterMaterial()
    mesh = MeshData().generation()
    initial_matrix = InitialMatrixQuadElement(mesh)

    # # Apply boundary conditions
    # initial_matrix.boundary_conditions(properties.initial_velocity, mesh)

    try:
        # solution of the equations
        solution = SolutionIterative()
        solution.iterative_solutions(
            properties=properties,
            parameters=parameters,
            mesh=mesh,
            matrices=initial_matrix,
        )
        # if solution.velocity:
        solution.print_quad_elements(mesh)

    except Exception:
        print("Problem with the solution, maybe some singular matrix")

    a = 1
