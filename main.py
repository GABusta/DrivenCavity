from initialConditions.initialParameters import ParameterMaterial, PropertyMaterial
from initialConditions.initialMatrices import InitialMatrixQuadElement
from meshing.meshFile import TestMeshData
from solutions_algorithms.iterative_solution import SolutionIterative

if __name__ == "__main__":
    properties = PropertyMaterial()
    parameters = ParameterMaterial()
    mesh = TestMeshData().generation()
    initial_matrix = InitialMatrixQuadElement(mesh)

    try:
        # solution of the equations
        solution = SolutionIterative()
        solution.iterative_solutions(
            properties=properties,
            parameters=parameters,
            mesh=mesh,
            matrices=initial_matrix,
        )
        # graphics
        solution.plot_quad_elements_solutions(mesh)

    except Exception:
        print("Problem with the solution, maybe some singular matrix")
