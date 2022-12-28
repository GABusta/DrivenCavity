from copy import deepcopy
import numpy as np


class SolutionIterative:
    def __init__(self):
        self.velocity = []

    def iterative_solutions(self, properties, parameters, mesh, matrices):
        error = 1.0
        iterations = 0
        previous_velocity = deepcopy(matrices.V0)
        matrices = matrices.apply_boundary_conditions(properties.initial_velocity, mesh)

        while (error >= parameters.tolerance) & (100 > iterations):
            matrices = matrices.quad_global_matrix_assembly(
                properties=properties,
                parameters=parameters,
                mesh=mesh)

            # solution
            matrices.V0 = np.linalg.solve((matrices.KG + matrices.NG), matrices.RG)

            # Error
            error = np.linalg.norm(matrices.V0 - previous_velocity) / np.linalg.norm(
                matrices.V0
            )
            iterations += 1

            previous_velocity = matrices.V0

        self.velocity = matrices.V0

        return self

    def plot_quad_elements_solutions(self, mesh):
        from Graphics.graphic_solutions import PlotMatrices

        image = PlotMatrices()
        image.plot_streamline_quad_element(self.velocity, mesh)
