from copy import deepcopy
import numpy as np


class SolutionIterative:
    def __init__(self):
        self.velocity = []

    def iterative_solutions(self, properties, parameters, mesh, matrices):
        error = 1.0
        iterations = 0
        previous_velocity = matrices.V0
        # matrices = deepcopy(initial_matrix)
        matrices.apply_boundary_conditions(properties.initial_velocity, mesh)

        while (error >= parameters.tolerance) & (100 > iterations):

            matrices.global_matrix_assembly(
                properties=properties,
                parameters=parameters,
                mesh=mesh)

            # Apply boundary conditions
            matrices.apply_boundary_conditions(properties.initial_velocity, mesh)

            # solution
            matrices.V0 = np.dot(
                np.linalg.inv(matrices.KG + matrices.NG), matrices.RG
            )

            # Error
            error = np.linalg.norm(matrices.V0 - previous_velocity) / np.linalg.norm(
                matrices.V0
            )
            iterations += 1

            previous_velocity = matrices.V0

            # # Reassign new values and override the old ones
            # initial_matrix = deepcopy(matrices)

        self.velocity = matrices.V0
        return self
        # NO ESTA ACTUALIZANDO LAS VELOCIDADES!!!!!!
    def print_quad_elements(self, mesh):
        from Graphics.graphic_solutions import PlotMatrices

        image = PlotMatrices()
        image.plot_streamline_quad_element(self.velocity, mesh)
