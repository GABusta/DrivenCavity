class PropertyMaterial:
    def __init__(self):
        self.rho = 1.0
        self.reynolds = 100
        self.initial_velocity = 1.0
        self.viscocity = self.rho * self.initial_velocity / self.reynolds
        self.max_length = 1.0
        self.ian = 1e7


class ParameterMaterial:
    def __init__(self):
        self.tolerance = 1e-6
        self.error = 1.0
