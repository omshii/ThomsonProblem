import numpy as np

#Helper functions
def calc_individual_forces(separations):
    if np.any(separations):
        return (separations/(np.linalg.norm(separations)) ** 3)
    else:
        return np.zeros(3)

def calc_energy(separations):
    if np.any(separations):
        return 1/(np.linalg.norm(separations))
    else:
        return np.zeros(1)

def calc_unit_vector(position):
    return position/np.linalg.norm(position)

#Initialization
velocities = np.zeros((3, 3))
separations = np.zeros((3, 3, 3))
potential_energy = np.zeros((1))
print("separations:", '\n', separations)
print("velocities:", '\n',velocities)
print("potential_energy:", '\n', potential_energy)

#Randomized positions
positions = np.random.rand(3,3)
positions = np.array([[0.50855717, 0.23848786, 0.83814653], [0.1436721, 0.87508403, 0.04696972], [0.60268039, 0.89345049, 0.02003009]])

print("randomized positions:", '\n', positions)
positions = np.apply_along_axis(calc_unit_vector, 1, positions)

#Compute separations
separations = np.einsum('ij, jik->ikj', positions, np.ones_like(separations)) - np.einsum('ij, kij->kij', positions, np.ones_like(separations))
print("Initialized separations:", '\n', separations)

dt = 0.01
t = 0

while t<50:
    #Calculate forces (without constraint)
    individual_forces = np.apply_along_axis(calc_individual_forces, 2, separations)
    total_forces = np.sum(individual_forces, axis=1)

    #Calculate constraint for forces
    unit_radii = np.apply_along_axis(calc_unit_vector, 1, positions)
    dot_product = np.einsum('ij, ij->i',total_forces, positions)
    non_component_forces = np.einsum('i, ij->ij', dot_product, unit_radii)
    component_forces = total_forces - non_component_forces

    #Calculate and update velocities
    velocities = velocities + component_forces*dt
    dot_product_velocities = np.einsum('ij, ij->i',velocities, positions)
    non_component_velocties = np.einsum('i, ij->ij', dot_product_velocities, unit_radii)
    velocities = 0.997*(velocities - non_component_velocties)

    #Calculate and update positions
    positions = positions + velocities*dt
    positions = np.apply_along_axis(calc_unit_vector, 1, positions)

    #Calculate and update separations
    separations = np.einsum('ij, jik->ikj', positions, np.ones_like(separations)) - np.einsum('ij, kij->kij', positions, np.ones_like(separations))

    #Calculate and update potential_energy
    temp_energy = np.apply_along_axis(calc_energy, 2, separations)
    sum = np.sum(temp_energy)
    potential_energy = sum/2

    print(potential_energy)

    t = t+dt
