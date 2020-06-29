import numpy as np

def calc_individual_forces(separations):
    if np.any(separations):
        return (separations/(np.linalg.norm(separations)) ** 3)
    else:
        return np.zeros(3)

def calc_unit_vector(position):
    return position/np.linalg.norm(position)

particles = np.zeros((3))
velocities = np.zeros((3))
separations = np.zeros((3, 3, 3))

print("particles: ", particles)
print("separations:", separations)

#Initialize positions
positions = np.random.rand(3,3)
print("positions:", positions)

#Initialize separations
for i in range(3):
    for j in range(i+1, 3):
        separations[i][j] = positions[i] - positions[j]
        separations[j][i] = positions[j] - positions[i]

print("Initialized separations:", separations)

dt = 0.01
individual_forces = np.apply_along_axis(calc_individual_forces, 2, separations)
total_forces = np.sum(individual_forces, axis=1)
unit_radii = np.apply_along_axis(calc_unit_vector, 1, positions)

dot_product = np.einsum('ij, ij->i',total_forces, positions)
non_component_forces = np.einsum('i, ij->ij', dot_product, unit_radii)
component_forces = total_forces - non_component_forces

velocities = velocities + component_forces*dt
dot_product_velocities = np.einsum('ij, ij->i',velocities, positions)
non_component_velocties = np.einsum('i, ij->ij', dot_product_velocities, unit_radii)
velocities = velocities - non_component_velocties

positions = positions + velocities*dt
positions = np.apply_along_axis(calc_unit_vector, 1, positions)


print("separations:", separations)
