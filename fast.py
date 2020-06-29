import numpy as np
import sys
import time

start_time = time.time()

particle_count = int(sys.argv[1])
dt = float(sys.argv[2])
duration = float(sys.argv[3])

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
velocities = np.zeros((particle_count, 3))
separations = np.zeros((particle_count, particle_count, 3))
potential_energy = np.zeros((1))

#Randomized positions
positions = np.apply_along_axis(calc_unit_vector, 1, (np.random.rand(particle_count,3)))

#Compute separations
separations = np.repeat(positions[:, np.newaxis, :], particle_count, axis=1) - np.repeat(positions[np.newaxis, :, :], particle_count, axis=0)

current_time = 0

while current_time<duration:
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
    positions = np.apply_along_axis(calc_unit_vector, 1, positions + velocities*dt)

    #Calculate and update separations
    separations = np.repeat(positions[:, np.newaxis, :], particle_count, axis=1) - np.repeat(positions[np.newaxis, :, :], particle_count, axis=0)

    #Calculate and update potential_energy
    potential_energy = np.sum(np.apply_along_axis(calc_energy, 2, separations))/2

    current_time = current_time + dt

print(potential_energy)
print("Time taken: ", time.time()-start_time)
