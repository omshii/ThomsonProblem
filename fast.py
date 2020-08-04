import numpy as np
import sys
import time
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


particle_count = int(sys.argv[1])
dt = float(sys.argv[2])
duration = float(sys.argv[3])

#Initialization
velocities = np.zeros((particle_count, 3))
separations = np.zeros((particle_count, particle_count, 3))
potential_energy = []
time_array = []

#Randomized positions
positions = np.random.rand(particle_count,3)
positions = np.repeat(1/np.linalg.norm(positions, axis=1)[:, np.newaxis], 3, axis=1)*positions

#Compute separations
separations = np.repeat(positions[:, np.newaxis, :], particle_count, axis=1) - np.repeat(positions[np.newaxis, :, :], particle_count, axis=0)

current_time = 0

while current_time<duration:

    #Calculate forces
    magnitudes = np.linalg.norm(separations, axis=2)
    total_forces = np.sum(np.repeat(np.nan_to_num(1/(magnitudes * magnitudes * magnitudes), 0, 0, 0)[:, :, np.newaxis], 3, axis=2)*separations, axis=1)
    unit_radii = np.repeat(1/np.linalg.norm(positions, axis=1)[:, np.newaxis], 3, axis=1)*positions
    constrained_forces = total_forces - np.einsum('i, ij->ij', (np.einsum('ij, ij->i',total_forces, positions)), unit_radii)

    #Calculate and update potential_energy
    potential_energy = np.append(potential_energy, np.sum(np.nan_to_num(1/magnitudes, 0, 0, 0))/2)
    time_array = np.append(time_array, current_time)

    #Calculate and update velocities
    velocities = (velocities + constrained_forces*dt)
    velocities = 0.997*(velocities - np.einsum('i, ij->ij', (np.einsum('ij, ij->i',velocities, positions)), unit_radii))

    #Calculate and update positions
    positions = positions + velocities*dt
    positions = np.repeat(1/np.linalg.norm(positions, axis=1)[:, np.newaxis], 3, axis=1)*positions

    #Calculate and update separations
    separations = np.repeat(positions[:, np.newaxis, :], particle_count, axis=1) - np.repeat(positions[np.newaxis, :, :], particle_count, axis=0)

    current_time = current_time + dt

plt.plot(time_array, potential_energy, color="blue")
plt.show()

print(potential_energy[-1])
