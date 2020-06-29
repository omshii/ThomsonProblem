import numpy as np
import sys
import time

start_time = time.time()
ftime = 0
vtime = 0
ptime = 0
stime = 0
etime = 0
np.seterr(divide='ignore', invalid='ignore')

particle_count = int(sys.argv[1])
dt = float(sys.argv[2])
duration = float(sys.argv[3])

#Helper functions
def calc_individual_forces(separations):
    return (separations/(np.linalg.norm(separations)) ** 3)

def calc_energy(separations):
    return 1/(np.linalg.norm(separations))

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

    #Calculate forces
    f = time.time()
    total_forces = np.sum((np.nan_to_num(np.apply_along_axis(calc_individual_forces, 2, separations), 0, 0, 0)), axis=1)
    unit_radii = np.apply_along_axis(calc_unit_vector, 1, positions)
    component_forces = total_forces - np.einsum('i, ij->ij', (np.einsum('ij, ij->i',total_forces, positions)), unit_radii)
    ftime = ftime + time.time() - f

    #Calculate and update velocities
    v = time.time()
    velocities = (velocities + component_forces*dt)
    velocities = 0.997*(velocities - np.einsum('i, ij->ij', (np.einsum('ij, ij->i',velocities, positions)), unit_radii))
    vtime = vtime + time.time() - v

    #Calculate and update positions
    p = time.time()
    positions = np.apply_along_axis(calc_unit_vector, 1, positions + velocities*dt)
    ptime = ptime + time.time() - p

    #Calculate and update separations
    s = time.time()
    separations = np.repeat(positions[:, np.newaxis, :], particle_count, axis=1) - np.repeat(positions[np.newaxis, :, :], particle_count, axis=0)
    stime = stime + time.time() - s

    #Calculate and update potential_energy
    e = time.time()
    potential_energy = np.sum(np.nan_to_num(1/np.linalg.norm(separations, axis=2), 0, 0, 0))/2
    etime = etime + time.time() - e

    current_time = current_time + dt

print(potential_energy)
print("Time taken: ", time.time()-start_time)
print("Force time: ", ftime)
print("Velocity time: ", vtime)
print("Postion time: ", ptime)
print("Sep time: ", stime)
print("Energy time: ", etime)
