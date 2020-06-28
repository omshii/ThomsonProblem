import numpy as np

def calc_individual_forces(separations):
    if np.any(separations):
        return (separations/(np.linalg.norm(separations)) ** 3)
    else:
        return np.zeros(3)

particles = np.zeros((3))
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

individual_forces = np.apply_along_axis(calc_individual_forces, 2, separations)
total_forces = np.sum(individual_forces, axis=1)

print("individual forces:", individual_forces)
print("total forces:", total_forces)
