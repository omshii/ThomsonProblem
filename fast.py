import numpy as np

def calc_force(sep):
    if sep!= 0:
        force = sep / (np.linalg.norm(sep)) ** 3
    else:
        force = np.zeros((3))
    return force

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
        separations[i][j] = positions[i][0:] - positions[j][0:]
        separations[j][i] = positions[j][0:] - positions[i][0:]

print("Initialized separations:", separations)

t = 0
tt = 5

#Start sim
while t < tt:
    force = np.where(separations!=0, (separations/(np.linalg.norm(separations)) ** 3), np.zeros(3))
    t = t + 1

print("final forces:", force)
