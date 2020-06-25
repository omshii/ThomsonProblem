from main import MainCycle
import numpy as np
import csv

all_data = open("all_data", "a")
average_energies = open("average_energies", "a")

average_pe = np.array([])

for i in range(2, 11):
    average_pe = np.append(average_pe, 0)
    for j in range(0, 20):
        cycle = MainCycle(i, 0.01)
        cycle.start_cycle(50)
        particle_list = cycle.get_particle_list()
        all_data.write(str(cycle.get_potential_energy()))
        for particle in particle_list:
            all_data.write(", "+np.array_str(particle.pos[0]))
        all_data.write('\n')
        average_pe[-1] = average_pe[-1] + cycle.get_potential_energy()
    average_pe[-1] = average_pe[-1]/20
    average_energies.write(str(i)+', '+str(average_pe[-1])+'\n')
    all_data.write('\n')
    all_data.write('\n')
    print("particles: ", i, " completed")
