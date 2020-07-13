import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# Adam Sargent 2020

betti_path = "C:/Temp/2secOver0BettiCurves.csv"

def graph(curve, p_num, c_num):
    ymax = 2000
    plt.figure()
    plt.plot(curve)
    plt.axis([0, 100, 0, ymax])
    plt.title("Moodable Upper Level Betti Curve")
    plt.savefig("output/emu/BC_" + p_num + "_" + c_num + ".png")
    plt.show()
    return

with open(betti_path, mode='r') as b_file:
    b_reader = csv.reader(b_file, delimiter=",")
    headers = next(b_reader, None)
    for row in b_reader:
        id = row[0][:-2]
        clip = '1'
        curve = np.array(row[1:]).astype(np.float)
        # if id == "emu3456":
        #     graph(curve, id, clip)
        if id == "moodable6475":
            graph(curve, id, clip)


