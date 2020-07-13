import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# Adam Sargent 2020

def graph(curve, p_num, q_num='', s_num=''):
    ymax = 4000
    plt.figure()
    plt.plot(curve)
    plt.axis([0, 100, 0, ymax])
    if q_num =='':
        plt.title("Betti Curve of Averages of " + p_num)
        plt.savefig("output/BC_" + p_num + "_average" + ".png")
    else:
        plt.title("DAIC-WOZ Sub Level Betti Curve")
        plt.savefig("output/BC_Sub_" + p_num + "_" + q_num + "_" + s_num + ".png")
    plt.show()
    return


folder_path = "C:/Users/Adam Sargent/PycharmProjects/mhsmqp1920/exp/TDA/output/curves/sublevels"
averages_file = "C:/Users/Adam Sargent/PycharmProjects/mhsmqp1920/exp/TDA/output/averages.csv"
# 364 - 0, 321 - 20, 400 - 7, 362 - 20
p_saved = ['364']#, '321', '400', '362']

# get slice curves for p_saved
for f in os.listdir(folder_path):
    p_num = f[:3]
    if p_num in p_saved:
        print(p_num)
        with open(folder_path + '/' + f, mode='r') as curves:
            c_reader = csv.reader(curves, delimiter=",")
            for row in c_reader:
                q_num = int(float(row[0]))
                s_num = int(float(row[1]))
                if p_num == p_saved[0]:
                    if (q_num == 22 and s_num == 0) or (q_num == 28 and s_num == 0):
                        curve = np.array(row[2:]).astype(np.float)
                        graph(curve, p_num, q_num=str(q_num), s_num=str(s_num))
                elif p_num == p_saved[1]:
                    if (q_num == 22 and s_num == 0) or (q_num == 41 and s_num == 1):
                        curve = np.array(row[2:]).astype(np.float)
                        graph(curve, p_num, q_num=str(q_num), s_num=str(s_num))
                elif p_num == p_saved[2]:
                    if (q_num == 24 and s_num == 0) or (q_num == 34 and s_num == 0):
                        curve = np.array(row[2:]).astype(np.float)
                        graph(curve, p_num, q_num=str(q_num), s_num=str(s_num))
                elif p_num == p_saved[3]:
                    if (q_num == 22 and s_num == 0) or (q_num == 32 and s_num == 0):
                        curve = np.array(row[2:]).astype(np.float)
                        graph(curve, p_num, q_num=str(q_num), s_num=str(s_num))
# get average curves
with open(averages_file, mode='r') as averages:
    a_reader = csv.reader(averages, delimiter=",")
    for row in a_reader:
        p_num = str(int(float(row[0])))
        if p_num in p_saved:
            curve = np.array(row[1:]).astype(np.float)
            graph(curve, p_num)


