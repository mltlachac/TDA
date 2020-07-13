import numpy as np
import os
import csv
import random
import matplotlib.pyplot as plt

# Adam Sargent 2020

components = 100
type = 3
level = 0
folder_path = "C:/Users/Adam Sargent/PycharmProjects/mhsmqp1920/exp/TDA/output/curves/sublevels/"
output_file = "C:/Users/Adam Sargent/PycharmProjects/mhsmqp1920/exp/TDA/DAIC_Curves_Sublevel_OpenSmile_Combined.csv"
opensmile = "C:/Users/Adam Sargent/PycharmProjects/mhsmqp1920/exp/TDA/OpenSmileDAICClean.csv"
phq_scores_file = "C:/Users/Adam Sargent/Documents/MQP/PHQ scores/PHQ-8withoutGender.csv"
with open(output_file, mode='w', newline='\n') as output:
    o_writer = csv.writer(output, delimiter=",")
    headers = ['id']
    for i in range(1, 101):
        headers.append("point_" + str(i))
    if type == 3:
        with open(opensmile, mode='r') as opensmile_input:
            opensmile_reader = csv.reader(opensmile_input, delimiter=",")
            for row in opensmile_reader:
                headers.extend(row[1:-1])
                break
    headers.append("phq_8")
    o_writer.writerow(headers)
    print(headers)
    for f in os.listdir(folder_path):
        p_num = f[:3]
        phq = -1
        with open(phq_scores_file, mode='r') as phq_scores:
            phq_reader = csv.reader(phq_scores, delimiter=",")
            for row in phq_reader:
                if row[0] == p_num:
                    phq = int(row[1])
        if phq != -1:
            with open(folder_path + '/' + f, mode='r') as input_:
                # random 5
                if type == 0:
                    in_reader = csv.reader(input_, delimiter=",")
                    lines = [line for line in in_reader]
                    random_choice = random.sample(lines, 5)
                    for row in random_choice:
                        id = p_num + "_" + str(int(float(row[0]))) + "_" + str(int(float(row[1])))
                        row_in = np.insert(
                            np.insert(np.array(row[2:]).astype(np.float).astype(np.int), 100, int(phq)).astype(np.str),
                            0, id)
                        o_writer.writerow(row_in)
                # first 5
                elif type == 1:
                    in_reader = csv.reader(input_, delimiter=",")
                    count = 0
                    for row in in_reader:
                        if count == 5:
                            break
                        id = p_num + "_" + str(int(float(row[0]))) + "_" + str(int(float(row[1])))
                        row_in = np.insert(
                            np.insert(np.array(row[2:]).astype(np.float).astype(np.int), 100, int(phq)).astype(np.str),
                            0, id)
                        o_writer.writerow(row_in)
                        count += 1
                # first clip from each question
                elif type == 2:
                    in_reader = csv.reader(input_, delimiter=",")
                    for row in in_reader:
                        if int(float(row[1])) == 0:
                            id = p_num + "_" + str(int(float(row[0]))) + "_" + str(int(float(row[1])))
                            row_in = np.insert(
                                np.insert(np.array(row[2:]).astype(np.float).astype(np.int), 100, int(phq)).astype(
                                    np.str), 0, id)
                            o_writer.writerow(row_in)
                # Combines with opensmile features (176)
                elif type == 3:
                    in_reader = csv.reader(input_, delimiter=",")
                    count = 0
                    for row in in_reader:
                        if count == 5:
                            break
                        id = p_num + "_" + str(int(float(row[0]))) + "_" + str(int(float(row[1])))
                        with open(opensmile, mode='r') as opensmile_input:
                            opensmile_reader = csv.reader(opensmile_input, delimiter=",")
                            found = False
                            for os_row in opensmile_reader:
                                if os_row[0] == id:
                                    row_in = [id]
                                    row_in.extend((np.array(row[2:]).astype(np.float).astype(np.int).astype(np.str).tolist()))
                                    row_in.extend(os_row[1:])
                                    o_writer.writerow(row_in)
                                    count += 1
                                    found = True
                                    break
                            if not found:
                                print("Opensmile feature not found for " + p_num)
        print("Wrote " + p_num)

    else:
        print(p_num + " does not have phq score, continuing")
