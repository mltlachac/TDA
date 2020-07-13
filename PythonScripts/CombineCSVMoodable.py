import os
import csv
import pandas as pd
import numpy as np

# Adam Sargent 2020

betti_path = "C:/Temp/2secOver0BettiCurves.csv"
opensmile_path = "C:/Temp/FeatureSelected_Labeled_NoMissing_2secOver0OpenSmile.csv"
output_path = "C:/Temp/2secOver0BCOS25.csv"

d_bc = pd.read_csv(betti_path)
bc_headers = d_bc.columns.values

with open(output_path, mode='w', newline="\n") as o_file:
    o_writer = csv.writer(o_file, delimiter=",")
    with open(opensmile_path, mode='r', newline="\n") as opensmile_file:
        opensmile_reader = csv.reader(opensmile_file, delimiter=",")
        os_headers = next(opensmile_reader, None)
        headers = bc_headers.tolist() + os_headers[5:]
        o_writer.writerow(headers)
        for row in opensmile_reader:
            if int(row[3][-1]) == 1 and row[3][-2] == '_':
                bc_row = d_bc.loc[d_bc["id"] == row[0]+"_1"]
                input_row = np.ndarray.flatten(bc_row.to_numpy()).tolist() + row[5:]
                o_writer.writerow(input_row)
print(headers)