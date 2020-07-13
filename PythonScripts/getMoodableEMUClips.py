import gudhi as gd
import numpy as np
import librosa as lr
import librosa.display as lr_disp
import matplotlib.pyplot as plt
import os
import csv
import exp.TDA.TDAFunctions as TDA

# Adam Sargent 2020

components = 100
folder_path = "C:/Temp/2secOver0/"
output_path = "C:/Temp/2secOver0BettiCurvesSub.csv"
n = 0
average = 0
totals = [0,0,0,0,0,0,0,0,0,0,0,0]
lengths = []
headers = ['id']
for i in range(1, 101):
    headers.append("point_" + str(i))
with open(output_path, mode='w', newline="\n") as o_file:
    o_writer = csv.writer(o_file, delimiter=",")
    o_writer.writerow(headers)
    for file in os.listdir(folder_path):
        id = file[:-10]+file[-6:-4]
        audio_wave, s_rate = lr.core.load(folder_path + "/" + file)
        clip_length = lr.core.get_duration(audio_wave)
        print(clip_length)
        if lengths == []:
            lengths = [clip_length]
        else:
            lengths.append(clip_length)
        clip_length_int = int(clip_length)
        if clip_length_int <= 10:
            totals[clip_length_int] += 1
        else:
            totals[11] += 1

        if clip_length == 2.0:
            dig_up, dig_dw = TDA.get_persistence_from_audio(audio_wave, sample=s_rate)
            betti = TDA.get_betti_curve_from_persistence(dig_dw, num_points=components).tolist()
            betti.insert(0,id)
            o_writer.writerow(betti)
            print("Wrote " + id)
        else:
            print("Skipped " + id)
print(totals)
plt.plot(lengths)
plt.ylim([0,10])
plt.show()