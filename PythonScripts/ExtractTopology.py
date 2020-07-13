import gudhi as gd
import numpy as np
import librosa as lr
import librosa.display as lr_disp
import matplotlib.pyplot as plt
import os
import csv
#import exp.TDA.TDAFunctions as TDA
# moved tda functions into this file to avoid problems with importing local files when running jobs

# Adam Sargent 2020

def get_persistence_from_audio(audio_wave, sample=22050, length=5, graph=False):
    simplex_up = gd.SimplexTree()  # for the upper level persistence data
    simplex_dw = gd.SimplexTree()  # for the lower level persistence data
    for i in np.arange(len(audio_wave)):
        simplex_up.insert([i, i + 1], filtration=audio_wave[i])
        simplex_dw.insert([i, i + 1], filtration=-audio_wave[i])
    for i in np.arange(len(audio_wave) - 1):
        simplex_up.insert([i, i + 1], filtration=audio_wave[i])
        simplex_dw.insert([i, i + 1], filtration=-audio_wave[i])
    simplex_up.initialize_filtration()
    simplex_dw.initialize_filtration()
    dig_up = simplex_up.persistence()
    dig_dw = simplex_dw.persistence()
    if graph:
        plt.figure()
        lr_disp.waveplot(audio_wave, sr=sample)
        plt.title("Audio Wave")
        plt.show()
        plt.figure()
        gd.plot_persistence_barcode(dig_up)
        plt.title("Upper Level Persistence Barcode")
        plt.show()
        plt.figure()
        gd.plot_persistence_barcode(dig_dw)
        plt.title("Sub Levels Persistence Barcode")
        plt.show()
        plt.figure()
        gd.plot_persistence_diagram(dig_up)
        plt.title("Upper Level Persistence Diagram")
        plt.show()
        plt.figure()
        gd.plot_persistence_diagram(dig_dw)
        plt.title("Sub Levels Persistence Diagram")
        plt.show()
    return dig_up, dig_dw  # NOTE: this does not filter out infinite values


# Taken from TDAToolbox.filtration
def functionize(val, descriptor):
    def dirichlet(x):
        return 1 if (x > descriptor[0]) and (x < descriptor[1]) else 0

    return np.vectorize(dirichlet)(val)


def get_betti_curve_from_persistence(dig, num_points=100, graph=False):
    dig = np.asarray([[ele[1][0], ele[1][1]] for ele in dig if ele[1][1] < np.inf])
    v = np.zeros(num_points)
    try:
        mn, mx = np.min(dig), np.max(dig)
        val_up = np.linspace(mn, mx, num=num_points)
        for ele in dig:
            v += functionize(val_up, ele)
        if graph:
            plt.figure()
            plt.plot(v)
            plt.title("Betti Curve")
            plt.show()
        return v
    except ValueError:
        print("Silent, returning 0")
        return v



components = 100
folder_path = "C:/Users/Adam Sargent/Documents/MQP/QUESTIONS/SLICES"
output_path = "C:/Users/Adam Sargent/PycharmProjects/mhsmqp1920/exp/TDA/output/curves/"
# NOTE: participant 385 only has slices less that 5 seconds
n = 0
average = 0
for participant in os.listdir(folder_path):
    p_num = participant[:-7]
    with open(output_path + p_num+"_curves.csv", mode='w', newline="\n") as p_file:
        p_writer = csv.writer(p_file, delimiter=",")
        for file in os.listdir(folder_path+"/"+participant):
            audio_wave, s_rate = lr.core.load(folder_path+"/"+participant+"/"+file)

            if lr.core.get_duration(audio_wave) >= 5:
                dig_up, dig_dw = get_persistence_from_audio(audio_wave, sample=s_rate)
                betti = get_betti_curve_from_persistence(dig_up, num_points=components)
                betti = np.insert(betti, 0, file[-6:-4].replace('_', ''))
                betti = np.insert(betti, 0, file[13:-6].replace('_', ''))
                p_writer.writerow(betti)
    print("Wrote " + p_num)



