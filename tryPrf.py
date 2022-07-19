import numpy as np
import matplotlib.pyplot as plt
import os

import profile as prf

import tkinter as tk
from tkinter import filedialog


# main
if __name__ == '__main__':
    # plt.style.use('dark_background')
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (12, 9)

    h = {}
    folder = 'C:/Elaborazione_profilometro/Symetrics/Txt_files/prf'

    root = tk.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(parent=root, title='Choose files to process')

    for fname in files:
        if os.path.isfile(fname):
            f = fname.removeprefix(folder)
            prof = prf.Profile()
            prof.openPrf(fname)
            prof.fitLineLS()
            prof.removeLine()
            print(prof.Ra(0.8, 5))

            prof.init_graphics()
            prof.prfPlot(f)
            prof.linePlot()

            plt.show()
