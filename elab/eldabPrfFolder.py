import numpy as np
import matplotlib.pyplot as plt
import os

import profile as prf

import tkinter as tk
from tkinter import filedialog
from tabulate import tabulate


# main
if __name__ == '__main__':
    # plt.style.use('dark_background')
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (12, 9)

    h = {}
    folder = filedialog.askdirectory(title='Choose directory to process')

    root = tk.Tk()
    root.withdraw()

    results = []

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            fname = os.path.join(subdir, file)
            if os.path.isfile(fname) and ('PRF' in fname):
                f = fname.removeprefix(folder)
                prof = prf.Profile()
                prof.openPrf(fname)
                prof.fitLineLS()
                hist, edges = prof.histMethod()
                prof.init_graphics()
                prof.prfPlot(fname)
                prof.linePlot()
                prof.histPlot(hist, edges)

                single = list(prof.roughnessParams(2.5, 5, True))
                single.insert(0, f)
                results.append(single)

                plt.show()

    print(tabulate(results, headers=['name', 'RA', 'RQ', 'RP', 'RV', 'RZ', 'RSK', 'RKU']))
