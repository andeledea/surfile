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

    root = tk.Tk()
    root.withdraw()

    results = []

    fname = filedialog.askopenfilename(parent=root, title='Choose files to process')

    prof = prf.Profile()
    prof.openPrf(fname)

    single = list(prof.roughnessParams(0.8, 5, plot=True))
    single.insert(0, 'ISO old')
    results.append(single)

    prof.morphFilter(0.0025)  # radius in mm

    single = list(prof.roughnessParams(0.8, 5, plot=True))
    single.insert(0, 'ISO new')
    results.append(single)

    prof.init_graphics()
    prof.prfPlot('Plot')

    plt.show()

    print(tabulate(results, headers=['name', 'RA', 'RQ', 'RP', 'RV', 'RZ', 'RSK', 'RKU']))

