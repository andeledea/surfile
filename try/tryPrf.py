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

    # fname = filedialog.askopenfilename(parent=root, title='Choose files to process')

    x = np.linspace(0, 100, 4500)
    y = 7 * x + 10 * np.sin(x) + 4
    y[500: 1000] = y[500: 1000] + 100 * np.sin(x[500: 1000] * 7) + 200
    y[3000: 3500] = y[3000: 3500] + 50 * np.sin(x[3000: 3500] * 7) - 100

    prof = prf.Profile()
    prof.setValues(x, y)
    prof.fitLineLS()

    prof.allignWithHist(0.001)

    hist, edges = prof.histMethod()

    prof.init_graphics()
    prof.prfPlot('Plot')
    prof.linePlot()
    prof.histPlot(hist, edges)

    plt.show()

