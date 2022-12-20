import numpy as np
import matplotlib.pyplot as plt
import os

import surface as sur

import tkinter as tk
from tkinter import filedialog
from tabulate import tabulate

# main
if __name__ == '__main__':
    # plt.style.use('dark_background')
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (12, 9)

    h = {}
    folder = 'C:/Elaborazione_profilometro/Symetrics/Txt_files'

    root = tk.Tk()
    root.withdraw()

    fname = filedialog.askopenfilename(parent=root, title='Choose files to process')

    surf = sur.Surface()
    surf.openTxt(fname)
    surf.cutSurfaceRectangle()

    surf.init_graphics()
    surf.pltPlot(fname)

    plt.show()
