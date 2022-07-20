import numpy as np
import matplotlib.pyplot as plt
import os

import plu
import profile as prf
import funct

import copy
import tkinter as tk
from tkinter import filedialog

# main
if __name__ == '__main__':
    # plt.style.use('dark_background')
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (12, 9)

    h = {}
    folder = 'C:/Elaborazione_profilometro/Symetrics/Txt_files'

    root = tk.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(parent=root, title='Choose files to process')

    for fname in files:
        if os.path.isfile(fname):
            f = fname.removeprefix(folder)
            plu = plu.Plu(fname)

            plu.fitPlaneLS_bound(lambda a, b: a < b)
            plu.removePlane()

            hist, edges = plu.histMethod(bins=250)
            funct.findHfromHist(hist, edges)

            extracted = copy.copy(plu.meanProfile('x'))
            steps, p_v = extracted.stepAuto()  # P_V are the peaks and valleys
            print(f'Steps found: {steps}')
            extracted.fitLineLS()

            # plot section
            print('plotting results')
            plu.init_graphics()

            plu.pltPlot(f)
            plu.pltCplot()
            plu.planePlot()
            plu.histPlot(hist, edges)

            extracted.init_graphics()
            extracted.prfPlot(f)
            extracted.roiPlot(p_v)
            extracted.linePlot()

            plt.show()
