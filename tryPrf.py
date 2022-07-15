from plu import *
import os
import copy
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
            prf = Profile()
            prf.openPrf(fname)
            prf.fitLineLS()
            prf.removeLine()
            print(prf.Ra(0.8, 5))

            prf.init_graphics()
            prf.prfPlot(f)
            prf.linePlot()

            plt.show()
