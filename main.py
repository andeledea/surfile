from plu import *
import os
import tkinter as tk
from tkinter import filedialog

# main
if __name__ == '__main__':
    # plt.style.use('dark_background')
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (10, 9)

    h = {}
    folder = 'C:/Elaborazione_profilometro/Symetrics/Txt_files'

    root = tk.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(parent=root)

    for fname in files:
        if os.path.isfile(fname):
            f = fname.removeprefix(folder)
            plu = Plu(fname)

            plu.fitPlaneLS_bound(lambda a, b: a < b)
            # plu.planePlot()
            plu.removePlane()

            # hist, edges = plu.histMethod(bins=200)
            # plu.histPlot(hist, edges)
            # h[f] = plu.findHfromHist(hist, edges)

            plu.pltPlot(f)
            plu.pltCplot(f)
            # print(h)

            plt.show()
