from plu import *
import os
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

    files = filedialog.askopenfilenames(parent=root)

    for fname in files:
        if os.path.isfile(fname):
            f = fname.removeprefix(folder)
            plu = Plu(fname)

            plu.fitPlaneLS_bound(lambda a, b: a < b)
            # plu.fitPlane3P()
            plu.removePlane()

            hist, edges = plu.histMethod(bins=200)
            # h[f] = plu.findHfromHist(hist, edges)
            extracted = copy.copy(plu.extractProfile())

            print('plotting results')
            plu.init_graphics()

            plu.pltPlot(f)
            plu.pltCplot(f)
            plu.planePlot()
            plu.histPlot(hist, edges)

            extracted.init_graphics()
            extracted.prfPlot()

            plt.show()
