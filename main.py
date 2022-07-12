from plu import *
import os

if __name__ == '__main__':
    # plt.style.use('dark_background')

    folder = 'C:/Elaborazione_profilometro/Symetrics/Txt_files'
    fname = os.listdir(folder)[5]
    f = os.path.join(folder, fname)

    if os.path.isfile(f):
        plu = Plu(f)

        plu.fitPlane3P()
        plu.planePlot()

        plu.removePlane()
        plu.pltPlot(fname)
        plt.show()
