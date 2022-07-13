import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from funct import *
import matplotlib.gridspec as gridspec


class Profile:
    def __init__(self):
        self.gs = None
        self.fig = None
        self.X = None
        self.Y = None

    def setValues(self, X, Y):
        self.X = X
        self.Y = Y

    #####################################################################################################
    #                                       PLOT SECTION                                                #
    #####################################################################################################
    def init_graphics(self):
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(3, 3)

    def prfPlot(self):
        ax_prf = self.fig.add_subplot(self.gs[0:-1, :])
        ax_prf.plot(self.X, self.Y, color='red')  # hot, viridis, rainbow
        persFig(
            ax_prf,
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]'
        )
        ax_prf.set_title('Profile')



