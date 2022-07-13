import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from funct import *
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec


class Profile:
    def __init__(self):
        self.gr = None
        self.roi = None
        self.gs = None
        self.fig = None
        self.X = None
        self.Y = None

    def setValues(self, X, Y):
        """
        Sets the values for the profile
        :param X: x values of profile
        :param Y: y values of profile
        """
        self.X = X
        self.Y = Y

    def stepAuto(self):
        """
        Calculates the step height using the auto method
        """
        self.gr = np.gradient(self.Y)
        peaks, _ = find_peaks(self.gr, height=500)
        valle, _ = find_peaks(-self.gr, height=500)

        self.roi = []  # regions of interest points
        p_v = np.sort(np.concatenate((peaks, valle)))  # every point of interest (INDEXES of x array)
        print(p_v)
        for i in range(len(p_v) - 1):
            print(f'Eval {p_v[i]}, {p_v[i + 1]}')
            locRange = round((p_v[i + 1] - p_v[i]) / 3)  # profile portion is 1/3 of region
            roi_start = p_v[i] + locRange
            roi_end = p_v[i + 1] - locRange
            print(f'Roi: from {roi_start} to {roi_end}')
            self.roi.append((self.X[roi_start: roi_end],
                             self.Y[roi_start: roi_end]))

    #####################################################################################################
    #                                       PLOT SECTION                                                #
    #####################################################################################################
    def init_graphics(self):  # creates a figure subdivided in grid for subplots
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(3, 3)

    def prfPlot(self):  # plots the profile
        ax_prf = self.fig.add_subplot(self.gs[0, :])
        ax_prf.plot(self.X, self.Y, color='teal')  # hot, viridis, rainbow
        persFig(
            ax_prf,
            gridcol='grey',
            xlab='x [um]',
            ylab='z [um]'
        )
        ax_prf.set_title('Profile')

    def roiPlot(self):
        ax_roi = self.fig.add_subplot(self.gs[1, :])
        ax_roi.plot(self.X, self.Y, color='teal')  # hot, viridis, rainbow

        for roi in self.roi:
            ax_roi.plot(roi[0], roi[1], color='red')  # hot, viridis, rainbow
            ax_roi.plot(self.X, self.gr, color='blue')

        persFig(
            ax_roi,
            gridcol='grey',
            xlab='x [um]',
            ylab='z [um]'
        )
