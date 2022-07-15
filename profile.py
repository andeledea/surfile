import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib import cm
from funct import *
from scipy import signal, ndimage, interpolate
import matplotlib.gridspec as gridspec


@dataclass
class Roi:
    X: list
    Z: list


class Profile:

    def __init__(self):
        self.c, self.q, self.m = 0, 0, 0  # line fit parameters

        self.gr = None  # first derivative
        self.roi = None  # ISO regions
        self.gs = None
        self.fig = None
        self.X = None
        self.Z = None
        self.Z0 = None

    def openPrf(self, fname):
        z = []
        xs = 0
        zs = 0
        with open(fname, 'r') as fin:
            for line in fin.readlines():
                word = line.split()[0]
                if word == 'SPACING':
                    xs = float(line.split()[2])
                    print(f'Spacing x: {xs}')
                if word == 'CZ':
                    zs = float(line.split()[4]) * 10**3
                    print(f'Scaling z: {zs}')

                try:
                    z.append(float(word) * zs)
                except ValueError:
                    a = None

            self.Z0 = self.Z = np.array(z)
            self.X = np.linspace(0, (len(z)) * xs, len(z))

    def setValues(self, X, Y):
        """
        Sets the values for the profile
        :param X: x values of profile
        :param Y: y values of profile
        """
        self.X = X
        self.Z0 = self.Z = Y

    def stepAuto(self):  # TODO: can you find a way to automate minD calculations?
        """
        Calculates the step height using the auto method
        :return: steps: array containing all found step heights on profile
        """

        def calcSteps() -> list:
            steps = []
            definedPeaks = True
            for j in range(len(self.roi) - 2):  # consider j, j+1, j+2
                outerMeanL = np.mean(self.roi[j].Z)
                outerMeanR = np.mean(self.roi[j + 2].Z)
                innerMean = np.mean(self.roi[j + 1].Z)

                outerStdL = np.std(self.roi[j].Z)
                outerStdR = np.std(self.roi[j + 2].Z)
                innerStd = np.std(self.roi[j + 1].Z)

                step = innerMean - (outerMeanL + outerMeanR) / 2
                steps.append(step)

                if outerStdL > abs(step) / 200 or outerStdR > abs(step) / 200 or innerStd > abs(step) / 200:
                    definedPeaks = False

            if not definedPeaks:
                print(Bcol.WARNING + 'STEP HEIGHT MIGHT BE INCORRECT (PEAKS ARE POURLY DEFINED)' + Bcol.ENDC)
            return steps

        self.gr = np.gradient(self.Z)

        thresh = np.max(self.gr[30:-30]) / 1.5  # derivative threshold to detect peak, avoid border samples
        zero_cross = np.where(np.diff(np.sign(self.Z)))[0]
        spacing = (zero_cross[1] - zero_cross[0]) / 1.5

        peaks, _ = signal.find_peaks(self.gr, height=thresh, distance=spacing)
        valle, _ = signal.find_peaks(-self.gr, height=thresh, distance=spacing)

        self.roi = []  # regions of interest points
        p_v = np.sort(np.concatenate((peaks, valle)))  # every point of interest (INDEXES of x array)

        for i in range(len(p_v) - 1):
            locRange = round((p_v[i + 1] - p_v[i]) / 3)  # profile portion is 1/3 of region
            roi_start = p_v[i] + locRange
            roi_end = p_v[i + 1] - locRange
            self.roi.append(Roi(self.X[roi_start: roi_end],  # append to roi X and Y values of roi
                                self.Z[roi_start: roi_end]))

        return calcSteps(), p_v

    def fitLineLS(self):
        """
        Least square line fit implementation
        """
        # create matrix and Z vector to use lstsq
        XZ = np.vstack([self.X.reshape(np.size(self.X)),
                        self.Z.reshape(np.size(self.Z))]).T
        (rows, cols) = XZ.shape
        G = np.ones((rows, 2))
        G[:, 0] = XZ[:, 0]  # X
        Z = XZ[:, 1]
        (m, q), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)  # calculate LS plane

        print(f'Params: m={m}, q={q}')

        self.m = m
        self.q = q
        self.c = -1  # y coefficient in line eq.

    def removeLine(self):
        z_line = (-self.X * self.m - self.q) * 1. / self.c
        self.Z = self.Z - z_line

    def filterGauss(self, roi, cutoff, order=0):
        nsample_cutoff = cutoff / (np.max(self.X) / np.size(self.X))
        sigma = nsample_cutoff * (1 - 0.68268949)  # corretto????
        print(f'Appliyng filter sigma: {sigma}')
        roi_filtered = ndimage.gaussian_filter1d(roi, sigma=sigma, order=order)

        return roi_filtered

    def Ra(self, cutoff, ncutoffs):  # TODO: check if this works
        nsample_cutoff = cutoff / (np.max(self.X) / np.size(self.X))
        nsample_region = nsample_cutoff * ncutoffs

        fig, ax = plt.subplots()
        ax.set_title('Ra cutoffs evaluation')

        border = round((np.size(self.Z) - nsample_region) / 2)
        roi = self.Z[border: -border]  # la roi sono i cutoff in mezzo
        ax.plot(self.X[border: -border], roi, alpha=0.3, label='roi unfiltered')

        envelope = self.filterGauss(roi, cutoff)
        roi = roi - envelope  # applico il filtro per il calcolo
        ax.plot(self.X[border: -border], roi, color='green', alpha=0.3, label='roi filtered')
        ax.plot(self.X[border: -border], envelope, color='red', label='filter envelope')

        ax.legend()
        return np.sum(abs(roi)) / np.size(roi)

    #####################################################################################################
    #                                       PLOT SECTION                                                #
    #####################################################################################################
    def init_graphics(self):  # creates a figure subdivided in grid for subplots
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(3, 2)  # rows, cols

    def prfPlot(self, fname):  # plots the profile
        ax_prf = self.fig.add_subplot(self.gs[0, :])
        ax_prf.plot(self.X, self.Z, color='teal')
        persFig(
            ax_prf,
            gridcol='grey',
            xlab='x [um]',
            ylab='z [nm]'
        )
        ax_prf.set_title('Profile ' + fname)

    def roiPlot(self, p_v=None):
        ax_roi = self.fig.add_subplot(self.gs[1, :])
        ax_roi.plot(self.X, self.Z, color='teal')
        ax_roi.plot(self.X, self.gr, color='blue')

        if p_v is not None:
            ax_roi.plot(self.X[p_v], self.gr[p_v], 'o', color='red')

        for roi in self.roi:
            ax_roi.plot(roi.X, roi.Z, color='red')  # hot, viridis, rainbow
            ax_roi.plot(self.X, self.gr, color='blue')

        persFig(
            ax_roi,
            gridcol='grey',
            xlab='x [um]',
            ylab='z [nm]'
        )

    def linePlot(self):
        ax_lin = self.fig.add_subplot(self.gs[2, 0])
        ax_lin.plot(self.X, self.Z0, color='teal')

        z_line = (- self.m * self.X - self.q) * 1. / self.c
        ax_lin.plot(self.X, z_line, color='red')

        persFig(
            ax_lin,
            gridcol='grey',
            xlab='x [um]',
            ylab='z [nm]'
        )

