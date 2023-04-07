import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from surfile import funct
from scipy import signal


@dataclass
class Roi:
    X: list
    Z: list


class Profile:
    def __init__(self):
        self.X = None
        self.Z = None
        self.Z0, self.X0 = None, None

        self.name = 'Profile'

    def openPrf(self, fname, bplt):
        z = []
        xs = 0
        zs = 0
        with open(fname, 'r') as fin:
            self.name = os.path.basename(fname)

            charlines = 0
            for line in fin.readlines():
                word = line.split()[0]
                if word == 'SPACING':  # linea di spacing x
                    xs = float(line.split()[2])
                    print(f'Spacing x: {xs}')
                if word == 'CZ':  # linea di spacing z
                    zs = float(line.split()[4]) * 10 ** 3
                    print(f'Scaling z: {zs}')

                try:  # salvo solo i valori numerici
                    z.append(float(word) * zs)
                except ValueError:
                    charlines += 1
            print(f'Skipped {charlines} word lines')
            z.pop(0)

            self.Z0 = self.Z = np.array(z)
            self.X0 = self.X = np.linspace(0, (len(z)) * xs, len(z))

            if bplt: self.__pltPrf()

    def openTS(self, fname, bplt):
        with open(fname, 'rb') as tsfile:
            firstline_sp = tsfile.readline().split(b'\x1a', maxsplit=1)
            names = firstline_sp[0]
            names = [a.decode('utf-8').strip() for a in names.split(b'\x03')]

            file_bytes = firstline_sp[-1] + tsfile.read()

            file_splits = []
            for i, name in enumerate(names[:-1]):
                d = name.encode('utf-8')
                s = file_bytes.split(d, maxsplit=1)
                if len(s) > 1:
                    file_splits.append(d + s[1])
                    file_bytes = s[1]

        for i, s in enumerate(file_splits):
            name = s[0: 42].decode('utf-8')
            print(f'{i}, Name = {name.strip()}')

        s = file_splits[int(input('Choose graph number: '))]

        Lcutoff = np.frombuffer(s[356: 360], dtype=np.single)[0]
        Factor = np.frombuffer(s[360: 364], dtype=np.single)[0]
        print(f'Lc_o = {Lcutoff}, Factor = {Factor}')

        dt = np.dtype('<i2')
        Ncutoffs = np.frombuffer(s[762: 764], dtype=dt)[0]
        Npoints = np.frombuffer(s[764: 766], dtype=dt)[0]
        Speed = np.frombuffer(s[766: 768], dtype=dt)[0]
        print(f'N = {Npoints}, Nc_o = {Ncutoffs}, Speed = {Speed}')

        self.X = self.X0 = np.linspace(0, Ncutoffs * Lcutoff, Npoints)
        self.Z = self.Z0 = np.frombuffer(s[920: 920 + 2 * Npoints], dtype=dt) / Factor

        if bplt: self.__pltPrf()

    def savecsv(self):
        name = input('Choose filename: ')
        np.savetxt('c:/monticone/' + name + '.csv',
                   np.hstack((self.X.reshape(len(self.X), 1), self.Z.reshape(len(self.Z), 1))), delimiter=';')

    def setValues(self, X, Y, bplt):
        """
        Sets the values for the profile

        Parameters
        ----------
        X: []
            The X values of the profile
        Y: []
            The Y values of the profile
        bplt: bool
            Plots the profile
        """
        self.X0 = self.X = X
        self.Z0 = self.Z = Y

        if bplt: self.__pltPrf()

    def stepAuto(self, bplt):
        """
        Calculates the step height using the auto method

        Parameters
        ----------
        bplt: bool
            Plots the step reconstruction

        Returns
        ----------
        steps: list
            The calculated step heights
        definedPeaks: bool
            False if the standard deviation of the flats is greater than step / 200
            it gives an indication on how well the steps are defined
        """

        def calcSteps():
            st = []
            defined = True
            for j in range(len(roi) - 2):  # consider j, j+1, j+2
                outerMeanL = np.mean(roi[j].Z)
                outerMeanR = np.mean(roi[j + 2].Z)
                innerMean = np.mean(roi[j + 1].Z)

                outerStdL = np.std(roi[j].Z)
                outerStdR = np.std(roi[j + 2].Z)
                innerStd = np.std(roi[j + 1].Z)

                step = innerMean - (outerMeanL + outerMeanR) / 2
                st.append(step)

                if outerStdL > abs(step) / 200 or outerStdR > abs(step) / 200 or innerStd > abs(step) / 200:
                    defined = False

            if not defined:
                print(funct.Bcol.WARNING + 'STEP HEIGHT MIGHT BE INCORRECT (PEAKS ARE POURLY DEFINED)' +
                      funct.Bcol.ENDC)

            return st, defined

        gr = np.gradient(self.Z)

        thresh = np.max(gr[30:-30]) / 1.5  # derivative threshold to detect peak, avoid border samples
        zero_cross = np.where(np.diff(np.sign(self.Z - np.mean(self.Z))))[0]
        spacing = (zero_cross[1] - zero_cross[0]) / 1.5

        peaks, _ = signal.find_peaks(gr, height=thresh, distance=spacing)
        valle, _ = signal.find_peaks(-gr, height=thresh, distance=spacing)

        roi = []  # regions of interest points
        p_v = np.sort(np.concatenate((peaks, valle)))  # every point of interest (INDEXES of x array)

        for i in range(len(p_v) - 1):
            locRange = round((p_v[i + 1] - p_v[i]) / 3)  # profile portion is 1/3 of region
            roi_start = p_v[i] + locRange
            roi_end = p_v[i + 1] - locRange
            roi.append(Roi(self.X[roi_start: roi_end],  # append to roi X and Y values of roi
                           self.Z[roi_start: roi_end]))
        steps, definedPeaks = calcSteps()

        if bplt: self.__pltRoi(gr, roi)
        return steps, definedPeaks

    def histMethod(self, bplt, bins=100):
        """
        Histogram method implementation

        Parameters
        ----------
        bins: int
            The number of bins of the histogram
        bplt: bool
            Plots the histogram of the profile

        Returns
        ----------
        (hist, edges)
            The histogram x and y
        """
        hist, edges = np.histogram(self.Z, bins)
        if bplt: self.__pltHist(hist, edges)
        return hist, edges

    def roughnessParams(self, cutoff, ncutoffs, bplt):  # TODO: adapt to filter class
        """
        Applies the indicated filter and calculates the roughness parameters

        Parameters
        ----------
        cutoff: float
            cutoff length
        ncutoffs: int
            number of cutoffs to considerate in the center of the profile
        bplt: bool
            shows roi plot

        Returns
        ----------
        (RA, RQ, RP, RV, RZ, RSK, RKU): (float, ...)
            Calculated roughness parameters
        """
        print(f'Applying filter cutoff: {cutoff}')

        # samples preparation for calculation of params
        def prepare_roi():
            nsample_cutoff = cutoff / (np.max(self.X) / np.size(self.X))
            nsample_region = nsample_cutoff * ncutoffs

            border = round((np.size(self.Z) - nsample_region) / 2)
            roi_I = self.Z[border: -border]  # la roi sono i cutoff in mezzo

            envelope = self.__gaussianKernel(self.Z, cutoff)  # TODO: move method to roughness module
            roi_F = roi_I - envelope[border: -border]  # applico il filtro per il calcolo

            if bplt: self.__pltRoughness(ncutoffs, cutoff, border, roi_I, roi_F, envelope)
            return roi_F  # cutoff roi

        roi = prepare_roi()

        RA = np.sum(abs(roi)) / np.size(roi)
        RQ = np.sqrt(np.sum(abs(roi ** 2)) / np.size(roi))
        RP = abs(np.max(roi))
        RV = abs(np.min(roi))
        RT = RP + RV
        RSK = (np.sum(roi ** 3) / np.size(roi)) / (RQ ** 3)
        RKU = (np.sum(roi ** 4) / np.size(roi)) / (RQ ** 4)
        return RA, RQ, RP, RV, RT, RSK, RKU

    def findMaxArcSlope(self, R):
        """
        Used to find the max measured slopes of arc of radius R

        Parameters
        ----------
        R: float
            The radius of the arc

        Returns
        ----------
        phi_max1: float
            The slope calculated at breackpoint 1 (first nan value)
        phi_max2: float
            The slope calculated at breackpoint 2 (last measured point)
        """
        try:
            bound_nan = np.argwhere(np.isnan(self.Z))[0][-1] - 1
        except IndexError:
            bound_nan = 0

        Rms_1 = self.X[bound_nan - 1] - self.X[0]
        Rms_2 = self.X[np.nanargmin(self.Z)] - self.X[0]  # find the furthest max point
        phi_max_1 = np.arcsin(Rms_1 / R)
        phi_max_2 = np.arcsin(Rms_2 / R)
        return phi_max_1, phi_max_2

    def arcRadius(self, bplt, skip=0.05):
        """
        Calculates the radius of the arc varying the z (top to bottom)

        Parameters
        ----------
        skip: float
            The first micrometers to skip
        bplt: bool
            Plots the calculated radius at all the z values

        Returns
        ----------
        (r, z): (np.array(), ...)
            The radius and the respective z values
        """
        r = []
        z = []
        for i, p in enumerate(self.Z[0:-1]):
            if np.isnan(p): break
            ri = self.X[i] - self.X[0]
            zeh = np.abs(self.Z[0] - p)
            if zeh > skip:  # skip the first nanometers
                z.append(zeh)
                radius = (ri ** 2 + zeh ** 2) / (2 * zeh)
                r.append(radius)

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(z, r)
            funct.persFig(
                [ax],
                gridcol='grey',
                xlab='Depth',
                ylab='Radius'
            )
            plt.show()
        return r, z

    #####################################################################################################
    #                                       PLOT SECTION                                                #
    #####################################################################################################
    def __pltPrf(self):  # plots the profile
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.X, self.Z, color='teal')
        funct.persFig(
            [ax],
            gridcol='grey',
            xlab='x [mm]',
            ylab='z [um]'
        )
        ax.set_title(self.name)
        plt.show()

    def pltCompare(self):  # plots the profile
        fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
        ax.plot(self.X0, self.Z0, color='teal')
        bx.plot(self.X, self.Z, color='teal')
        funct.persFig(
            [ax, bx],
            gridcol='grey',
            xlab='x [mm]',
            ylab='z [um]'
        )
        ax.set_title(self.name)
        plt.show()

    def __pltRoi(self, gr, rois):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.X, self.Z, color='teal')
        ax.plot(self.X, gr, color='blue')

        for roi in rois:
            ax.plot(roi.X, roi.Z, color='red')
            ax.plot(self.X, gr, color='blue', linewidth=0.2)

        funct.persFig(
            [ax],
            gridcol='grey',
            xlab='x [mm]',
            ylab='z [um]'
        )
        plt.show()

    # def linePlot(self):
    #     ax_lin = self.fig.add_subplot(self.gs[2, 0])
    #     ax_lin.plot(self.X, self.Z0, color='teal')
    #
    #     z_line = (- self.m * self.X - self.q) * 1. / self.c
    #     ax_lin.plot(self.X, z_line, color='red')
    #
    #     funct.persFig(
    #         [ax_lin],
    #         gridcol='grey',
    #         xlab='x [mm]',
    #         ylab='z [um]'
    #     )

    def __pltHist(self, hist, edges):
        fig = plt.figure()
        ax_ht = fig.add_subplot(111)
        ax_ht.hist(edges[:-1], bins=edges, weights=hist / np.size(self.Z) * 100, color='red')
        funct.persFig(
            [ax_ht],
            gridcol='grey',
            xlab='z [nm]',
            ylab='pixels %'
        )
        plt.show()

    def __pltRoughness(self, ncutoffs, cutoff, border, roi_I, roi_F, envelope):
        fig, ax = plt.subplots()

        twin = ax.twinx()
        twin.set_ylabel('Filtered roi')

        ax.set_title(f'Gaussian filter: cutoffs: {ncutoffs}, cutoff length: {cutoff}')
        ax.plot(self.X, self.Z, alpha=0.2, label='Data')
        ax.plot(self.X[border: -border], roi_I, alpha=0.3, label='roi unfiltered')

        twin.set_ylim(np.min(roi_F) - 0.7 * (np.max(roi_F) - np.min(roi_F)),
                      np.max(roi_F) + 0.7 * (np.max(roi_F) - np.min(roi_F)))
        rF, = twin.plot(self.X[border: -border], roi_F, color='green', alpha=0.6, label='roi filtered')
        twin.tick_params(axis='y', colors=rF.get_color())

        ax.plot(self.X, envelope, color='red', label='filter envelope')

        ax.legend()
        plt.show()
