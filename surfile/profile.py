import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from dataclasses import dataclass
from alive_progress import alive_bar

from surfile import funct
from scipy import signal, ndimage, integrate


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

    # @funct.timer
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
        :param X: x values of profile
        :param Y: y values of profile
        """
        self.X0 = self.X = X
        self.Z0 = self.Z = Y

        if bplt: self.__pltPrf()

    def stepAuto(self, bplt):  # TODO: can you find a way to automate minD calculations?
        """
        Calculates the step height using the auto method
        :return: steps: array containing all found step heights on profile
        """

        def calcSteps():
            st = []
            definedPeaks = True
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
                    definedPeaks = False

            if not definedPeaks:
                print(funct.Bcol.WARNING + 'STEP HEIGHT MIGHT BE INCORRECT (PEAKS ARE POURLY DEFINED)' +
                      funct.Bcol.ENDC)

            return st, definedPeaks

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
        steps, ok = calcSteps()

        if bplt: self.__pltRoi(gr, roi)
        return steps, ok

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

        print(f'LS line method -> Params: m={m:.3f}, q={q:.3f}')
        self.__removeLine(m, q, -1)
        return m, q

    def allignWithHist(self, final_m, bplt):
        m, q = self.fitLineLS()  # preprocess inclination
        tot_bins = int(np.size(self.X) / 20)
        # threshold = 50 / tot_bins

        line_m = m / 10  # start incline
        print(f'Hist method -> Start slope  {line_m}')

        fig = plt.figure()
        ax_h = fig.add_subplot(211)
        bx_h = fig.add_subplot(212)

        def calcNBUT():
            hist, edges = self.histMethod(bins=tot_bins, bplt=False)  # make the hist
            weights = hist / np.size(self.Z) * 100
            threshold = np.max(weights) / 20
            n_bins_under_threshold = np.size(np.where(weights < threshold)[0])  # how many bins under th

            if bplt:
                ax_h.clear()
                ax_h.hist(edges[:-1], bins=edges, weights=weights, color='red')
                ax_h.plot(edges[:-1], integrate.cumtrapz(hist / np.size(self.Z) * 100, edges[:-1], initial=0))
                ax_h.text(.25, .75, f'NBUT = {n_bins_under_threshold} / {tot_bins}, line_m = {line_m:.3f} -> {final_m}',
                          horizontalalignment='left', verticalalignment='bottom', transform=ax_h.transAxes)
                ax_h.axhline(y=threshold, color='b')

                bx_h.clear()
                bx_h.plot(self.X, self.Z)
                plt.draw()
                plt.pause(0.05)

            return n_bins_under_threshold

        param = calcNBUT()
        n_row = 0
        # until I have enough bins < th keep loop
        while np.abs(line_m) > final_m:  # nbut < (tot_bins - tot_bins / 20):
            self.Z = self.Z - self.X * line_m
            param_old = param
            param = calcNBUT()

            if param < param_old:  # invert rotation if we are going the wrong way
                line_m = -line_m / 2

            if param == param_old:
                n_row += 1
                if n_row >= 15: break  # we got stuck for too long
            else:
                n_row = 0
        print(f'Hist method -> End slope    {line_m}')

    def histMethod(self, bplt, bins=100):
        """
        histogram method implementation
        :return: histogram values, bin edges values
        """
        hist, edges = np.histogram(self.Z, bins)
        if bplt: self.__pltHist(hist, edges)
        return hist, edges

    def __removeLine(self, m, q, c):
        z_line = (-self.X * m - q) * 1. / c
        self.Z = self.Z - z_line

    def __gaussianKernel(self, roi, cutoff, order=0):
        nsample_cutoff = cutoff / (np.max(self.X) / np.size(self.X))

        alpha = np.sqrt(np.log(2) / np.pi)
        sigma = nsample_cutoff * (alpha / np.sqrt(2 * np.pi))  # da norma ISO 16610-21  # * (1 - 0.68268949)
        print(f'Appliyng filter sigma: {sigma}')
        roi_filtered = ndimage.gaussian_filter1d(roi, sigma=sigma, order=order)

        return roi_filtered

    def removeFormPolynomial(self, degree, bplt, comp=lambda a, b: a < b, bound=None):
        """
        Removes the form from the profile calculated as a polynomial fit of the data
        :param comp: comparison method between comp and profile
        :param degree: polynomial degree
        :param bound: if not set the fit uses all points, if set the fit uses all points below the values, if set to
        True the fit uses only the values below the average value of the profile
        """
        if bound is None:
            coeff = np.polyfit(self.X, self.Z, degree)
        elif bound:
            bound = np.mean(self.Z)
            ind = np.argwhere(comp(self.Z, bound)).ravel()
            coeff = np.polyfit(self.X[ind], self.Z[ind], degree)
        else:
            ind = np.argwhere(comp(self.Z, bound)).ravel()
            coeff = np.polyfit(self.X[ind], self.Z[ind], degree)

        form = np.polyval(coeff, self.X)
        if bplt:
            fig, ax = plt.subplots()
            ax.plot(self.X, self.Z, self.X, form)
            plt.show()
        self.Z -= form

    def gaussianFilter(self, cutoff):
        self.Z = self.__gaussianKernel(self.Z, cutoff)

    def morphFilter(self, radius):
        """
        Apllies a morphological filter as described in ISO-21920,
        rolls a disk  of radius R (in mm) along the original profile
        """

        def morph(profile_x, profile_y, radius):
            spacing = profile_x[1] - profile_x[0]
            n_radius = int(radius / spacing)
            n_samples = len(profile_x)

            fig, ax = plt.subplots()
            ax.axis('equal')

            filler_L = np.ones(n_radius) * profile_y[0]
            filler_R = np.ones(n_radius) * profile_y[-1]

            profile_x_filled = np.arange(start=profile_x[0] - radius, stop=profile_x[-2] + radius, step=spacing)
            profile_y_filled = np.concatenate([filler_L, profile_y, filler_R])

            profile_out = profile_y_filled - radius

            with alive_bar(n_samples, force_tty=True,
                           title='Morph', theme='smooth',
                           elapsed_end=True, stats_end=True, length=30) as bar:
                for i in range(n_radius, n_samples + n_radius):
                    loc_x = np.linspace(profile_x_filled[i - n_radius], profile_x_filled[i + n_radius], 1000)
                    loc_p = np.interp(loc_x, profile_x_filled[i - n_radius:i + n_radius],
                                      profile_y_filled[i - n_radius:i + n_radius])

                    alpha = profile_x_filled[i]
                    beta = profile_out[i]  # start under the profile

                    cerchio = np.sqrt(-(alpha ** 2 - radius ** 2) + 2 * alpha * loc_x - loc_x ** 2) + beta

                    dbeta = -10 * radius
                    disp = 0

                    bar()
                    up = len(np.argwhere((cerchio - loc_p) > 0))
                    if up > 1:
                        while np.abs(dbeta) > radius / 1000:
                            cerchio += dbeta
                            disp -= dbeta
                            up = len(np.argwhere((cerchio - loc_p) > 0))

                            if (dbeta < 0 and up == 0) or (dbeta > 0 and up != 0):
                                dbeta = -dbeta / 2

                        profile_out[i] -= disp

            profile_out = profile_out[n_radius: -n_radius]
            ax.plot(profile_x, profile_y, profile_x, profile_out)
            plt.show()

            return profile_out

        self.Z = morph(self.X, self.Z, radius)

    def cutProfile(self):
        def onClose(event):
            xmin, xmax = span.extents
            print(xmin, xmax)
            i_near = lambda arr, val: (np.abs(arr - val)).argmin()
            start_x, end_x = i_near(self.X, xmin), i_near(self.X, xmax)

            self.X = self.X[start_x: end_x]
            self.Z = self.Z[start_x: end_x]

        fig, ax = plt.subplots()
        span = SpanSelector(ax, lambda a, b: None,
                            direction='horizontal', useblit=True,
                            button=[1, 3],  # don't use middle button
                            interactive=True)

        ax.plot(self.X, self.Z)
        ax.set_title('Choose region')
        fig.canvas.mpl_connect('close_event', onClose)
        plt.show()

    def roughnessParams(self, cutoff, ncutoffs, bplt):  # TODO: check if this works
        """
        Applies the indicated filter and calculates the roughness parameters
        :param bplt: shows roi plot
        :param cutoff: co length
        :param ncutoffs: number of cutoffs to considerate in the center of the profile
        :return: RA, RQ, RP, RV, RZ, RSK, RKU
        """
        print(f'Applying filter cutoff: {cutoff}')
        # samples preparation for calculation of params
        def prepare_roi():
            nsample_cutoff = cutoff / (np.max(self.X) / np.size(self.X))
            nsample_region = nsample_cutoff * ncutoffs

            border = round((np.size(self.Z) - nsample_region) / 2)
            roi_I = self.Z[border: -border]  # la roi sono i cutoff in mezzo

            envelope = self.__gaussianKernel(self.Z, cutoff)
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
        Used to find the max measured slope of sphere of radius R
        """
        try:
            bound_nan = np.argwhere(np.isnan(self.Z))[0][-1] - 1
        except IndexError:
            bound_nan = 0

        Rms_1 = self.X[bound_nan-1] - self.X[0]
        Rms_2 = self.X[np.nanargmin(self.Z)] - self.X[0]  # find the furthest max point
        phi_max_1 = np.arcsin(Rms_1 / R)
        phi_max_2 = np.arcsin(Rms_2 / R)
        return phi_max_1, phi_max_2

    def arcRadius(self, bplt):
        # TODO : check if it works
        r = []
        z = []
        for i, p in enumerate(self.Z):
            if np.isnan(p): break
            ri = self.X[i] - self.X[0]
            zeh = np.abs(self.Z[0] - p)
            z.append(zeh)
            r.append((ri**2 + zeh**2) / (2 * zeh))

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
