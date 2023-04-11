import copy

import numpy as np
from dataclasses import dataclass

from alive_progress import alive_bar
from matplotlib import pyplot as plt
from scipy import signal

from surfile import profile, surface, funct, extractor


@dataclass
class Roi:
    X: list
    Z: list


def findHfromHist(hist, edges):
    """
    Finds the 2 maximum values in the histogram and calculates the distance of
    the peaks -> gives info about sample step height

    Parameters
    ----------
    hist: np.array
        histogram y values
    edges: np.array
        histogram bins

    Returns
    ----------
    h: float
        Height of sample
    """
    ml = 0
    mh = 0
    binl = 0
    binh = 0
    i = 0
    for edge in edges[:-1]:
        if edge < 0:
            binl = edge if hist[i] > ml else binl
            ml = max(ml, hist[i])

        else:
            binh = edge if hist[i] > mh else binh
            mh = max(mh, hist[i])

        i = i + 1

    print(f'Max left {ml} @ {binl} \nMax right {mh} @ {binh}')
    print(f'Height: {binh - binl}')

    return binh - binl


class ProfileForm:
    @staticmethod
    def stepAuto(obj: profile.Profile, bplt=False):
        """
        Calculates the step height using the auto method

        Parameters
        ----------
        obj : profile.Profile
            The profile object on wich the steps are calculated
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
            for j in range(len(rois) - 2):  # consider j, j+1, j+2
                outerMeanL = np.mean(rois[j].Z)
                outerMeanR = np.mean(rois[j + 2].Z)
                innerMean = np.mean(rois[j + 1].Z)

                outerStdL = np.std(rois[j].Z)
                outerStdR = np.std(rois[j + 2].Z)
                innerStd = np.std(rois[j + 1].Z)

                step = innerMean - (outerMeanL + outerMeanR) / 2
                st.append(step)

                if outerStdL > abs(step) / 200 or outerStdR > abs(step) / 200 or innerStd > abs(step) / 200:
                    defined = False

            if not defined:
                print(funct.Bcol.WARNING + 'STEP HEIGHT MIGHT BE INCORRECT (PEAKS ARE POURLY DEFINED)' +
                      funct.Bcol.ENDC)

            return st, defined

        gr = np.gradient(obj.Z)

        thresh = np.max(gr[30:-30]) / 1.5  # derivative threshold to detect peak, avoid border samples
        zero_cross = np.where(np.diff(np.sign(obj.Z - np.mean(obj.Z))))[0]
        spacing = (zero_cross[1] - zero_cross[0]) / 1.5

        peaks, _ = signal.find_peaks(gr, height=thresh, distance=spacing)
        valle, _ = signal.find_peaks(-gr, height=thresh, distance=spacing)

        rois = []  # regions of interest points
        p_v = np.sort(np.concatenate((peaks, valle)))  # every point of interest (INDEXES of x array)

        for i in range(len(p_v) - 1):
            locRange = round((p_v[i + 1] - p_v[i]) / 3)  # profile portion is 1/3 of region
            roi_start = p_v[i] + locRange
            roi_end = p_v[i + 1] - locRange
            rois.append(Roi(obj.X[roi_start: roi_end],  # append to roi X and Y values of roi
                            obj.Z[roi_start: roi_end]))
        steps, definedPeaks = calcSteps()

        if bplt:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(obj.X, obj.Z, color='teal')
            ax.plot(obj.X, gr, color='blue')

            for roi in rois:
                ax.plot(roi.X, roi.Z, color='red')
                ax.plot(obj.X, gr, color='blue', linewidth=0.2)

            funct.persFig(
                [ax],
                gridcol='grey',
                xlab='x [mm]',
                ylab='z [um]'
            )
            plt.show()
        return steps, definedPeaks

    @staticmethod
    def histHeight(obj: profile.Profile, bins=100, bplt=False):
        """
        Histogram method implementation

        Parameters
        ----------
        obj : profile.Profile
            The profile object on wich the height is calculated
        bins: int
            The number of bins of the histogram
        bplt: bool
            Plots the histogram of the profile

        Returns
        ----------
        height: float
            The calculated height of the surface
        (hist, edges)
            The histogram x and y
        """
        hist, edges = np.histogram(obj.Z, bins)
        height = findHfromHist(hist=hist, edges=edges)
        if bplt:
            fig = plt.figure()
            ax_ht = fig.add_subplot(111)
            ax_ht.hist(edges[:-1], bins=edges, weights=hist / np.size(obj.Z) * 100, color='red')
            funct.persFig(
                [ax_ht],
                gridcol='grey',
                xlab='z [nm]',
                ylab='pixels %'
            )
            plt.show()
        return height, hist, edges

    @staticmethod
    def arcSlope(obj: profile.Profile, R):
        """
        Used to find the max measured slopes of arc of radius R

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the maximum meaasured slope is calculated
        R: float
            The nominal radius of the arc

        Returns
        ----------
        phi_max1: float
            The slope calculated at breackpoint 1 (first nan value)
        phi_max2: float
            The slope calculated at breackpoint 2 (last measured point)
        """
        try:
            bound_nan = np.argwhere(np.isnan(obj.Z))[0][-1] - 1
        except IndexError:
            bound_nan = 0

        Rms_1 = obj.X[bound_nan - 1] - obj.X[0]
        Rms_2 = obj.X[np.nanargmin(obj.Z)] - obj.X[0]  # find the furthest max point
        phi_max_1 = np.arcsin(Rms_1 / R)
        phi_max_2 = np.arcsin(Rms_2 / R)
        return phi_max_1, phi_max_2

    @staticmethod
    def arcRadius(obj: profile.Profile, skip=0.05, bplt=False):
        """
        Calculates the radius of the arc varying the z (top to bottom)

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the radius is calculated
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
        for i, p in enumerate(obj.Z[0:-1]):
            if np.isnan(p): break
            ri = obj.X[i] - obj.X[0]
            zeh = np.abs(obj.Z[0] - p)
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


class SurfaceForm:
    @staticmethod
    def histHeight(obj: surface.Surface, bins=100, bplt=False):
        """
        Histogram method implementation

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the height is calculated
        bins: int
            The number of bins of the histogram
        bplt: bool
            Plots the histogram of the profile

        Returns
        ----------
        height: float
            The calculated height of the surface
        (hist, edges)
            The histogram x and y
        """
        hist, edges = np.histogram(obj.Z, bins)
        height = findHfromHist(hist=hist, edges=edges)
        if bplt:
            fig = plt.figure()
            ax_ht = fig.add_subplot(111)
            ax_ht.hist(edges[:-1], bins=edges, weights=hist / np.size(obj.Z) * 100, color='red')
            funct.persFig(
                [ax_ht],
                gridcol='grey',
                xlab='z [nm]',
                ylab='pixels %'
            )
            plt.show()
        return height, hist, edges

    @staticmethod
    def sphereSlope(obj: surface.Surface, R, angleStep, start='local', bplt=False):
        """
        Returns the maximum measurable slope in every direction

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the maximum slope is calculated
        R: float
            The nominal radius of the sphere
        angleStep : int
            The angle used to rotate the image after every iteration
        start : str
            Method used to find the start (x, y) point on the topography
                'max': the start point is the maximum Z of the topography
                'fit': the start point is the center of the best fit sphere
                'center': the start point is the center of the topography
                'local': the start point is the local maximum closest to the center of the topography
        bplt: bool
            Plots the slope at the different angles

        Returns
        ----------
        (phi_max1, phi_max2): (np.array(), ...)
            The 2 slopes calculated at breackpoints 1 and 2 respectively
        """
        meas_slope1, meas_slope2 = [], []
        with alive_bar(int(360 / angleStep), force_tty=True,
                       title='Slope', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for a in range(0, 360, angleStep):
                obj.rotate(a)
                slopeprofile = copy.copy(extractor.SphereExtractor.sphereProfile(startP=start, bplt=False))
                ms1, ms2 = ProfileForm.arcSlope(slopeprofile, R)  # 350 um radius
                meas_slope1.append(np.rad2deg(ms1))
                meas_slope2.append(np.rad2deg(ms2))
                bar()

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(range(0, 360, angleStep), meas_slope1, 'r', label='Max slope Rms1')
            ax.plot(range(0, 360, angleStep), meas_slope2, 'b', label='Max slope Rms2')
            ax.legend()

            fig2, bx = plt.subplots(subplot_kw={'projection': 'polar'})
            bx.plot(np.deg2rad(range(0, 360, angleStep)), meas_slope1, 'r', label='Max slope Rms1')
            bx.plot(np.deg2rad(range(0, 360, angleStep)), meas_slope2, 'b', label='Max slope Rms2')
            bx.legend()
            plt.show()

        return meas_slope1, meas_slope2

    @staticmethod
    def sphereRadius(obj: surface.Surface, angleStepSize, start='local', bplt=False):
        """
        Returns the radius of the profile in every direction

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the radius is calculated
        angleStepSize : int
            The angle used to rotate the image after every iteration
        start : str
            Method used to find the start (x, y) point on the topography
                'max': the start point is the maximum Z of the topography
                'fit': the start point is the center of the best fit sphere
                'center': the start point is the center of the topography
                'local': the start point is the local maximum closest to the center of the topography
        bplt: bool
            Plots the radius at the different angles

        Returns
        ----------
        (yr, yz): (np.array(), ...)
            The mean of the radius and the mean of the different heights where the radius is calculated
        """
        rs = []
        zs = []
        fig, ax = plt.subplots()

        with alive_bar(int(360 / angleStepSize), force_tty=True,
                       title='Radius', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for a in range(0, 360, angleStepSize):
                obj.rotate(a)
                radiusprofile = copy.copy(extractor.SphereExtractor.sphereProfile(startP=start, bplt=False))
                r, z = ProfileForm.arcRadius(radiusprofile, bplt=False)  # 350 um radius
                rs.append(r)
                zs.append(z)
                if bplt: ax.plot(z, r, alpha=0.2)
                bar()

        yr, error = funct.tolerant_mean(rs)
        yz, error = funct.tolerant_mean(zs)
        ax.plot(yz, yr, color='red')
        ax.set_ylim(0, max(yr))
        if bplt: plt.show()

        return yr, yz
