"""
'surfile.analysis'
- analysis for:
    - Profiles
    - Surfaces

@author: Andrea Giura, Dorothee Hueser
"""

import copy

import numpy as np
from dataclasses import dataclass

from alive_progress import alive_bar
from matplotlib import pyplot as plt, cm
from scipy import signal, optimize, stats

from surfile import geometry, profile, surface, funct, extractor
from surfile.funct import classOptions, options, rcs


@dataclass
class Roi:
    """Simple class to handle profile sections"""
    X: list
    Z: list


def _findHfromHist(hist, edges):
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


def _tolerant_mean(arrs: list):
    """
    Calculates the average between multiple arrays of different length

    Parameters
    ----------
    arrs: list
        The arrays to be processed

    Returns
    -------
    mean: np.array
        The mean calculated
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)


@classOptions(decorator=options(
    bplt=rcs.params['bpMor'],
    save=rcs.params['spMor'],
    csvPath=rcs.params['cpMor'])
)
class ProfileAnalysis:
    """
    Class that contains all the methods relative to Profile
    analysis

    Notes
    -----
    All the methods in this class are implemented as a @staticmethod
    so this class is only used as a namespace to include all the routines
    relative to profile analysis.
    
    Many of the methods of this function are called from functions 
    of the SurfaceAnalysis class, but they can be called also to process
    single profile objects.
    """
    @staticmethod
    def stepAuto(obj: profile.Profile, bplt=False):
        """
        Calculates the step height by finding the position of
        the walls automatically.

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
            
        Notes
        -----
        To automate the selection of the areas @ the left, right and center of the step,
        which will henceforth be called regions of interest (ROIs), the programme calculates 
        the first derivative of the profile p'(x) and a threshold value that will be used 
        to search peaks above its value. Once the peaks have been defined, it is possible to 
        derive the ROIs as segments in between the peaks and one third the width of the distance 
        between the two peaks that include it.
        
        $h_{step}=\\overline{ROI_{middle}}-\\frac{\\overline{ROI_{left}}+\\overline{ROI_{right}}}{2}$
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
        # spacing = (zero_cross[1] - zero_cross[0]) / 1.5

        peaks, _ = signal.find_peaks(gr, height=thresh)   # , distance=spacing)
        valle, _ = signal.find_peaks(-gr, height=thresh)  # , distance=spacing)

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
            ax.set_title(obj.name)
        return steps, definedPeaks

    @staticmethod
    def histHeight(obj: profile.Profile, bins=None, bplt=False):
        """
        Calculates the height of the sample using the histogram 
        method.

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
            
        Notes
        -----
        The histogram of the Z values of the object describes how many pixels 
        are present in the image at a certain height. In the case of a step
        height sample the histogram presents two peaks, the program calculates
        the difference between the two peaks and returns the height.
        """
        b = bins
        if bins is None:
            b = 2 * stats.iqr(obj.Z) / (obj.Z.size ** (1 / 3))  # Freedman-Diaconis
            print(f'Using {b} bins in hist')

        hist, edges = np.histogram(obj.Z, b)
        height = _findHfromHist(hist=hist, edges=edges)

        perc_hist = hist / np.size(obj.Z) * 100
        af_curve = np.zeros(bins)  # abbott firestone curve
        af_curve[0] = perc_hist[0]
        for i, ele in enumerate(np.flip(perc_hist[1:])):
            af_curve[i + 1] = af_curve[i] + ele

        if bplt:
            fig, (ax_ht, bx_af) = plt.subplots(nrows=1, ncols=2)
            ax_ht.hist(edges[:-1], bins=edges, weights=perc_hist, color='red')
            bx_af.plot(af_curve, np.flip(edges[:-1]))
            funct.persFig(
                [ax_ht],
                gridcol='grey',
                xlab='z [nm]',
                ylab='pixels %'
            )
            funct.persFig(
                [bx_af],
                gridcol='grey',
                xlab='pixels %',
                ylab='z [nm]'
            )
            ax_ht.set_title(obj.name)
        return height, hist, edges

    @staticmethod
    def arcSlope(obj: profile.Profile, R):
        """
        Finds the max measured slopes of arc of radius R at two breakpoints

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
        
        Notes
        -----
        The first breakpoint $b_1$ is taken at the first non-measured point, 
        the second breakpoint $b_2$ is taken at the last measured point.
        
        $\\Phi_{MS1}=asin(\\frac{b_1}{R})$
        $\\Phi_{MS2}=asin(\\frac{b_2}{R})$
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
            
        Notes
        -----
        The radius of the arc for a specific height is then calculated as:
        $ R_i=\\frac{x_i^2+z_{eh,i}^2}{2z_{eh,i}}$
        where $x_i^2$ is the $x$ coordinate of the $i_{th}$ point, $z_{(eh,i)}$ 
        is the distance between the maximum value of the profile and the $z$ 
        coordinate of point $i$; $R_i$ represent the radius calculated at point $i$.
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
            ax.set_title(obj.name)
        return r, z

    @staticmethod
    def grating_1d(obj: profile.Profile, nom_pitch, bplt=False):
        """
        Determines height and pitch of 1D gratings
        of line bars of rectangular cross-section, where the trench width
        is equal to the bar width.
        The bar cross-sections are modelled as rectangular boxes with
        smoothed corners using a combination of sigmoidal functions.

        Parameters
        ----------
        obj: profile.Profile
            The profile on which the steps are evaluated
        nom_pitch: float
            the nominal pitch of the sample
        bplt: Bool
            if true plots the sine and the sigmoid fit

        Returns
        ----------
        x_c: np.array
            The center position of the features
        h_c: np.array
            The calculated heights of the features
            
        Notes
        -----
        At first the extracted profile is fitted with a least square sine wave,
        this is used to find the centres of the features.\n
        $\\min_b {\\sum_{i=0}^{n_1}(z_i-\\frac{1}{2}b_0cos(\\frac{\\pi}{b_2}(x_i-b_1))-b_3-b_4x_i)^2}$\n
        The sigmoidal funtion fit is in the form:\n
        $f_s(x_{SW})=\\frac{1}{1+e^{\\frac{s(p_1-x)-x_{SW}}{p_4}}}$\n
        $z_M(\\mathbf{p},x)=p_0(f_{+1}(\\frac{1}{2}p_2)f_{-1}(\\frac{1}{2}p_2)-f_{+1}
        (\\frac{3}{2}p_2)f_{-1}(\\frac{3}{2}p_2))$
        """
        x_c = []  # center positions of features
        h_c = []  # height of features

        # cosine fit for the whole profile
        _cos = lambda p: 0.5 * p[0] * np.cos(np.pi * (obj.X - p[1]) / p[2]) + p[3] + p[4] * obj.X
        _fs = lambda p, s, xsw, x: 1 / (1 + np.exp((s * (p[1] - x) - xsw) / p[4]))
        _sigm = lambda p, x: p[0] * (_fs(p, 1, 1 / 2 * p[2], x) * _fs(p, -1, 1 / 2 * p[2], x) -
                                     _fs(p, 1, 3 / 2 * p[2], x) * _fs(p, -1, 3 / 2 * p[2], x) + 
                                     1 / 2) + p[3]
        
        zmax = np.nanmax(obj.Z)
        zmin = np.nanmin(obj.Z)
        p_init = np.array([zmax - zmin, 0, 0.5 * nom_pitch, 0.5 * (zmax + zmin), 0])
        popt = optimize.leastsq(lambda p: _cos(p) - obj.Z, p_init)[0]
        # print(f'Cosine period (sample pitch approx): {2 * popt[2]}')

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(obj.X, obj.Z, obj.X, _cos(popt))
            ax.set_title(obj.name)
            
            funct.persFig([ax], xlab='x [um]', ylab='z [um]')

        # first maximum is in p1, the following are in p1 + 2p2 * ip
        period = 2 * popt[2]
        for ip in range(1, int(np.nanmax(obj.X) / period)):
            xip = popt[1] + period * ip
            boolbox = (xip - 1.3 * popt[2] < obj.X) & (obj.X < xip + 1.3 * popt[2])
            xbox = obj.X[boolbox]  # x of the single box
            zbox = obj.Z[boolbox]  # z of the single box

            # now we can fit the sigmoid
            p_init_sigm = np.array([np.ptp(zbox), xip,  period / 2, np.nanmin(zbox) + np.ptp(zbox) / 2, 0.2])
            popt_sigm = optimize.leastsq(lambda p: _sigm(p, xbox) - zbox, p_init_sigm)[0]

            h = _sigm(popt_sigm, popt_sigm[1]) - _sigm(popt_sigm, popt_sigm[1] + popt_sigm[2])
            d_form = popt_sigm[0] / h
            th = 1
            h_c.append(h if 1 - th < d_form < 1 + th else np.nan)  # check on std
            x_c.append(popt_sigm[1])

            if bplt: ax.plot(xbox, _sigm(popt_sigm, xbox))
            # fig2, bx = plt.subplots()
            # bx.plot(xbox, _sigm(popt_sigm, xbox), 'b', 
            #         xbox, _sigm(p_init_sigm, xbox), 'r',
            #         xbox, zbox, 'g')

        return np.array(x_c), np.abs(np.array(h_c))


@classOptions(decorator=options(
    bplt=rcs.params['bsMor'],
    save=rcs.params['ssMor'],
    csvPath=rcs.params['csMor'])
)
class SurfaceAnalysis:
    """
    Class that contains all the methods relative to Surface
    analysis

    Notes
    -----
    All the methods in this class are implemented as a @staticmethod
    so this class is only used as a namespace to include all the routines
    relative to surface analysis.
    """
    @staticmethod
    def histHeight(obj: surface.Surface, bins=None, bplt=False):
        """
        Calculates the height of the sample using the histogram 
        method.

        Parameters
        ----------
        obj : surface.Surface
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
            
        Notes
        -----
        The histogram of the Z values of the object describes how many pixels 
        are present in the image at a certain height. In the case of a step
        height sample the histogram presents two peaks, the program calculates
        the difference between the two peaks and returns the height.
        """
        b = bins
        if bins is None:
            # bw = 2 * stats.iqr(obj.Z[np.isfinite(obj.Z)]) / (obj.Z.size ** (1/3))  # Freedman-Diaconis
            b = int(np.sqrt(obj.Z.size))
            print(f'Using {b} bins in hist')

        hist, edges = np.histogram(obj.Z[np.isfinite(obj.Z)], bins=b)
        height = _findHfromHist(hist=hist, edges=edges)
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
            ax_ht.set_title(obj.name)
        return height, (hist, edges)

    @staticmethod
    def maxMeasSlope(obj: surface.Surface, R, angleStep, start='local', bplt=False):
        """
        Calculates the maximum measurable slope in the radial directions
        given a topography of a sphere measured with the instrument.
        See ProfileAnalysis.arcSlope() for details.

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the maximum slope is calculated
        R : float
            The nominal radius of the sphere
        angleStep : int
            The angle used to rotate the image after every iteration
        start : str
            Method used to find the start (x, y) point on the topography
                'max': the start point is the maximum Z of the topography
                'fit': the start point is the center of the best fit sphere
                'center': the start point is the center of the topography
                'local': the start point is the local maximum closest to the center of the topography
        bplt : bool
            Plots the slope at the different angles (linear and radial plots)

        Returns
        ----------
        phi_max1 : np.array
        phi_max2 : np.array
            The 2 slopes calculated at breackpoints 1 and 2 respectively
        """
        meas_slope1, meas_slope2 = [], []
        with alive_bar(int(360 / angleStep), force_tty=True,
                       title='Slope', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for a in range(0, 360, angleStep):
                obj.rotate(a)
                slopeprofile = copy.copy(extractor.SphereExtractor.sphereProfile(obj, startP=start, bplt=False))
                ms1, ms2 = ProfileAnalysis.arcSlope(slopeprofile, R)  # 350 um radius
                meas_slope1.append(np.rad2deg(ms1))
                meas_slope2.append(np.rad2deg(ms2))
                bar()

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(range(0, 360, angleStep), meas_slope1, 'r', label='Max slope Rms1')
            ax.plot(range(0, 360, angleStep), meas_slope2, 'b', label='Max slope Rms2')
            ax.legend()
            funct.persFig([ax], xlab='Radial angle [deg]', ylab="Measured slope [deg]")
            ax.set_title(obj.name)

            fig2, bx = plt.subplots(subplot_kw={'projection': 'polar'})
            bx.plot(np.deg2rad(range(0, 360, angleStep)), meas_slope1, 'r', label='Max slope Rms1')
            bx.plot(np.deg2rad(range(0, 360, angleStep)), meas_slope2, 'b', label='Max slope Rms2')
            bx.legend()
            bx.set_title(obj.name)

        return meas_slope1, meas_slope2

    @staticmethod
    def sphereRadius(obj: surface.Surface, angleStepSize, start='local', bplt=False):
        """
        Returns the radius of the sphere in the radial direction
        and at each height. Calculates and plots the average value of
        the radius at each height.
        See ProfileAnalysis.arcRadius() for details.

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
                radiusprofile = copy.copy(extractor.SphereExtractor.sphereProfile(obj, startP=start, bplt=False))
                r, z = ProfileAnalysis.arcRadius(radiusprofile, bplt=False)  # 350 um radius
                rs.append(r)
                zs.append(z)
                if bplt: ax.plot(z, r, alpha=0.2)
                bar()

        yr, error = _tolerant_mean(rs)
        yz, error = _tolerant_mean(zs)
        if bplt:
            ax.plot(yz, yr, color='red')
            ax.set_ylim(0, max(yr))
            funct.persFig([ax], xlab='Z_eh [um]', ylab='R [um]')
            ax.set_title(obj.name)

        return yr, yz

    @staticmethod
    def cylinder(obj: surface.Surface, radius, phiCone=None, alphaZ=0, concavity='convex', base=False, bplt=False):
        """
        Evaluates radius and form deviation of a cylinder by fitting a least square cylinder
        to the points and applying cuts to the surface to avoid edge points.

        Parameters
        ----------
        obj : surface.Surface
            The surface on which the processing is applied
        radius : float
            The nominal radius of the cylinder
        phiCone : float
            Angle in degree of the FOV of the instrument
        alphaZ : float
            Rotation of the cylinder axis about the Y axis (radian)
        concavity : str
            Can be either 'convex' or 'concave'
        base : bool
            If true removes the points at the base of the cylinder
        bplt : bool
            Plots the sphere fitted to the data points

        Returns
        -------
        R_all : float
            The radius of the best fit cylinder to all points
        FD_all : float
            The form deviation of the best fit cylinder to all points
        R_2s : float
            The radius of the best fit cylinder to only the points with residue < 2 * sigma
        FD_2s : float
            The form deviation of the best fit cylinder to only the points with residue < 2 * sigma
            
        Notes
        -----
        At first, the points of the cylinder measured by the optical profilometer are fitted with the 
        least squares algorithm Surfile.geometry.cylinder(); this fit is used to calculate the equation 
        of the cylinder and the residuals of the measured points compared to the calculated points.
        Based on the results obtained, the measured points are eliminated where the following inequality 
        is satisfied, in order to cut out any outlier point far from the cylinder axis:

        $|y-(\\frac{m}{l})x-y_0|>\\frac{2Rsin \\cdot (\\phi)}{cos(\\alpha_z)}$
        where $m$ and $l$ are the components of the versor that represent the cylinder axis direction,
        R is the estimated radius of the circular base of the cylinder, $\\alpha_z$ is the rotation of the 
        cylinder axis around the Z-axis, $\\phi$ is the cone angle of the objective specified by the instrument
        manufacturer, $y_0$  the intercept of the axis.
        A second cylinder fit is then applied to all remaining points and $R_{all}$ and $FD_{all}$ parameters are 
        calculated, a third cylinder fit is done on the points that in the initial fit had a radial residue 
        below $2\\sigma$ and $R_{2\\sigma}$ and $FD_{2\\sigma}$ can be finally calculated.
        """
        # TODO: if alphaZ = 90 deg the method is really slow (whyyy??)
        def fitCyl():
            _p = geometry.Cylinder.formFit(obj, radius, alphaZ=alphaZ, concavity=concavity,
                                         base=base, finalize=False, bplt=bplt)
            _r = np.abs(_p[0])

            _l = np.cos(_p[3]) * np.cos(_p[4])
            _m = np.sin(_p[3])
            _n = np.cos(_p[3]) * np.sin(_p[4])

            return _r, _p, _l, _m, _n

        def calcResid():
            # calculation of radial distance from fitted cylinder axis
            d = -(l * obj.X + m * obj.Y + n * obj.Z)
            t = -(d + m * est_p[1] + n * est_p[2])

            H = np.array([l * t, est_p[1] + m * t, est_p[2] + n * t])
            P = np.array([obj.X, obj.Y, obj.Z])

            dist = np.linalg.norm(P - H, axis=0)

            # calculation of radial residues as dist - fitR
            _resid = dist - R

            return _resid, np.mean(_resid[~np.isnan(_resid)]), np.std(_resid[~np.isnan(_resid)])

        def findPhiCone():
            masknan = np.isfinite(obj.Z)
            a = m / l
            b = -1
            c = est_p[1]

            dists = np.abs(a * obj.X[masknan] + b * obj.Y[masknan] + c) / (a ** 2 + b ** 2)
            maxd = np.max(dists)
            return np.rad2deg(np.arcsin(maxd / R))

        # fit the first approx cyl
        R, est_p, l, m, n = fitCyl()
        resid, avg, std = calcResid()

        below_i = (np.abs(resid) < 2 * std)  # points with residues below 2sigma
        outliers_i = (np.abs(resid) > 10 * std)
        obj.Z[outliers_i] = np.nan  # remove the evident outliers

        if phiCone is not None:  # remove points outside cone from topo
            if phiCone is True:
                phiCone = findPhiCone()
                print(f'Using {phiCone=}')

            base = R * np.sin(np.deg2rad(phiCone))
            base = base if -np.pi/2 < est_p[3] < np.pi/2 else -base  # invert polarity for alphaZ >< +-90°
            # keep only values inside the range +- base centered on the cylinder axis
            discard_i = np.abs(obj.Y - (m / l) * obj.X - est_p[1]) > (2 * base) / np.cos(est_p[3])
            obj.Z[discard_i] = np.nan

            if bplt: obj.pltC()

        # fit the all cyl
        R_all, est_p, l, m, n = fitCyl()
        resid_all, avg_all, std_all = calcResid()
        FD_all = np.ptp(resid_all[~np.isnan(resid_all)])

        # fit the 2sigma cyl
        obj.Z[~below_i] = np.nan
        if bplt: obj.pltC()
        R_2s, est_p, l, m, n = fitCyl()
        resid_2s, avg_2s, std_2s = calcResid()
        FD_2s = np.ptp(resid_2s[~np.isnan(resid_2s)])

        return R_all, FD_all, R_2s, FD_2s, avg_all, std_all, avg_2s, std_2s

    @staticmethod
    def grating_1d(obj: surface.Surface, nom_pitch, direction='x', bplt=False):
        """
        Determines height and pitch of 1D gratings
        of line bars of rectangular cross-section, where the trench width
        is equal to the bar width.
        The bar cross-sections are modelled as rectangular boxes with
        smoothed corners using a combination of sigmoidal functions.
        See ProfileAnalysis.grating_1d() for details on the sigmoidal fit.
        
        Parameters
        ----------
        obj: surface.Surface
            The "surface" on which the analysis is carried out
        nom_pitch: float
            The nominal pitch of the grating
        direction: str
            Orientation of the features (perpendicular to the grating bars)
        bplt: bool
            If true plots the calculated heights and regression lines
            
        Returns
        ----------
        hs: float
            Calculated mean height
        pitch: float
            Calculated mean pitch
        s_pitch: float
            standard error of the pitch
        """
        xs = None
        hs = None
        ys = obj.y
        profiles = obj.toProfiles(axis=direction).tolist()
        with alive_bar(len(profiles), force_tty=True,
                       title='lateral', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for i, p in enumerate(profiles):
                if i == 0:
                    xs, hs = ProfileAnalysis.grating_1d(p, nom_pitch=nom_pitch, bplt=bplt)
                else:
                    x_p, h_p = ProfileAnalysis.grating_1d(p, nom_pitch=nom_pitch, bplt=False)
                    xs = np.vstack((xs, x_p))
                    hs = np.vstack((hs, h_p))
                bar()
        ms = []
        qs = []
        for c in xs.T:  # fit the regression lines
            yx = np.vstack([obj.y, c.T]).T
            (rows, cols) = yx.shape
            G = np.ones((rows, 2))
            G[:, 0] = yx[:, 0]  # X
            Z = yx[:, 1]
            (m, q), _, _, _ = np.linalg.lstsq(G, Z, rcond=None)
            ms = np.append(ms, m)
            qs = np.append(qs, q)
        gamma = np.mean(np.arctan(ms))
        x_posnom = np.arange(0, len(qs)) * nom_pitch / np.cos(gamma)
        p_line, cov_p = np.polyfit(x_posnom, qs, 1)
        pitch = p_line[0] * nom_pitch
        s_pitch = np.sqrt(cov_p[0][0] * nom_pitch**2 + cov_p[1][1])

        if bplt:
            Xs, Ys = np.meshgrid(xs[0], ys)
            fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
            mcm = copy.copy(cm.Greys)
            mcm.set_bad(color='r', alpha=1.)
            mask_h = np.ma.array(hs, mask=np.isnan(hs))
            Min = np.mean(mask_h) - 2 * np.std(mask_h)
            Max = np.mean(mask_h) + 2 * np.std(mask_h)
            p = ax.pcolormesh(xs.T, Ys.T, hs.T, vmin=Min, vmax=Max, cmap=mcm)
            fig.colorbar(p, ax=ax)
            for i, c in enumerate(xs.T):
                bx.plot(c, obj.y, 'r')
                bx.plot(ms[i] * obj.y + qs[i], obj.y, alpha=0.5)
                bx.pcolormesh(obj.X, obj.Y, obj.Z, alpha=0.2)
            
            funct.persFig([ax, bx], xlab='x [um]', ylab='y [um]')
            ax.set_title(obj.name)
        
        return np.mean(hs), np.mean(pitch), s_pitch
    

class TipCorrection():
    """
    Class for tip correction utilities
    
    Notes
    -----
    <span style="color:orange">This class will be moved to a utility module in the future
    use with caution !!!</span>.
    """
    @staticmethod
    def erosion(obj: profile.Profile, radius, bplt=False):
        """
        Tip correction method implementation,
        rolls a disk  of radius R (in mm) along the original profile

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the filter is applied
        radius: float
            The radius of the sphere of the contact instrument
        bplt: bool
            Plots the envelope of the filter if true
        """
        def morph(profile_x, profile_y, radius):
            spacing = profile_x[1] - profile_x[0]
            n_radius = int(radius / spacing)
            n_samples = len(profile_x)

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
                    beta = profile_out[i]

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

            if bplt:
                fig, ax = plt.subplots()
                ax.axis('equal')
                ax.plot(profile_x, profile_y, profile_x, profile_out)
                plt.show()

            return profile_out

        obj.Z = morph(obj.X, obj.Z, radius)

    @staticmethod
    def naive(obj: profile.Profile, radius, bplt=False):
        """
        Tip correction implementation,
        rolls a disk  of radius R (in mm) along the original profile
        Uses the naive approach described in:
        "Algorithms for morph profile filters and their comparison"
        Shan Lou, Xiangqian Jiang, Paul J. Scott. (2012)

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the filter is applied
        radius: float
            The radius of the sphere of the contact instrument
        bplt: bool
            Plots the envelope of the filter if true
        """
        spacing = obj.X[1] - obj.X[0]
        n_radius = int(radius // spacing)
        
        x = np.linspace(-radius, radius, n_radius * 2)
        struct = -np.sqrt(radius ** 2 - x ** 2)
        
        l = struct.size // 2  # half lenght for cut
        
        def findMax(i):
            max = np.min(obj.Z[i-l : i+l] + struct)
            return max        

        filtered = profile.Profile()
        filtered.setValues(
            obj.X[l: -l],
            np.array([findMax(i) for i in range(l, obj.Z.size - l, 1)]),
            bplt=False
        )
        
        if bplt:
                fig, ax = plt.subplots()
                ax.axis('equal')
                ax.plot(obj.X, obj.Z, filtered.X, filtered.Z)
                plt.show()