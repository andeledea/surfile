"""
'surfile.morph'
- analysis of morphological features for:
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

from surfile import profile, surface, funct, extractor, remover
from surfile.funct import classOptions, options, rcs


@dataclass
class Roi:
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
class ProfileMorph:
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
    def wireParams(obj: profile.Profile, bplt=False):
        """
        Given a nanowire cross section calculates max value
        height of the wire, width of the wire

        Parameters
        ----------
        obj : profile.Profile
            The nanowire cross section
        bplt : bool
            If true plots the profile and the section used for
            the height and width calculation

        Returns
        -------
        height : float
            The heigth of the nanowire
        width : float
            The width of the nanowire
        max : float
            The maximum value of the cross section
        max_i : int
            The index of the max
        """
        bound = np.min(obj.Z) + np.ptp(obj.Z) / 2

        top = np.max(obj.Z)
        bot = np.mean(obj.Z[obj.Z < bound])

        height = top - bot
        bound = bot + 0.1 * height
        width = obj.X[obj.Z > bound][-1] - obj.X[obj.Z > bound][0]

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(obj.X, obj.Z)
            ax.plot(obj.X[obj.Z > bound], obj.Z[obj.Z > bound], 'r')

            ax.set_title(obj.name)

        return height, width, np.max(obj.Z), np.argmax(obj.Z)

    @staticmethod
    def histHeight(obj: profile.Profile, bins=None, bplt=False):
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
            ax.set_title(obj.name)
        return r, z

    @staticmethod
    def lateral(obj: profile.Profile, nom_pitch, bplt=False):
        """
        Evaluate RSM - N parameters with the sigmoid fit method
        @dorothee_hueser

        Parameters
        ----------
        obj: The profile on which the steps are evaluated
        nom_pitch: the nominal pitch of the sample
        bplt: if true 

        Returns
        ----------
        x_c: np.array
            The center position of the features
        h_c: np.array
            The calculated heights of the features
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
class SurfaceMorph:
    @staticmethod
    def histHeight(obj: surface.Surface, bins=None, bplt=False):
        """
        Histogram method implementation

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the height is calculated
        bins: int
            The number of bins of the histogram, if set to none the program will calculate
            automatically the number of bins
        bplt: bool
            Plots the histogram of the profile

        Returns
        ----------
        height: float
            The calculated height of the surface
        (hist, edges)
            The histogram x and y
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
    def wireParams(obj: surface.Surface, bplt=False):
        """
        Given a nanowire topography extracts all the cross sections
        and calculates the parameters explained in ProfileMorph.wireParams

        Parameters
        ----------
        obj : surface.Surface
            the nanowire topography
        bplt : bool
            if true plots the results of the processing
        """
        prfs = obj.toProfiles('x')

        hs = []
        ws = []
        ts = []

        mZ = np.ma.masked_array(obj.Z)

        for i, p in enumerate(prfs):
            h, w, t, ti = ProfileMorph.wireParams(p, bplt=False)

            mZ[i, ti] = np.ma.masked

            hs.append(h)
            ws.append(w)
            ts.append(t)

        fig1, (ax, bx) = plt.subplots(nrows=1, ncols=2)
        ax.pcolormesh(obj.X, obj.Y, mZ, cmap=cm.viridis)  # hot, viridis, rainbow
        bx.plot(obj.y, ts, label='top_profile')

        fig2, (ax, bx) = plt.subplots(nrows=1, ncols=2)
        ax.plot(obj.y, hs, label='height')
        bx.plot(obj.y, ws, label='width')

        ax.legend()
        bx.legend()

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
                ms1, ms2 = ProfileMorph.arcSlope(slopeprofile, R)  # 350 um radius
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
                radiusprofile = copy.copy(extractor.SphereExtractor.sphereProfile(obj, startP=start, bplt=False))
                r, z = ProfileMorph.arcRadius(radiusprofile, bplt=False)  # 350 um radius
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
        to the points

        Parameters
        ----------
        obj: surface.Surface
            The surface on which the processing is applied
        radius: float
            The nominal radius of the cylinder
        phiCone: float
            Angle in degree of the FOV of the instrument
        alphaZ: float
            Rotation of the cylinder axis about the Y axis (radian)
        concavity: str
            Can be either 'convex' or 'concave'
        base: bool
            If true removes the points at the base of the cylinder
        bplt: bool
            Plots the sphere fitted to the data points

        Returns
        -------
        R_all: float
            The radius of the best fit cylinder to all points
        FD_all: float
            The form deviation of the best fit cylinder to all points
        R_2s: float
            The radius of the best fit cylinder to only the points with residue < 2 * sigma
        FD_2s: float
            The form deviation of the best fit cylinder to only the points with residue < 2 * sigma
        """
        # TODO: if alphaZ = 90 deg the method is really slow (whyyy??)
        def fitCyl():
            _p = remover.Cylinder.remove(obj, radius, alphaZ=alphaZ, concavity=concavity,
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
    def lateral(obj: surface.Surface, nom_pitch, direction='x', bplt=False):
        """
        Performs the 1D sigmoid analysis for each profile in the topography
        
        Parameters
        ----------
        obj: The surface on which the analysis is carried out
        nom_pitch: The nominal pitch of the grating
        direction: Orientation of the features (perpendicular to the steps)
        bplt: If true plots the 

        Returns
        ----------
        hs: float
            Calculated mean height
        pitch: float
            Calculated mean pitch
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
                    xs, hs = ProfileMorph.lateral(p, nom_pitch=nom_pitch, bplt=bplt)
                else:
                    x_p, h_p = ProfileMorph.lateral(p, nom_pitch=nom_pitch, bplt=False)
                    xs = np.vstack((xs, x_p))
                    hs = np.vstack((hs, h_p))
                bar()

        ms, qs = [], []
        for c in xs.T:  # fit the regression lines
            yx = np.vstack([obj.y, c.T]).T
            (rows, cols) = yx.shape
            G = np.ones((rows, 2))
            G[:, 0] = yx[:, 0]  # X
            Z = yx[:, 1]
            (m, q), _, _, _ = np.linalg.lstsq(G, Z, rcond=None)
            ms.append(m)
            qs.append(q)

        pitch = np.ediff1d(qs)

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
        
        return np.mean(hs), np.mean(pitch)
