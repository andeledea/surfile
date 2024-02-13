"""
'surfile.filter'
- Applies a filter to a profile or a surface:
    - gaussian profile
    - gaussian surface
    - spline profile
    - gaussian robust profile

@author: Andrea Giura, Dorothee Hueser
"""

from alive_progress import alive_bar
from matplotlib import cm
from scipy import ndimage, sparse, special, signal
import numpy as np
import copy

from surfile import profile, surface, funct

import matplotlib.pyplot as plt


# ISO 16610:
# Part 21 - Profile Gaussian filter
#   Microroughness filtering (lambda S)
#   Separation of roughness and waviness profiles (lambda C)
#   Band-pass filtering
# Part 22 - Profile Spline filter
# Part 29 - Profile Spline wavelets
# Part 31 - Profile Robust Gaussian filter
# Part 41 - Profile Morphological filter
# Part 45 - Profile Segmentation filter
# Part 49 - Profile Scale space technique
# Part 61 - Areal Gaussian filter
#   Microroughness S-Filter
#   L-Filter for the generation of the roughness S-L surface
# Part 62 - Areal Spline filter
# Part 71 - Areal Robust regression Gaussian filter
#   Microroughness S-Filter on stratified and structured surfaces
#   L-Filter for the generation of the roughness S-L surface on stratified and structured surfaces
#   F-Filter for the generation of S-F surface
#   Outlier detection
# Part 81 - Areal Morphological filter
#   F-Filter used to flatten a surface with the upper or lower envelope
#   Tip deconvolution of AFM instrument
# Part 85 - Areal Segmentation filter
#   Identification of structures (grains, pores, cells, ...)
#   Automatic leveling of MEMS
# Part 89 - Areal Scale space technique


class Filter:
    cutoff: float

    def applyFilter(self, obj, bplt=False):
        pass

    @staticmethod
    def plotEnvelope(X, Z, envelope):
        fig, ax = plt.subplots()
        ax.plot(X, Z, X, envelope)
        ax.set_ylim(min(Z), max(Z))
        
        funct.persFig([ax], 'x[um]', 'z[um]')

        plt.show()

    @staticmethod
    def plot3DEnvelope(X, Y, Z, envelope):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_surface(X, Y, Z, cmap=cm.Greys, alpha=0.4)
        ax.plot_surface(X, Y, envelope, cmap=cm.rainbow)

        plt.show()


class ProfileGaussian(Filter):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def applyFilter(self, obj: profile.Profile, bplt=False):
        self.filter(obj, cutoff=self.cutoff, bplt=bplt)

    @staticmethod
    def filter(obj: profile.Profile, cutoff, bplt=False):
        """
        Applies to a profile object a gaussian filter ISO 16610-21.
        The resulting profile is cut at the borders to avoid border effects.

        Parameters
        ----------
        obj : profile.Profile
            The profile object on wich the filter is applied
        cutoff: float
            The cutoff of the gaussian filter
        bplt: bool
            Plots the envelope of the filter if true
        """
        order = 0
        nsample_cutoff = cutoff / (np.nanmax(obj.X) / np.size(obj.X))
        ncutoffs = np.floor(np.nanmax(obj.X) / cutoff)
        nsample_region = nsample_cutoff * ncutoffs

        border = round((np.size(obj.Z) - nsample_region) / 2)

        alpha = np.sqrt(np.log(2) / np.pi)
        sigma = nsample_cutoff * (alpha / np.sqrt(2 * np.pi))
        envelope = copy.copy(obj.Z)
        envelope[~np.isnan(obj.Z)] = ndimage.gaussian_filter1d(obj.Z[~np.isnan(obj.Z)], 
                                                               sigma=sigma, order=order)
        filtered = obj.Z - envelope

        if bplt:
            Filter.plotEnvelope(obj.X, obj.Z, envelope)
            
        obj.X = obj.X[border:-border]
        obj.Z = filtered[border:-border]


class ProfileSpline(Filter):
    def __init__(self, cutoff, beta):
        self.cutoff = cutoff
        self.beta = beta  # tension parameter

    def applyFilter(self, obj: profile.Profile, bplt=False):
        self.filter(obj, cutoff=self.cutoff, beta=self.beta, bplt=bplt)

    @staticmethod
    def filter(obj: profile.Profile, cutoff, beta=0.5, bplt=False):
        """
        Applies to a profile object a gaussian filter ISO 16610-22.
        The resulting profile is cut at the borders to avoid border effects.

        Parameters
        ----------
        obj : profile.Profile
            The profile object on wich the filter is applied
        cutoff: float
            The cutoff of the spline filter
        beta: float
            The tension parameter of the filter (0 < beta < 1)
        bplt: bool
            Plots the envelope of the filter if true
        """
        if not 0 < beta < 1:
            raise Exception('Invalid tension value beta must be in the range [0 1]')

        n = obj.Z.size
        deltaX = np.max(obj.X) / np.size(obj.X)
        alpha = 1 / (2 * np.sin(np.pi * deltaX / cutoff))

        M = np.zeros((n, n))
        for i in range(n):  # create diagonal
            if i in [0, n-1]:
                P = 1
                Q = 1
                M[i][i] = 1 + beta * alpha ** 2 * P + (1 - beta) * alpha ** 4 * Q
            else:
                P = 2
                Q = 5 if i in [1, n-2] else 6
                M[i][i] = 1 + beta * alpha ** 2 * P + (1 - beta) * alpha ** 4 * Q

            if i < n-1:  # crete second diagonal
                P = -1
                Q = -2 if i in [0, n-2] else -4
                M[i][i + 1] = beta * alpha ** 2 * P + (1 - beta) * alpha ** 4 * Q
                M[i + 1][i] = beta * alpha ** 2 * P + (1 - beta) * alpha ** 4 * Q

            if i < n-2:  # create third diagonal
                P = 0
                Q = 1
                M[i][i + 2] = beta * alpha ** 2 * P + (1 - beta) * alpha ** 4 * Q
                M[i + 2][i] = beta * alpha ** 2 * P + (1 - beta) * alpha ** 4 * Q

        envelope = sparse.linalg.spsolve(sparse.csc_matrix(M), obj.Z)
        filtered = obj.Z - envelope
        obj.Z = filtered

        if bplt:
            Filter.plotEnvelope(obj.X0, obj.Z0, envelope)


class ProfileRegression(Filter):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def applyFilter(self, obj: profile.Profile, bplt=False):
        self.filter(obj, cutoff=self.cutoff, bplt=bplt)

    # TODO: this is still too slow even with sparse calculations
    @staticmethod
    def filter_matrix(obj: profile.Profile, cutoff, bplt=False):
        z = obj.Z
        n = obj.Z.size
        deltaX = np.max(obj.X) / np.size(obj.X)

        w = np.zeros(z.shape)
        xk = np.zeros((n, 3))
        sk = np.zeros((n, n))
        with alive_bar(n, force_tty=True,
                       title='Slope', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for k in range(n):
                for l in range(n):
                    xlk = (l - k) * deltaX
                    xk[l] = [1, xlk, xlk**2]

                    gmco = 0.7309 * cutoff

                    c = 4.4478 * np.median(np.abs(z - w))
                    deltal = (1 - ((z[l] - w[l]) / c)**2)**2 if np.abs(z[l] - w[l]) <= c else 0
                    slk = 1 / gmco * np.exp(-np.pi * (xlk / gmco)**2)
                    sk[l][l] = slk * deltal

                M = xk.T @ sk @ xk

                T = np.matmul([1, 0, 0], M)
                T = np.matmul(T, xk.T)
                T = np.matmul(T, sk)
                wk = np.matmul(T, z)
                w[k] = wk
                bar()
        filtered = obj.Z - w
        obj.Z = filtered

        if bplt:
            Filter.plotEnvelope(obj.X0, obj.Z0, w)

    @staticmethod
    def filter_regression_p1(obj: profile.Profile, cutoff, bplt=False):
    def filter_regression_p1(obj: profile.Profile, cutoff, bplt=False):
        """
        Applies to a profile object a gaussian regression with a degree 1 poly.

        Parameters
        ----------
        obj : profile.Profile
            The profile object on wich the filter is applied
        cutoff: float
            The cutoff of the gaussian filter
        bplt: bool
            Plots the envelope of the filter if true
        The use of this function requires the following citacion:
            Seewig, Linear and robust Gaussian regression filters,
            2005 J. Phys.: Conf. Ser. 13 254, doi:10.1088/1742-6596/13/1/059
        """
        x = obj.X
        deltaX = np.max(obj.X) / np.size(obj.X)
        n = obj.Z.size

        alpha = np.sqrt(np.log(2) / np.pi)  # filter constant

        nf = 2 * int(2 ** (round(np.log2(n))))
        xf = np.zeros(nf)
        xf[0:n] = x
        xf[nf - n + 1:nf] = -np.flip(x[1:n], axis=0)  # mirror the x-axis
        z = np.zeros(nf)
        z[0:n] = obj.Z
        zf = np.fft.fft(z)

        sgauss = np.exp(-np.pi * np.square(xf) / (alpha * cutoff) ** 2) / (alpha * cutoff)
        Fsgauss = np.fft.fft(sgauss)
        sgaussx = np.multiply(xf, sgauss)
        Fsgaussx = np.fft.fft(sgaussx)
        # spR1 = np.zeros((1, nf), dtype=np.complex128)
        # spR2 = np.zeros((1, nf), dtype=np.complex128)
        spR1 = np.multiply(zf, Fsgauss)
        spR2 = np.multiply(zf, Fsgaussx)
        R1 = np.fft.ifft(spR1) * deltaX
        R2 = np.fft.ifft(spR2) * deltaX

        nh = n // 2
        x3 = np.ones(n) * np.float64(nh) * deltaX
        x3[0:nh] = x[0:nh]
        xhlp = np.flip(x[1:1 + n - nh], axis=0)
        x3[nh:n] = xhlp
        sgauss = np.exp(-np.pi * np.square(x3) / (alpha * cutoff) ** 2) / (alpha * cutoff)

        mu0 = 0.5 * (special.erf(np.sqrt(np.pi) * x3 / (alpha * cutoff)) + 1)
        mu1 = -((cutoff * alpha) / (2.0 * np.pi)) * sgauss
        mu2 = np.add(mu0 * ((cutoff * alpha) ** 2) / (2.0 * np.pi), np.multiply(x3, mu1))

        d = np.subtract(np.multiply(mu2, mu0), np.square(mu1))
        w = np.zeros(n)

        b0 = mu2[0:nh] / d[0:nh]
        b1 = -mu1[0:nh] / d[0:nh]
        w[0:nh] = b0 * np.real(R1[0:nh]) + b1 * np.real(R2[0:nh])
        b0 = mu2[nh:n] / d[nh:n]
        b1 = mu1[nh:n] / d[nh:n]
        w[nh:n] = b0 * np.real(R1[nh:n]) + b1 * np.real(R2[nh:n])

        envelope = w

        filtered = obj.Z - envelope
        obj.Z = filtered

        if bplt:
            Filter.plotEnvelope(obj.X0, obj.Z0, envelope)


class TipCorrection(Filter):
    @staticmethod
    def filter(obj: profile.Profile, radius, bplt=False):
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
    

class SurfaceGaussian(Filter):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def applyFilter(self, obj: surface.Surface, bplt=False):
        self.filter(obj, cutoff=self.cutoff, bplt=bplt)

    @staticmethod
    def filter(obj: surface.Surface, cutoff, bplt=False):
        """
        Applies to a surface object a gaussian filter ISO 16610-21.
        Applies to a surface object a gaussian filter ISO 16610-21.
        The resulting profile is cut at the borders to avoid border effects.
        TODO !!!

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the filter is applied
        cutoff: float
            The cutoff of the gaussian filter
        bplt: bool
            Plots the envelope of the filter if true
        """
        deltaX = np.max(obj.x) / np.size(obj.x)
        deltaY = np.max(obj.y) / np.size(obj.y)
        #
        xconv = np.arange(-cutoff, cutoff, deltaX)
        yconv = np.arange(-cutoff, cutoff, deltaY)
        ml = len(xconv)
        nl = len(yconv)
        alpha = np.sqrt(np.log(2) / np.pi)
        gk = np.zeros((nl, ml))
        for iy in range(0, nl):
            for ix in range(0, ml):
                gk[iy][ix] = np.exp(-np.pi * (xconv[ix] ** 2 + yconv[iy] ** 2) / (alpha * cutoff) ** 2) / (alpha * cutoff)
        gk = gk / np.sum(gk)
        envelope = signal.convolve2d(obj.Z, gk, 'same')
        obj.Z = obj.Z - envelope

        # TODO: very hard to see if this works correctly from the topographies
        if bplt:
            Filter.plot3DEnvelope(obj.X0, obj.Y0, obj.Z0, envelope)
