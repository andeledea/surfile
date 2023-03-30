from scipy import ndimage, sparse
from abc import ABC, abstractmethod
import numpy as np

from surfile import profile, surface, funct
from functools import lru_cache

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


class Filter(ABC):
    @staticmethod
    @abstractmethod
    def filter(obj, cutoff):  # applies the filter without creting a filter obj
        pass

    @staticmethod
    def plotEnvelope(X, Z, envelope):
        fig, ax = plt.subplots()
        ax.plot(X, Z, X, envelope)

        plt.show()


class ProfileGaussian(Filter):
    @staticmethod
    def filter(obj: profile.Profile(), cutoff, bplt=False):
        """
        Applies to a profile object a gaussian filter ISO 16610-21.
        The resulting profile is cut at the borders to avoid border effects.
        Parameters
        ----------
        obj : profile.Profile()
            The profile object on wich the filter is applied
        cutoff: float
            The cutoff of the gaussian filter
        bplt: bool
            Plots the envelope of the filter if true
        """
        order = 0
        nsample_cutoff = cutoff / (np.max(obj.X) / np.size(obj.X))
        ncutoffs = np.floor(np.max(obj.X) / cutoff)
        nsample_region = nsample_cutoff * ncutoffs

        border = round((np.size(obj.Z) - nsample_region) / 2)

        alpha = np.sqrt(np.log(2) / np.pi)
        sigma = nsample_cutoff * (alpha / np.sqrt(2 * np.pi))
        envelope = ndimage.gaussian_filter1d(obj.Z, sigma=sigma, order=order)
        filtered = obj.Z - envelope

        obj.X = obj.X[border:-border]
        obj.Z = filtered[border:-border]

        if bplt:
            Filter.plotEnvelope(obj.X0, obj.Z0, envelope)


class ProfileSpline(Filter):
    @staticmethod
    def filter(obj, cutoff, bplt=False):
        """
        Applies to a profile object a gaussian filter ISO 16610-22.
        The resulting profile is cut at the borders to avoid border effects.
        Parameters
        ----------
        obj : profile.Profile()
            The profile object on wich the filter is applied
        cutoff: float
            The cutoff of the spline filter
        bplt: bool
            Plots the envelope of the filter if true
        """
        n = obj.Z.size
        deltaX = np.max(obj.X) / np.size(obj.X)
        iden = np.eye(n)

        beta = .5  # tension patameter 0 < beta < 1
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


class ProfileRobust(Filter):
    @staticmethod
    def filter(obj: profile.Profile(), cutoff, bplt=False):
        n = obj.Z.size
        deltaX = np.max(obj.X) / np.size(obj.X)

        w = []
        xk = np.ones((n, 3))
        sk = np.zeros(n)
        for k in range(n):
            for l in range(n):
                xlk = (l - k) * deltaX
                xk[l] = [1, xlk, xlk**2]

                gmco = 0.7309 * cutoff
                deltal =
                slk = 1 / gmco * np.exp(-np.pi * (xlk / gmco)**2)
                sk[l][l] = slk * deltal

            wk =