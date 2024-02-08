"""
'surfile.cutter'
- Cutting operations for profiles and surfaces

Notes
-----
These utilities are implemented as classes in order to allow the creation 
of templates to apply the same processing to multiple images.
The class implementation also allows the creation of cutter and selector 
objects in other methods such as levelling or feature extraction routines.

@author: Andrea Giura
"""

from abc import ABC, abstractmethod
import numpy as np

from surfile import geometry, profile, surface, funct

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib import cm

import tkinter as tk
from tkinter import filedialog


class Cutter(ABC):
    """
    Class that provides methods for profile / surface cutting
    static methods are used to work directly on profiles / surfaces, cutter objects
    can be instantiated to apply the same cut to multiple profiles / surfaces
    """
    def __init__(self):
        self.extents = None

    @abstractmethod
    def templateExtents(self):
        """
        Asks the user to open a template profile / topography
        plots the template and asks where the user wants to cut.
        The edges are saved and used everytime the cutter is
        applied to a profile / topography obj, see applyCut()
        """
        pass

    @abstractmethod
    def applyCut(self, obj):
        """
        Applies the cut to the object passed using the extents
        defined previously by the user, see templateExtents()
        """
        if self.extents is None:
            raise Exception('Cut extents are not defined')
        pass


class ProfileCutter(Cutter, ABC):
    def templateExtents(self):
        root = tk.Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(
            parent=root,
            title='Choose template cut profile'
        )

        prf = profile.Profile()
        prf.openPrf(fname, bplt=False)
        self.extents, _ = ProfileCutter.cut(prf, finalize=False)

    def applyCut(self, obj: profile.Profile, finalize=True):
        """
        Applies the cut to the object passed using the extents
        defined previously by the user, see templateExtents()

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the cut is performed
        finalize: bool
            If False the cut is not finalized on the profile object
            the method returns the cut vectors

        Returns
        ----------
        cuts: tuple
            tuple of cut arrays x and z respectively
        """
        if self.extents is None:
            raise Exception('Cut extents are not defined')

        xmin, xmax = self.extents
        print(xmin, xmax)
        i_near = lambda arr, val: (np.abs(arr - val)).argmin()
        start_x, end_x = i_near(obj.X, xmin), i_near(obj.X, xmax)

        x_cut = obj.X[start_x: end_x]
        z_cut = obj.Z[start_x: end_x]
        if finalize:
            obj.X = x_cut
            obj.Z = z_cut

        return x_cut, z_cut

    @staticmethod
    def cut(obj: profile.Profile, finalize=True):
        """
        Cuts the profile at the margins defined manually by the user

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the cut is applied
        finalize: bool
            If set to False the cut will not alter the profile,
            the method will only return the extents chosen by the user

        Returns
        ----------
        extents: tuple
            The cut span selected left and right x values
        cuts: list
            The x and z arrays obtained after the cut
        """

        def onClose(event):
            xmin, xmax = span.extents
            i_near = lambda arr, val: (np.abs(arr - val)).argmin()
            start_x, end_x = i_near(obj.X, xmin), i_near(obj.X, xmax)

            if end_x < start_x: start_x, end_x = end_x, start_x

            x_cut = obj.X[start_x: end_x]
            z_cut = obj.Z[start_x: end_x]
            if finalize:
                obj.X = x_cut
                obj.Z = z_cut

            cuts.append(x_cut)
            cuts.append(z_cut)

        fig, ax = plt.subplots()
        span = SpanSelector(ax, lambda a, b: None,
                            direction='horizontal', useblit=True,
                            button=[1, 3],  # don't use middle button
                            interactive=True)

        cuts = []

        ax.plot(obj.X, obj.Z)
        ax.set_title('Choose region')
        fig.canvas.mpl_connect('close_event', onClose)
        plt.show()

        return span.extents, cuts
    
    def circleCut(obj: profile.Profile, startP, bplt=False):
        """
        Function to divide a circle profile starting from the
        maximum value to the 2 edges of the profile

        Parameters
        ----------
        obj : profile.Profile
            The profile to be divided in two parts
        startP : str
            The method used to find the maximum point
            - 'max': uses the maximum value of the profile
            - 'fit': uses the center coordinate calculated with a LS fit
            
        Returns
        -------
        prfl : profile.Profile
        prfr : profile.Profile
            The two extracted profiles

        Raises
        ------
        Exception
            If the startP parameter is not correct
        """
        prfl = profile.Profile()
        prfr = profile.Profile()
        
        if startP == 'max':
            # split the profile @ max value
            split_i = np.nanargmax(obj.Z)
            
        elif startP == 'fit':
            # split the profile at center of fit
            _, _, center = geometry.Circle.formFit(obj, finalize=False, bplt=False)
            split_i = np.nanargmin(np.abs(obj.X - center[0]))
            
        else: raise Exception(f'{startP} is not valid option for startP')
        
        xc = obj.X[split_i]
        print(f'Cutting @ {xc=}')
        
        x = +(obj.X[split_i:0:-1] - xc)
        z = obj.Z[split_i:0:-1]
        prfl.setValues(x, z, bplt=False)
        
        x = -(obj.X[split_i:] - xc)
        z = obj.Z[split_i:]
        prfr.setValues(x, z, bplt=False)
        
        if bplt:
            fig, ax = plt.subplots()
            ax.plot(obj.X, obj.Z)
            ax.plot(obj.X[split_i], obj.Z[split_i], 'or')
            funct.persFig([ax], xlab='x [um]', ylab='z [um]', gridcol='None')
            
            plt.show()
        
        return prfl, prfr
        
        
class SurfaceCutter(Cutter, ABC):
    def templateExtents(self):
        root = tk.Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(
            parent=root,
            title='Choose template cut surface'
        )

        sur = surface.Surface()
        sur.openFile(fname, bplt=False)
        self.extents, _ = SurfaceCutter.cut(sur, finalize=False)

    def applyCut(self, obj: surface.Surface, finalize=True):
        """
        Applies the cut to the object passed using the extents
        defined previously by the user, see templateExtents()

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the cut is performed
        finalize: bool
            If False the cut is not finalized on the profile object
            the method returns the cut vectors

        Returns
        ----------
        cuts: (np.array, np.array, np.array)
            Tuple of cut arrays x y and z respectively
        """
        if self.extents is None:
            raise Exception('Cut extents are not defined')
        xmin, xmax, ymin, ymax = self.extents

        i_near = lambda arr, val: (np.abs(arr - val)).argmin()  # find the closest index
        start_x, end_x = i_near(obj.x, xmin), i_near(obj.x, xmax)
        start_y, end_y = i_near(obj.y, ymin), i_near(obj.y, ymax)

        x_cut = obj.X[start_y: end_y, start_x: end_x]
        y_cut = obj.Y[start_y: end_y, start_x: end_x]
        z_cut = obj.Z[start_y: end_y, start_x: end_x]

        if finalize:
            obj.X = x_cut
            obj.Y = y_cut
            obj.Z = z_cut

            obj.x = obj.x[start_x: end_x]
            obj.y = obj.y[start_y: end_y]

        return x_cut, y_cut, z_cut

    @staticmethod
    def cut(obj: surface.Surface, finalize=True):
        """
        Cuts the surface with a rectangle drawn by the user

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the cut is applied
        finalize: bool
            If set to False the cut will not alter the profile,
            the method will only return the extents chosen by the user

        Returns
        ----------
        extents (xmin, xmax, ymin, ymax):  (float, ...)
            The cut borders
        cuts: list
            The x, y, z, arrays obtained after the cut
        """
        def onSelect(eclick, erelease):
            pass

        def onClose(event):
            xmin, xmax, ymin, ymax = RS.extents

            i_near = lambda arr, val: (np.abs(arr - val)).argmin()  # find the closest index
            start_x, end_x = i_near(obj.x, xmin), i_near(obj.x, xmax)
            start_y, end_y = i_near(obj.y, ymin), i_near(obj.y, ymax)

            x_cut = obj.X[start_y: end_y, start_x: end_x]
            y_cut = obj.Y[start_y: end_y, start_x: end_x]
            z_cut = obj.Z[start_y: end_y, start_x: end_x]

            if finalize:
                obj.X = x_cut
                obj.Y = y_cut
                obj.Z = z_cut

                obj.x = obj.x[start_x: end_x]
                obj.y = obj.y[start_y: end_y]

            cuts.append(x_cut)
            cuts.append(y_cut)
            cuts.append(z_cut)

        fig, ax = plt.subplots()
        # rectangle selector drawtype is deprecated
        RS = RectangleSelector(ax, onSelect,
                               useblit=True,
                               button=[1, 3],  # don't use middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)

        cuts = []

        ax.pcolormesh(obj.X, obj.Y, obj.Z, cmap=cm.viridis)
        ax.set_title('Choose cut region')
        fig.canvas.mpl_connect('close_event', onClose)

        plt.show()
        return RS.extents, cuts


class HistCutter(Cutter, ABC):
    @staticmethod
    def cut(obj, bins=None, finalize=True):
        """
        Cuts the surface on the Z axis keeping only the
        points with an height included in the selection

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the cut is applied
        bins : int
            The number of bins in the histogram
            if None the program calculates the optimal value
        finalize: bool
            If set to False the cut will not alter the profile,
            the method will only return the extents chosen by the user

        Returns
        ----------
        extents (zmin, zmax):  (float, ...)
            The cut values
        """
        b = bins
        if bins is None:
            # bw = 2 * stats.iqr(obj.Z[np.isfinite(obj.Z)]) / (obj.Z.size ** (1/3))  # Freedman-Diaconis
            b = int(np.sqrt(obj.Z.size))
            print(f'Using {b} bins in hist')

        hist, edges = np.histogram(obj.Z[np.isfinite(obj.Z)], bins=b)
        fig = plt.figure()
        ax_ht = fig.add_subplot(111)
        ax_ht.hist(edges[:-1], bins=edges, weights=hist / np.size(obj.Z) * 100, color='red')
        funct.persFig(
            [ax_ht],
            gridcol='grey',
            xlab='z [nm]',
            ylab='pixels %'
        )
        ax_ht.set_title('Choose cut region')

        def onClose(event):
            zmin, zmax = span.extents
            # i_near = lambda arr, val: (np.abs(arr - val)).argmin()
            # start_z, end_z = i_near(obj.Z, zmin), i_near(obj.Z, zmax)
            # print(start_z, end_z)

            if finalize:
                exclude = np.logical_or(obj.Z < zmin, obj.Z > zmax)
                obj.Z[exclude] = np.nan

        span = SpanSelector(ax_ht, lambda a, b: None,
                            direction='horizontal', useblit=True,
                            button=[1, 3],  # don't use middle button
                            interactive=True)

        fig.canvas.mpl_connect('close_event', onClose)
        plt.show()

        return span.extents
