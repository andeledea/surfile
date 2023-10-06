"""
'surfile.cutter'
- Cutting operations for profiles and surfaces

@author: Andrea Giura
"""

from abc import ABC, abstractmethod
import numpy as np

from surfile import profile, surface, funct

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
        self.extents, _ = SurfaceCutter.cut(sur)

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

        ax.pcolormesh(obj.X, obj.Y, obj.Z, cmap=cm.rainbow)
        ax.set_title('Choose cut region')
        fig.canvas.mpl_connect('close_event', onClose)

        plt.show()
        return RS.extents, cuts
