"""
'surfile.extractor'
- Creates a profile from a surface provides:
    - SimpleExtractor: profile parallel to x or y direction
    - ComplexExtractor: profile can be any (even pieceWise defined)
    - SphereExtractor: profile starting from the maximum point of the surface
    
Notes
-----
Since the study of profiles is of great interest for studying (i) roughness 
parameters and (ii) height measurements of step or groove samples, profile 
extraction methodologies have been implemented within the program.

@author: Andrea Giura
"""

from abc import ABC, abstractmethod

import numpy as np
import math

from scipy import ndimage

from surfile import profile, surface, funct, cutter as cutr

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib import cm

import tkinter as tk
from tkinter import filedialog


class Extractor(ABC):
    """
    Class that provides methods for profile extraction from topographies
    static methods are used to work directly on surfaces, extractor objects
    can be instantiated to apply the same extraction to multiple topographies

    The provided extraction can be divided into:
    - simple extraction: the profile is parallel to x or y
    - complex extraction: the profile can be any, even peacewise defined
    - sphere extraction: the profile is along the x direction but starts from
                         the center of the spherical cup
    """

    def __init__(self):
        self.options = None  # {'x': 0, 'y': 0, 'dir': 'x', 'wid': width} for simple parallel extraction
        self.points = None  # used only in the complex estraction method

    @staticmethod
    def _openTopo():  # protected method
        root = tk.Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(
            parent=root,
            title='Choose template surface for extraction'
        )

        sur = surface.Surface()
        sur.openFile(fname, bplt=False)
        return sur

    @abstractmethod
    def template(self):
        """
        Asks the user to open a template topography
        plots the template and asks how to extract a profile.
        The parameters are saved and used everytime the extractor is
        applied to a topography obj, see apply()
        """
        self._openTopo()
        pass

    @abstractmethod
    def apply(self, obj: surface.Surface, bplt=False):
        """
        Applies the extraction to the object passed using the parameters
        defined previously by the user, see template()
        """
        pass


def _extractProfileWithOptions(obj: surface.Surface, opt, prf: profile.Profile = None, bplt=False):
    """
    Protected method
    Applies the class option to extract a profile from a topography

    Parameters
    ----------
    obj: surface.Surface
        The surface object on wich the profile is extracted
    opt: dict
        The position selected by the user and the direction of extraction
    prf: profile.Profile, optional
        The profile on wich the function saves the results, if None an empty profile
        is created
    bplt: bool
        If True plots the extracted profile

    Returns
    -------
    prf: profile.Profile
            The extracted profile
    """
    if prf is None:
        prf = profile.Profile()  # if no profile is passed work on a new object
    choice = opt['dir']
    width = opt['wid']
    if width == 1:
        if choice == 'x':  # the user chooses the direction of extraction
            prf.setValues(obj.x, obj.Z[opt['y'], :], bplt=bplt)
        elif choice == 'y':
            prf.setValues(obj.y, obj.Z[:, opt['x']], bplt=bplt)
        else:
            raise Exception(f'Direction {choice} does not exist')
    else:  # avg between adjacent profiles
        if choice == 'x':  # the user chooses the direction of extraction
            prf.setValues(obj.x, np.mean(obj.Z[opt['y'] - width: opt['y'] + width, :], axis=0), bplt=bplt)
        elif choice == 'y':
            prf.setValues(obj.y, np.mean(obj.Z[:, opt['x'] - width: opt['x'] + width], axis=1), bplt=bplt)
        else:
            raise Exception(f'Direction {choice} does not exist')

    return prf


class SimpleExtractor(Extractor, ABC):
    def template(self):
        sur = Extractor._openTopo()
        _, self.options = SimpleExtractor.profile(sur,
                                                  direction=input('Select profile template direction: '),
                                                  width=int(input('Select profile template width: ')),
                                                  bplt=False)

    def apply(self, obj: surface.Surface, bplt=False):
        prf = _extractProfileWithOptions(obj, self.options, bplt=bplt)
        return prf

    @staticmethod
    def profile(obj: surface.Surface, direction='x', width=1, bplt=False):
        """
        Extracts a profile along x or y at the position indicated by the user

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the profile is extracted
        direction: str
            The direction of the profile mean 'x' or 'y'
        width: int
            The lateral number of values averged when extracting the profile
        bplt: bool
            If True plots the extracted profile

        Returns
        ----------
        profile: prf.Profile
            Profile object exctracted from the topography
        options: dict
            The position selected by the user and the direction of extraction
        """
        if width < 1: raise Exception('Width must be > 0')
        option = {'x': 0, 'y': 0, 'dir': direction, 'wid': width}
        prf = profile.Profile()

        def pointPick(point, mouseevent):  # called when pick event on fig
            if mouseevent.xdata is None:
                return False, dict()
            xdata = mouseevent.xdata
            ydata = mouseevent.ydata

            xind = np.where(obj.X[0, :] <= xdata)[0][-1]
            yind = np.where(obj.Y[:, 0] <= ydata)[0][-1]
            print(f'Chosen point {xind}, {yind}')  # print the point the user chose

            option['x'] = xind
            option['y'] = yind
            return True, dict(pickx=xind, picky=yind)

        def onClose(event):  # called when fig is closed
            _extractProfileWithOptions(obj, option, prf, bplt=bplt)
            plt.close(fig)

        fig, ax = plt.subplots()  # create the fig for profile selection
        ax.pcolormesh(obj.X, obj.Y, obj.Z, cmap=cm.rainbow, picker=pointPick)
        ax.set_title('Extract profile')
        fig.canvas.mpl_connect('close_event', onClose)

        plt.show()
        return prf, option

    @staticmethod
    def meanProfile(obj: surface.Surface, direction='x', cutter=None, bplt=False):
        """
        Extracts the mean profile along x or y

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the profile is extracted
        direction: str
            The direction of the profile mean 'x' or 'y'
        cutter: cutter.Cutter
            -if not set, the fit uses all points
            -if true allows the user to select manually the region of interest
            -if a cutter obj is passed the fit is done only on the cutted profile points
             and then applied on the whole profile
        bplt: bool
            If True plots the extracted profile

        Returns
        ----------
        prf: profile.Profile
            The mean profile
        """
        if cutter is True:
            _, (x, y, z) = cutr.SurfaceCutter.cut(obj, finalize=False)
        elif cutter is not None:
            x, y, z = cutter.applyCut(obj, finalize=False)
        else:
            x, y, z = obj.X, obj.Y, obj.Z

        prf = profile.Profile()
        if direction == 'x':
            prf.setValues(x[0, :], np.mean(z, axis=0), bplt=bplt)

        if direction == 'y':
            prf.setValues(y[:, 0], np.mean(z, axis=1), bplt=bplt)

        return prf


def _extractProfileWithPoints(obj: surface.Surface, points, prf: profile.Profile = None, bplt=False):
    """
    Protected method
    Applyes the class points to extract a profile from a topography

    Parameters
    ----------
    obj: surface.Surface
        The surface object on wich the profile is extracted
    points: dict
        The positions selected by the user
    prf: profile.Profile, optional
        The profile on wich the function saves the results, if None an empty profile
        is created
    bplt: bool
        If True plots the extracted profile

    Returns
    -------
    prf: profile.Profile
            The extracted profile
    """
    def connect(ends):
        d0, d1 = np.abs(np.diff(ends, axis=0))[0]
        if d0 > d1:
            return np.c_[np.linspace(ends[0, 0], ends[1, 0], d0 + 1, dtype=np.int32),
                         np.round(np.linspace(ends[0, 1], ends[1, 1], d0 + 1)).astype(np.int32)]
        else:
            return np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1 + 1)).astype(np.int32),
                         np.linspace(ends[0, 1], ends[1, 1], d1 + 1, dtype=np.int32)]

    if prf is None:
        prf = profile.Profile()  # if no profile is passed work on a new object

    ind = []
    poi = []

    tot_dist = 0
    tot_n = 0
    Z = np.array([])

    for (a, b) in points:
        xind = np.where(obj.X[0, :] <= a)[0][-1]
        yind = np.where(obj.Y[:, 0] <= b)[0][-1]

        ind.append([xind, yind])
        poi.append([a, b])

    for j in range(len(ind) - 1):
        a = ind[j]  # first point of the line
        b = ind[j + 1]  # second point of the line
        tot_dist += abs(math.dist(poi[j], poi[j + 1]))  # distance to cal the x values
        serie = connect(np.array([a, b]))  # get all points laying under the segment

        tot_n += serie.shape[0]  # number of points in the profile
        z = obj.Z[serie[:, 1], serie[:, 0]]
        Z = np.hstack((Z, z))  # create the total profile

    x = np.linspace(0, tot_dist, tot_n)
    prf.setValues(x, Z, bplt=bplt)

    return prf


class ComplexExtractor(Extractor, ABC):
    def template(self):
        sur = Extractor._openTopo()
        _, self.points = ComplexExtractor.profile(sur, width=1, bplt=False)

    def apply(self, obj: surface.Surface, bplt=False):
        prf = _extractProfileWithPoints(obj, self.points, bplt=bplt)
        return prf

    # TODO: width
    @staticmethod
    def profile(obj: surface.Surface, width=1, bplt=False):
        """
        Opens a plot figure to choose the profile and extracts it
        The profile is taken with a mpl.widgets.PolygonSelector

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the profile is extracted
        width: int
            The lateral number of values averged when extracting the profile
        bplt: bool
            If True plots the extracted profile

        Returns
        ----------
        prf: profile.Profile
            The extracted profile
        points: the points selected by the user
        """
        def onClose(event):
            _extractProfileWithPoints(obj, selector.verts, prf, bplt=bplt)

        fig, ax = plt.subplots()
        ax.pcolormesh(obj.X, obj.Y, obj.Z, cmap=cm.rainbow)
        ax.set_title('3 Points plane fit')
        fig.canvas.mpl_connect('close_event', onClose)

        selector = PolygonSelector(ax, lambda *args: None)
        prf = profile.Profile()

        plt.show()

        return prf, selector.verts


class SphereExtractor(Extractor, ABC):
    @staticmethod
    def sphereProfile(obj: surface.Surface, startP, bplt=False):
        """
        Returns the profile starting from the start point on the positive x direction

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the profile is extracted
        startP : str
            Method used to find the start (x, y) point on the topography
                'max': the start point corresponds to the maximum Z of the topography
                'fit': the start point is the center coords of the best fit sphere
                'center': the start point is the center of the topography
                'local': the start point is the local maximum closest to the center of the topography
        bplt: bool
            If True plots the topography and the line where the profile is taken from

        Returns
        ----------
        profile: prof.Profile()
            The extracted profile
        """
        # TODO : check if xind and yind are inverted
        prf = profile.Profile()
        if startP == 'max':
            raveled = np.nanargmax(obj.Z)
            unraveled = np.unravel_index(raveled, obj.Z.shape)
            xind = unraveled[0]
            yind = unraveled[1]
        elif startP == 'fit':
            from surfile.geometry import sphere

            r, C = sphere.remove(obj, bplt=False)
            yc = C[0][0]
            xc = C[1][0]
            xind = np.argwhere(obj.x > xc)[0][0]
            yind = np.argwhere(obj.y > yc)[0][0]
        elif startP == 'center':
            xind = int(len(obj.x) / 2)
            yind = int(len(obj.y) / 2)
        elif startP == 'local':
            maxima = (obj.Z == ndimage.filters.maximum_filter(obj.Z, 5))
            mid = np.asarray(obj.Z.shape) / 2
            maxinds = np.argwhere(maxima)  # find all maxima indices
            center_max = maxinds[np.argmin(np.linalg.norm(maxinds - mid, axis=1))]

            xind = center_max[0]
            yind = center_max[1]
        else:
            raise Exception(f'{startP} is not a valid start method')

        if bplt:
            fig, ax = plt.subplots()
            ax.pcolormesh(obj.X, obj.Y, obj.Z, cmap=cm.viridis)  # hot, viridis, rainbow
            funct.persFig(
                [ax],
                gridcol='grey',
                xlab='x [um]',
                ylab='y [um]'
            )
            plt.hlines(obj.y[xind], obj.x[yind], obj.x[-1])
            plt.show()

        prf.setValues(obj.x[yind:-1], obj.Z[xind][yind:-1], bplt=bplt)

        return prf
