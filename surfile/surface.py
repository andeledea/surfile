import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, lines
import os
from scipy import interpolate, ndimage
from alive_progress import alive_bar

from surfile import profile as prf
from surfile import funct
from surfile import measfile_io


# plu classes

class Surface:
    def __init__(self):
        self.X0, self.Y0, self.Z0 = None, None, None
        self.Z = None
        self.Y = None
        self.X = None

        self.y = None
        self.x = None
        self.rangeY = None
        self.rangeX = None

        self.name = 'Figure'

    def openTxt(self, fname, bplt):
        with open(fname, 'r') as fin:
            self.name = os.path.basename(fin.name)

            line = fin.readline().split()
            sx = int(line[0])  # read number of x points
            sy = int(line[1])  # read number of y points
            print(f'Pixels: {sx} x {sy}')

            spacex = float(line[2])  # read x spacing
            spacey = float(line[3])  # read y spacing

        plu = np.loadtxt(fname, usecols=range(sy), skiprows=1)
        plu = (plu - np.mean(plu)) * (10 ** 6)  # 10^6 from mm to nm

        self.rangeX = sx * spacex
        self.rangeY = sy * spacey
        self.x = np.linspace(0, sx * spacex, num=sx)
        self.y = np.linspace(0, sy * spacey, num=sy)

        # create main XYZ and backup of original points in Z0
        self.X0, self.Y0 = self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z0 = self.Z = np.transpose(plu)

        if bplt: self.pltC()

    def openFile(self, fname, bplt, interp=False):
        self.name = os.path.basename(fname)

        userscalecorrections = [1.0, 1.0, 1.0]
        dx, dy, z_map, weights, magnification, measdate = \
            measfile_io.read_microscopedata(fname, userscalecorrections, interp)
        (n_y, n_x) = z_map.shape
        self.rangeX = n_x * dx
        self.rangeY = n_y * dy
        self.x = np.linspace(0, self.rangeX, num=n_x)
        self.y = np.linspace(0, self.rangeY, num=n_y)

        # create main XYZ and backup of original points in Z0
        self.X0, self.Y0 = self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z0 = self.Z = z_map

        if bplt: self.pltC()

    def rotate(self, angle):
        """
        Rotates the original topography by the specified angle

        Parameters
        ----------
        angle: float
            The angle of rotation
        """
        self.Z = ndimage.rotate(self.Z0, angle, order=0, reshape=False, cval=np.nan)

    def resample(self, newXsize, newYsize):
        """
        Resamples the topography and fills the non-measured points

        Parameters
        ----------
        newXsize: int
            Number of points desired on the x-axis
        newYsize: int
            Number of points desired on the y-axis
        """
        xi = np.linspace(0, self.rangeX, newXsize)
        yi = np.linspace(0, self.rangeY, newYsize)
        Xi, Yi = np.meshgrid(xi, yi)

        XY = np.vstack([self.X.reshape(np.size(self.X)),
                        self.Y.reshape(np.size(self.Y))]).T

        Zi = interpolate.griddata(XY, self.Z.reshape(np.size(self.Z)), (Xi, Yi), method='cubic')  # 'linear' 'cubic'
        print(np.shape(self.Z), np.shape(Zi))

        self.x = xi
        self.y = yi

        self.X = Xi
        self.Y = Yi
        self.Z = Zi

    #################
    # PLOT SECTION  #
    #################
    def plt3D(self):
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.plot_surface(self.X, self.Y, self.Z, cmap=cm.rainbow)  # hot, viridis, rainbow
        funct.persFig(
            [ax_3d],
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]',
            zlab='z [nm]'
        )
        ax_3d.set_title(self.name)
        # ax_3d.colorbar(p)
        plt.show()

    def pltCompare(self):
        fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
        p1 = ax.pcolormesh(self.X0, self.Y0, self.Z0, cmap=cm.jet)  # hot, viridis, rainbow
        p2 = bx.pcolormesh(self.X, self.Y, self.Z, cmap=cm.jet)  # hot, viridis, rainbow
        fig.colorbar(p1, ax=ax)
        fig.colorbar(p2, ax=bx)
        funct.persFig(
            [ax, bx],
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]'
        )
        plt.show()

    def pltC(self):
        fig = plt.figure()
        ax_2d = fig.add_subplot(111)
        ax_2d.pcolormesh(self.X, self.Y, self.Z, cmap=cm.viridis)  # hot, viridis, rainbow
        funct.persFig(
            [ax_2d],
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]'
        )
        ax_2d.set_title(self.name)
        ax_2d.grid(False)
        ax_2d.set_aspect('equal')
        plt.show()
