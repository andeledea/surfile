"""
'surfile.surface'
- data structure for surface objects
- plots of the surface
- io operation

Example
-------
>>> from surfile import surface
>>> sur = surface.Surface() # instantiate an empty surface
>>> sur.openFile('path to file', bplt=False)
>>> sur.pltC()

@author: Andrea Giura
"""
import copy
import profile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from scipy import interpolate, ndimage, special

from surfile import profile, measfile_io, funct
from surfile.funct import options, rcs


class Surface:
    """
    Class for handling surface data
    Provides io file operations in different formats
    Provides simple visualization plots
    """
    def __init__(self):
        """Instantiate an empty Surface object"""
        self.X0, self.Y0, self.Z0 = None, None, None
        self.Z = None
        self.Y = None
        self.X = None

        self.y = None
        self.x = None
        self.rangeY = None
        self.rangeX = None

        self.name = 'Figure'

    def openTxt(self, fname, bplt, typ='x'):
        """
        Opens a txt file containing the values of the topography
        see Notes for more information on file format.

        Parameters
        ----------
        fname : str
            The file path
        bplt : bool
            If true plots the opened surface
        typ : str, optional
            Type of text file see Notes, by default 'x'
            
        Notes
        -----
        This method accepts two different txt formats:
        - typ = 's' txt file with 4 header lines with values respectively 
        [nx, ny, dx, dy] and then the list of Z values of the array.
        - typ = 'x' txt file with three columns [x, y, z]
        """
        self.name = os.path.basename(fname)
        self.name = os.path.splitext(self.name)[0]

        if typ is None:
            typ = input("choose txt type [Xyz, Spacez, ...]")
            typ = typ.lower()
        if typ == 'x':
            self.X, self.Y, self.Z, self.x, self.y = measfile_io.read_xyztxt(fname)
        if typ == 's':
            self.X, self.Y, self.Z, self.x, self.y = measfile_io.read_spaceZtxt(fname)

        self.X0 = self.X
        self.Y0 = self.Y
        self.Z0 = self.Z

        if bplt: self.pltC()

    def openFile(self, fname, bplt, interp=False, userscalecorrections=[1.0, 1.0, 1.0]):
        """
        Opens a file from a supported intrument, the list of 
        supported types is in measfile_io documentation.

        Parameters
        ----------
        fname : str
            The file path
        bplt : bool
            If true plots the opened surface
        interp : bool, optional
            If true uses interpolation to fill NaNs, by default False
        userscalecorrections : list
            Array to correct the values of the topography [x_mul, y_mul, z_mul].
        """
        self.name = os.path.basename(fname)
        self.name = os.path.splitext(self.name)[0]

        dx, dy, z_map, weights, magnification, measdate = \
            measfile_io.read_microscopedata(fname, userscalecorrections, interp)
        (n_y, n_x) = z_map.shape
        self.rangeX = n_x * dx
        self.rangeY = n_y * dy
        self.x = np.linspace(0, self.rangeX, num=n_x)
        self.y = np.linspace(0, self.rangeY, num=n_y)

        # create main XYZ and backup of original points in Z0
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = z_map

        self.X0, self.Y0, self.Z0 = copy.copy(self.X), copy.copy(self.Y), copy.copy(self.Z)

        if bplt: self.pltC()

    def saveAsc(self, fname):
        """
        Saves the topography in the .asc file format of the tracOptic project

        Parameters
        ----------
        fname : str
            If fname is a folder the file will be saved in that folder with the surface name
            If fname is not a folder the file will be saved at fname
            
        Notes
        -----
        The tracOptic file format: \n
        {self.name}\n
        X - length: {max(self.x) - min(self.x)}\n
        Y - length: {max(self.y) - min(self.y)}\n
        X - pixel number: {len(self.x)}\n
        Y - pixel number: {len(self.y)}\n
        Z - data array start: (all the values of the Z array)
        """
        def saveLine(line):
            line = line / 1000  # in um
            line.tofile(fout, sep='\t', format='%.4f')
            fout.write('\n')

        name = os.path.join(fname, self.name + '.asc') if os.path.isdir(fname) else os.path.splitext(fname)[0] + '.asc'
        with open(name, 'w') as fout:
            fout.write(f'# {self.name}\n')
            fout.write(f'# X - length:\t{max(self.x) - min(self.x)}\n')
            fout.write(f'# Y - length:\t{max(self.y) - min(self.y)}\n')
            fout.write(f'# X - pixel number:\t{len(self.x)}\n')
            fout.write(f'# Y - pixel number:\t{len(self.y)}\n\n')
            fout.write(f'# Z - data array start:\n')

            np.apply_along_axis(saveLine, axis=1, arr=self.Z)

    def saveTxt(self, fname):
        """
        Saves the topography in the .txt file format with three cols

        Parameters
        ----------
        fname : str
            If fname is a folder the file will be saved in that folder with the surface name
            If fname is not a folder the file will be saved at fname
        """
        name = os.path.join(fname, self.name + '.txt') if os.path.isdir(fname) else os.path.splitext(fname)[0] + '.txt'
        np.savetxt(name, np.c_[self.X.ravel().T, self.Y.ravel().T, self.Z.ravel().T], fmt='%.4e')

    def rotate(self, angle):
        """
        Rotates the original topography by the specified angle

        Parameters
        ----------
        angle: float
            The angle of rotation
            
        Notes
        -----
        <span style="color:orange">This function will be moved to a utility module in the future
        use with caution !!!</span>.
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
            
        Notes
        -----
        <span style="color:orange">This function will be moved to a utility module in the future
        use with caution !!!</span>.
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

    def fillNM(self, method='cubic'):
        """
        Fills the surface non measured points
        
        Parameters
        ----------
        method : str
            The interpolation method used
            
        Notes
        -----
        <span style="color:orange">This function will be moved to a utility module in the future
        use with caution !!!</span>.
        """
        z_ma = np.ma.masked_invalid(self.Z)
        self.Z = interpolate.griddata((self.X[~z_ma.mask], self.Y[~z_ma.mask]),
                                      z_ma[~z_ma.mask].ravel(),
                                      (self.X, self.Y),
                                      method=method)

    def chauvenet(self, iterative=True, threshold=0.5, mean=None, stdv=None):
        """
        Removes the outliers from the topography
        using the chouvenet criterion

        Parameters
        ----------
        iterative : bool
            If true keeps calling the function until the number of outliers is 0
            If true the mean and stdv parameters are ignored as if they were not provided
        threshold : float
            The acceptance threshold of the criterion
        mean : float
            The mean of the expected distribution
            If none the program calculates the mean of the distribution
        stdv : float
            The std dev of the expected distribution
            If none the program calculates the std dev of the distribution
            
        Notes
        -----
        <span style="color:orange">This function will be moved to a utility module in the future
        use with caution !!!</span>.
        """
        # https://github.com/msproteomicstools/msproteomicstools/blob/master/msproteomicstoolslib/math/chauvenet.py
        prenan = np.count_nonzero(np.isnan(self.Z))

        if mean is None or iterative:
            mean = np.nanmean(self.Z)
        if stdv is None or iterative:
            stdv = np.nanstd(self.Z)
        N = self.Z.size  # Lenght of incoming arrays
        criterion = 1.0 / (2 * N)
        d = np.abs(self.Z - mean) / stdv  # Distance of a value to mean in stdv's
        d /= 2.0 ** threshold
        prob = special.erfc(d)
        fil = prob >= criterion

        self.Z[~fil] = np.nan
        postnan = np.count_nonzero(np.isnan(self.Z))
        addednan = postnan - prenan
        print(f'{prenan=}, {postnan=}, diff = {addednan}')

        if addednan > 0 and iterative:
            self.chauvenet(iterative=True, threshold=threshold)
        else:
            return

    def toProfiles(self, axis='x'):
        """
        Splits the topography into vertical or horizontal profiles

        Parameters
        ----------
        axis: str
            'x' or 'y', the axis on which the f is applied

        Returns
        -------
        res: []
            The resulting profiles
        """
        if axis not in ['x', 'y']: raise Exception(f'{axis} is not a valid axis')

        def toPrf(vals):
            prof = profile.Profile()
            prof.setValues(self.x if axis == 'x' else self.y, vals, bplt=False)
            return prof

        profiles = np.apply_along_axis(toPrf, arr=self.Z, axis=1 if axis == 'x' else 0)

        return profiles

    #################
    # PLOT SECTION  #
    #################
    @options(bplt=rcs.params['bs3_D'], save=rcs.params['ss3_D'])
    def plt3D(self):
        """Plots a 3D view of the surface"""
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        p = ax_3d.plot_surface(self.X, self.Y, self.Z, cmap=cm.rainbow)  # hot, viridis, rainbow
        funct.persFig(
            [ax_3d],
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]',
            zlab='z [nm]'
        )
        ax_3d.set_title(self.name)

    @options(bplt=rcs.params['bsCom'], save=rcs.params['ssCom'])
    def pltCompare(self):
        """Plots the current topography data and the original data"""
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
        ax.set_title(self.name)

    @options(bplt=rcs.params['bsCol'], save=rcs.params['ssCol'])
    def pltC(self):
        """Plots the topography (pcolormesh)"""
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
