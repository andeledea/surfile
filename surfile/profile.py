"""
'surfile.profile'
- data structure for profile objects
- plots of the profile
- io operation for data storage

Example
-------
>>> from surfile import profile
>>> prf = profile.Profile() # instantiate an empty profile
>>> prf.openPrf('path to file', bplt=False)
>>> prf.pltPrf()

@author: Andrea Giura
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import copy

from surfile import funct
from surfile.funct import options, rcs


class Profile:
    """
    Class for handling profile data
    Provides io file operations in different formats
    Provides simple visualization plots
    """
    def __init__(self):
        """Instantiate an empty Profile object"""
        self.X = None
        self.Z = None
        self.Z0, self.X0 = None, None

        self.name = 'Profile'

    def openPrf(self, fname, bplt):
        """
        Opens a .prf file from taylor hobson profilometers

        Parameters
        ----------
        fname : str
            The file path
        bplt : bool
            If true plots the opened profile
        """
        z = []
        xs = 0
        zs = 0
        with open(fname, 'r') as fin:
            self.name = os.path.basename(fname)

            charlines = 0
            for line in fin.readlines():
                word = line.split()[0]
                if word == 'SPACING':  # linea di spacing x
                    xs = float(line.split()[2])
                    print(f'Spacing x: {xs}')
                if word == 'CZ':  # linea di spacing z
                    zs = float(line.split()[4]) * 10 ** 3
                    print(f'Scaling z: {zs}')

                try:  # salvo solo i valori numerici
                    z.append(float(word) * zs)
                except ValueError:
                    charlines += 1
            print(f'Skipped {charlines} word lines')
            z.pop(0)

            self.Z0 = self.Z = np.array(z)
            self.X0 = self.X = np.linspace(0, (len(z)) * xs, len(z))

            if bplt: self.pltPrf()

    def openCHR(self, dirname, bplt):
        """
        Opens a CHR folder containing the profile data and the
        stitching positions.

        Parameters
        ----------
        dirname : str
            The directory path
        bplt : bool
            If true plots the opened profile
        """
        data = np.genfromtxt(dirname + '/data.CHRdat', delimiter=',')
        print(data.shape)
        # stitch = np.loadtxt(dirname + '/stitching.py.CHRdat')
        yB = np.array(data[:, 2])
        yC = np.array(data[:, 3])

        self.name = os.path.basename(dirname)

        self.X = np.array(data[:, 0])
        self.Z = (yB - yC)

        self.X0 = copy.copy(self.X)
        self.Z0 = copy.copy(self.Z)

        if bplt:
            fig, ax = plt.subplots()
            ax.axis('equal')
            ax.plot(self.X, self.Z, 'r')
            ax.set_title(self.name)

            ax.set_xlabel('x [um]')
            ax.set_ylabel('y [um]')

            # try:
            #     xF = np.array(stitch[:, 0]) * 1000
            #     yF = stitch[:, 2]
            #     yF = np.array([-i for i in yF])
            #
            #     ax.plot(xF, yF, 'o')
            # except:
            #     print("No stitching.py needed")

            plt.show()

    def openTS(self, fname, bplt):
        """
        OBSOLETE: opens the talyStep data files with multiple measurements
        ads asks the user which measurement to consider.

        Parameters
        ----------
        fname : str
            The file path
        bplt : bool
            If true plots the opened profile
        """
        with open(fname, 'rb') as tsfile:
            firstline_sp = tsfile.readline().split(b'\x1a', maxsplit=1)
            names = firstline_sp[0]
            names = [a.decode('utf-8').strip() for a in names.split(b'\x03')]

            file_bytes = firstline_sp[-1] + tsfile.read()

            file_splits = []
            for i, name in enumerate(names[:-1]):
                d = name.encode('utf-8')
                s = file_bytes.split(d, maxsplit=1)
                if len(s) > 1:
                    file_splits.append(d + s[1])
                    file_bytes = s[1]

        for i, s in enumerate(file_splits):
            name = s[0: 42].decode('utf-8')
            print(f'{i}, Name = {name.strip()}')

        s = file_splits[int(input('Choose graph number: '))]

        Lcutoff = np.frombuffer(s[356: 360], dtype=np.single)[0]
        Factor = np.frombuffer(s[360: 364], dtype=np.single)[0]
        print(f'Lc_o = {Lcutoff}, Factor = {Factor}')

        dt = np.dtype('<i2')
        Ncutoffs = np.frombuffer(s[762: 764], dtype=dt)[0]
        Npoints = np.frombuffer(s[764: 766], dtype=dt)[0]
        Speed = np.frombuffer(s[766: 768], dtype=dt)[0]
        print(f'N = {Npoints}, Nc_o = {Ncutoffs}, Speed = {Speed}')

        self.X = self.X0 = np.linspace(0, Ncutoffs * Lcutoff, Npoints)
        self.Z = self.Z0 = np.frombuffer(s[920: 920 + 2 * Npoints], dtype=dt) / Factor

        if bplt: self.pltPrf()

    def openTxt(self, fname, bplt, header=0):
        """
        Opens a txt file with 2 columns [x, z]

        Parameters
        ----------
        fname : str
            The file path
        bplt : bool
            If true plots the opened profile
        header : int, optional
            Number of lines to be discarded when opening the file, by default 0
            
        Notes
        -----
        This function uses np.genfromtxt() with converter
        >>> converters={0: lambda s: float(s or np.nan)}
        this is used to correctly detect all NaN formats in txt
        """ 
        self.name = os.path.basename(fname)
        
        self.X, self.Z = np.genfromtxt(fname, 
                                       skip_header=header, 
                                       usecols=[0, 1], unpack=True,
                                       converters={0: lambda s: float(s or np.nan)})
        self.X0 = copy.copy(self.X)
        self.Z0 = copy.copy(self.Z)
        if bplt: self.pltPrf()
    
    def saveTxt(self, fname):
        """
        Saves the profile in a txt format 2 columns [x, z]

        Parameters
        ----------
        fname : str
            The output file path
        """
        name = os.path.join(fname, self.name + '.txt') if os.path.isdir(fname) else os.path.splitext(fname)[0] + '.txt'
        np.savetxt(name, np.c_[self.X.ravel().T, self.Z.ravel().T], fmt='%.4e')

    def setValues(self, X, Y, bplt):
        """
        Sets the values for the profile

        Parameters
        ----------
        X: []
            The X values of the profile
        Y: []
            The Y values of the profile
        bplt: bool
            Plots the profile
        """
        self.X0 = self.X = np.asarray(X)
        self.Z0 = self.Z = np.asarray(Y)

        if bplt: self.pltPrf()
        
    def fillNM(self, bplt=False):
        nans, f = funct.nan_helper(self.Z)
        self.Z[nans]= np.interp(f(nans), f(~nans), self.Z[~nans])
        
        if bplt: self.pltCompare()

    #################
    # PLOT SECTION  #
    #################
    @options(bplt=rcs.params['bpPrf'], save=rcs.params['spPrf'])
    def pltPrf(self):
        """Plots the profile"""
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.X, self.Z, color='teal')
        funct.persFig(
            [ax],
            gridcol='grey',
            xlab='x [mm]',
            ylab='z [um]'
        )
        ax.set_title(self.name)

    @options(bplt=rcs.params['bpCom'], save=rcs.params['spCom'])
    def pltCompare(self):
        """Plots the current profile and the original data"""
        fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
        ax.plot(self.X0, self.Z0, color='teal')
        bx.plot(self.X, self.Z, color='teal')
        funct.persFig(
            [ax, bx],
            gridcol='grey',
            xlab='x [mm]',
            ylab='z [um]'
        )
        ax.set_title(self.name)
