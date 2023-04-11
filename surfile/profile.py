import os
import numpy as np
import matplotlib.pyplot as plt

from surfile import funct
from scipy import signal


class Profile:
    def __init__(self):
        self.X = None
        self.Z = None
        self.Z0, self.X0 = None, None

        self.name = 'Profile'

    def openPrf(self, fname, bplt):
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

    def openTS(self, fname, bplt):
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

    def savecsv(self):
        name = input('Choose filename: ')
        np.savetxt('c:/monticone/' + name + '.csv',
                   np.hstack((self.X.reshape(len(self.X), 1), self.Z.reshape(len(self.Z), 1))), delimiter=';')

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
        self.X0 = self.X = X
        self.Z0 = self.Z = Y

        if bplt: self.pltPrf()

    def roughnessParams(self, cutoff, ncutoffs, bplt):  # TODO: adapt to filter class
        """
        Applies the indicated filter and calculates the roughness parameters

        Parameters
        ----------
        cutoff: float
            cutoff length
        ncutoffs: int
            number of cutoffs to considerate in the center of the profile
        bplt: bool
            shows roi plot

        Returns
        ----------
        (RA, RQ, RP, RV, RZ, RSK, RKU): (float, ...)
            Calculated roughness parameters
        """
        print(f'Applying filter cutoff: {cutoff}')

        # samples preparation for calculation of params
        def prepare_roi():
            nsample_cutoff = cutoff / (np.max(self.X) / np.size(self.X))
            nsample_region = nsample_cutoff * ncutoffs

            border = round((np.size(self.Z) - nsample_region) / 2)
            roi_I = self.Z[border: -border]  # la roi sono i cutoff in mezzo

            envelope = self.__gaussianKernel(self.Z, cutoff)  # TODO: move method to roughness module
            roi_F = roi_I - envelope[border: -border]  # applico il filtro per il calcolo

            if bplt: self.__pltRoughness(ncutoffs, cutoff, border, roi_I, roi_F, envelope)
            return roi_F  # cutoff roi

        roi = prepare_roi()

        RA = np.sum(abs(roi)) / np.size(roi)
        RQ = np.sqrt(np.sum(abs(roi ** 2)) / np.size(roi))
        RP = abs(np.max(roi))
        RV = abs(np.min(roi))
        RT = RP + RV
        RSK = (np.sum(roi ** 3) / np.size(roi)) / (RQ ** 3)
        RKU = (np.sum(roi ** 4) / np.size(roi)) / (RQ ** 4)
        return RA, RQ, RP, RV, RT, RSK, RKU

    #################
    # PLOT SECTION  #
    #################
    def pltPrf(self):  # plots the profile
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.X, self.Z, color='teal')
        funct.persFig(
            [ax],
            gridcol='grey',
            xlab='x [mm]',
            ylab='z [um]'
        )
        ax.set_title(self.name)
        plt.show()

    def pltCompare(self):  # plots the profile
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
        plt.show()

    def __pltRoughness(self, ncutoffs, cutoff, border, roi_I, roi_F, envelope):
        fig, ax = plt.subplots()

        twin = ax.twinx()
        twin.set_ylabel('Filtered roi')

        ax.set_title(f'Gaussian filter: cutoffs: {ncutoffs}, cutoff length: {cutoff}')
        ax.plot(self.X, self.Z, alpha=0.2, label='Data')
        ax.plot(self.X[border: -border], roi_I, alpha=0.3, label='roi unfiltered')

        twin.set_ylim(np.min(roi_F) - 0.7 * (np.max(roi_F) - np.min(roi_F)),
                      np.max(roi_F) + 0.7 * (np.max(roi_F) - np.min(roi_F)))
        rF, = twin.plot(self.X[border: -border], roi_F, color='green', alpha=0.6, label='roi filtered')
        twin.tick_params(axis='y', colors=rF.get_color())

        ax.plot(self.X, envelope, color='red', label='filter envelope')

        ax.legend()
        plt.show()
