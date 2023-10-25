"""
'surfile.roughness'
- analysis of roughness features for:
    - Profiles
    - Surfaces

@author: Dorothee Hueser, Andrea Giura
"""

from matplotlib import cm
import numpy as np
from dataclasses import dataclass

from surfile import profile, surface, filter, remover, funct

import matplotlib.pyplot as plt


def eval_pinter(PSD, fx0, fy0, frc, theta, nxDC, nyDC):
    def bilinear_interp(x, y, x1, x2, y1, y2, z11, z12, z21, z22):
        r1 = ((x2 - x) / (x2 - x1)) * z11 + ((x - x1) / (x2 - x1)) * z21
        r2 = ((x2 - x) / (x2 - x1)) * z12 + ((x - x1) / (x2 - x1)) * z22
        p = ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2
        return p

    (Ny, Nx) = PSD.shape
    fxc = frc * np.cos(theta)
    fyc = frc * np.sin(theta)
    ix = int(np.floor(fxc / fx0))
    iy = int(np.floor(fyc / fy0))
    ixp = nxDC + ix
    iyp = nyDC + iy
    ixpp = ixp + 1
    iypp = iyp + 1
    if ixpp == Nx:
        ixpp = 0
    if iypp == Ny:
        iypp = 0
    pinter = bilinear_interp(fxc, fyc, fx0 * ix, fx0 * (ix + 1),
                             fy0 * iy, fy0 * (iy + 1), PSD[iyp][ixp], PSD[iypp][ixp],
                             PSD[iyp][ixpp], PSD[iypp][ixpp])
    return pinter


class Psd:
    def __init__(self):
        self.deltaX = None
        self.deltaY = None
        self.psd = None
        self.psdp = None  # polar spectra

        self.fx = None
        self.fy = None

        self.psdx = None
        self.psdy = None

    def evalPsd(self, obj: surface.Surface, bplt=False):
        """
        Evaluate the power spectral density of the topography

        Parameters
        ----------
        obj: surface.Surface
            The topography
        bplt: bool
            If True plots the PSD result

        Returns
        ----------
        psd: np.ndarray
            The calculated psd of the surface
        fx, fy: np.array
            The x and y axis of the psd
        psd_x, psd_y: np.ndarray
            The psd along the given axis
        """
        self.deltaX = np.max(obj.x) / np.size(obj.x)
        self.deltaY = np.max(obj.y) / np.size(obj.y)
        Z = obj.Z - np.mean(obj.Z)
        (Ny, Nx) = Z.shape

        Fz = np.fft.fft2(Z) / (np.float64(Nx * Ny))  # dimension of a height or length
        Fzx = np.fft.fft(Z, axis=1) / (np.float64(Nx))  # dimension length
        Fzy = np.fft.fft(Z, axis=0) / (np.float64(Ny))
        PSD = Fz * np.conj(Fz)
        RePSD = np.real(np.fft.fftshift(PSD, axes=(0, 1)))
        PSDx = Fzx * np.conj(Fzx)
        RePSDx = np.real(np.fft.fftshift(PSDx, axes=1))
        PSDy = Fzy * np.conj(Fzy)
        RePSDy = np.real(np.fft.fftshift(PSDy, axes=0))

        Lx = Nx * self.deltaX
        Ly = Ny * self.deltaY
        RePSD = RePSD * (Lx * Ly)  # dimension length^4
        RePSDx = RePSDx * Lx  # dimension length^3
        RePSDy = RePSDy * Ly
        dfx = 1 / Lx
        dfy = 1 / Ly

        fx = (np.arange(0, Nx) - np.floor(0.5 * Nx))
        fy = (np.arange(0, Ny) - np.floor(0.5 * Ny))

        fx = fx * dfx
        fy = fy * dfy

        print(f'Sq = {np.sqrt(np.sum(Z * Z) / (np.float64(Nx * Ny)))}')
        print(f'dfx: {dfx} dfy: {dfy}')
        print(f'from PSD: {np.sqrt(np.sum(RePSD * dfx * dfy))} Lx*Ly = {Lx * Ly} Lx = {Lx} Ly = {Ly}')
        print(f'Rq x: {np.sqrt(np.sum(np.mean(RePSDx, axis=0) * dfx))}')
        print(f'Rq y: {np.sqrt(np.sum(np.mean(RePSDy, axis=1) * dfy))}')

        # definition of PSD is Fourier*conj(Fourier) * Lx *Ly
        # PSD has dimension length^4
        # PSDx, PSDy have dimension length^3
        self.psd = RePSD

        self.fx = fx
        self.fy = fy

        self.psdx = RePSDx
        self.psdy = RePSDy

        if bplt:
            fig, ax = plt.subplots()
            ax.imshow(np.log10(RePSD), extent=[fx[0], fx[-1], fy[0], fy[-1]])
            funct.persFig([ax], xlab=r'$f_x / \mu m^{-1}$', ylab=r'$f_y / \mu m^{-1}$')

            fig2, (ax, bx) = plt.subplots(nrows=1, ncols=2)
            ax.imshow(np.log10(self.psdx), extent=[fx[0], fx[-1], 0, Ny * self.deltaY],
                      aspect=2 * fx[-1] / (Ny * self.deltaY))
            bx.imshow(np.log10(self.psdy), extent=[0, Nx * self.deltaX, fy[0], fy[-1]],
                      aspect=0.5 * Nx * self.deltaX / (fy[-1]))
            funct.persFig([ax], xlab=r'$f_x / \mu m^{-1}$', ylab=r'$y / \mu m$')
            funct.persFig([bx], xlab=r'$x / \mu m$', ylab=r'$f_y / \mu m^{-1}$')
            plt.show()
        return RePSD, fx, RePSDx, fy, RePSDy

    def polarSpectra(self, df_fct, bplt=False):
        """
        df_fct: float
        Calculate the spectra in polar coordinates

        Parameters
        ----------
        bplt: bool
            If true the spectra is plotted in a 3d projection axis

        Returns
        -------
        psdP: np.ndarray
            The polar spectra evaluated
        Fr, Th: np.array
            The polar coordinate vectors
        """
        if self.psd is None: raise Exception('Polar spectra failed: psd has not been evaluated')

        (ny, nx) = self.psd.shape
        df_x = 1.0 / (nx * self.deltaX)
        df_y = 1.0 / (ny * self.deltaY)
        nxDC = int(np.floor(nx / 2))
        nyDC = int(np.floor(ny / 2))

        frmax = np.min([df_x * (nx - 2) / 2, df_y * (ny - 2) / 2])
        fr0 = df_fct * (df_x + df_y)
        fr = np.linspace(0.1 * fr0, frmax, 512);
        #    thetas = np.linspace(0.001, 2*np.pi, 360)
        thetas = np.linspace(0.0, 2 * np.pi, 205) + 0.1 * np.pi
        Th, Fr = np.meshgrid(thetas, fr)
        (nf, nt) = Fr.shape
        PSDp = np.ones((nf, nt)) * 1e-4
        for ir in range(0, nf):
            frc = fr[ir]
            for ith in range(0, nt):
                PSDp[ir][ith] = eval_pinter(self.psd, df_x, df_y, frc, thetas[ith], nxDC, nyDC)

        self.psdp = PSDp
        if bplt:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            X, Y = Fr * np.cos(Th), Fr * np.sin(Th)
            ax.plot_surface(X, Y, np.log10(PSDp), cmap=cm.gnuplot)

            plt.show()

        return PSDp, Fr, Th

    def angleIntegratedSpectra(self, df_fct, bplt=None):
        """
        Calculate the angle integrated

        Parameters
        ----------
        df_fct: float
        bplt: bool
            If true the angle integrated spectra is plotted

        Returns
        -------
        psdP: np.ndarray
            The polar spectra evaluated
        Fr, Th: np.array
            The polar coordinate vectors
        """
        if self.psd is None: raise Exception('Polar spectra failed: psd has not been evaluated')

        (ny, nx) = self.psd.shape
        df_x = 1.0 / (nx * self.deltaX)
        df_y = 1.0 / (ny * self.deltaY)
        nxDC = int(np.floor(nx / 2))
        nyDC = int(np.floor(ny / 2))

        frmax = np.min([df_x * (nx - 2) / 2, df_y * (ny - 2) / 2])
        fr0 = df_fct * (df_x + df_y)
        fr = np.arange(fr0, frmax, fr0)
        nf = len(fr)
        PSDr = np.zeros(nf)
        PSDav = np.zeros(nf)
        for ir in range(0, nf):
            frc = fr[ir]
            nth = int(np.ceil(2 * np.pi * frc / fr0))
            dth = 2 * np.pi / nth
            thetas = np.linspace(0.0, 2 * np.pi, nth)  # + 0.1*np.pi
            dth = thetas[1] - thetas[0]
            psum = 0
            for theta in thetas:
                pinter = eval_pinter(self.psd, df_x, df_y, frc, theta, nxDC, nyDC)
                psum = psum + pinter
            PSDav[ir] = psum / nth  # dimension of height^2 * lateral^2 i.e. length^4
            PSDr[ir] = dth * frc * psum / (2 * np.pi)  # frc has dimension of 1 / lateral
            # therefore PSDr: height^2 * lateral
        # PSDr = PSDr * fr0
        print(f'Rq r: {np.sqrt(np.sum(PSDr * fr0) * (2 * np.pi))}')
        print(f'Rq r: {np.sqrt(2 * np.pi * np.sum(PSDav * fr) * fr0)}')

        if bplt:
            fig, ax = plt.subplots()
            # ax.loglog(fr, PSDr, 'bx-', markersize=3)
            ax.loglog(fr, PSDav, 'rx-', markersize=3)
            plt.title('Average over all polar angles')
            plt.show()

        return PSDr, PSDav, fr, fr0

    @funct.options(bplt=True, csvPath='out\\')
    def averageSpectra(self, bplt=False):
        """
        Calculate the average specra in the x and y directions

        Parameters
        ----------
        bplt: bool
            Plots the mean spectra if true

        Returns
        -------
        psdMeanx, psdMeany: np.array
            The mean psd arrays
        """
        PSDxmean = np.mean(self.psdx, axis=0)
        PSDymean = np.mean(self.psdy, axis=1)

        if bplt:
            fig, ((ax, bx), (cx, dx)) = plt.subplots(nrows=2, ncols=2)
            ax.loglog(self.fx, PSDxmean, 'rx-', markersize=3, label='mean 1d PSDx in x dir')
            bx.loglog(self.fy, PSDymean, 'kx-', markersize=3, label='mean 1d PSDy in y dir')

            cx.plot(self.fx, PSDxmean, 'rx-', markersize=3, label='mean 1d PSDx in x dir')
            dx.plot(self.fy, PSDymean, 'kx-', markersize=3, label='mean 1d PSDy in y dir')

            funct.persFig([ax, bx], xlab=r'$f$ in $\frac{1}{\mu m}$', ylab=r'PSD in $\mu m^3$')
            fig.legend()

        return (self.fx[self.fx >= 0], PSDxmean[self.fx >= 0]), (self.fy[self.fy >= 0], PSDymean[self.fy >= 0])


@dataclass
class Roi:
    X: np.array
    Z: np.array


class Parameters:
    @staticmethod
    def calc(obj: profile.Profile, rem: remover.Remover = None, fil: filter.Filter = None, bplt=False):
        """
        Calculates the roughness parameters of a profile

        Parameters
        ----------
        obj: profile.Profile
            The profile on which the parameters are calculates
        rem: remover.Remover
            - if None, the low frequency components are not filtered
        fil: filter.Filter
            The filter that is applied before the calculations. The cutoff
            of the filter is used to select the central region of the profile,
            half cutof is not condsidered at the adges of the rofile
        bplt: bool
            If true plots the profile after the pre-proocessing

        Returns
        -------
        RA, RQ, RP, RV, RZ, RSK, RKU: (float, ...)
            Calculated roughness parameters
        """
        border = 0

        if rem is not None:
            rem.applyRemover(obj)
        if fil is not None:
            fil.applyFilter(obj)
            cutoff = fil.cutoff
            nsample_cutoff = cutoff / (np.max(obj.X) / np.size(obj.X))
            border = nsample_cutoff / 2

        roi = Roi(obj.X[border: -border], obj.Z[border: -border])

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(obj.X0, obj.Z0, obj.X, obj.Z, alpha=0.5)
            ax.plot(roi.X, roi.Z)

        RA = np.sum(abs(roi.Z)) / np.size(roi.Z)
        RQ = np.sqrt(np.sum(abs(roi.Z ** 2)) / np.size(roi.Z))
        RP = abs(np.max(roi.Z))
        RV = abs(np.min(roi.Z))
        RT = RP + RV
        RSK = (np.sum(roi.Z ** 3) / np.size(roi.Z)) / (RQ ** 3)
        RKU = (np.sum(roi.Z ** 4) / np.size(roi.Z)) / (RQ ** 4)
        return RA, RQ, RP, RV, RT, RSK, RKU
