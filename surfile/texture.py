"""
'surfile.roughness'
- analysis of roughness features for:
    - Profiles
    - Surfaces

Example
-------
>>> psd = texture.Psd()
>>> psd.evalPsd(sur, bplt=False)
>>> avg = psd.averageSpectra(bplt=False)
>>> psd.angleIntegratedSpectra(dt_fct=0.8, bplt=True)
>>> psd.polarSpectra(df_fct=0.8, bplt=True)

@author: Dorothee Hueser, Andrea Giura
"""

from matplotlib import cm
import numpy as np
import scipy.stats as st
from dataclasses import dataclass

from surfile import geometry, profile, surface, filter, funct

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
    """
    Class that provides the data structure and methods for PSD
    calculation.
    """
    def __init__(self):
        """Instantiate an empty psd object"""
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
            
        Notes
        -----
        The power spectral density (PSD) function gives indication on  the strength 
        of different frequency components  as function of the spatial frequency and 
        is calculated from the absolute value of the square of the Fourier transform 
        of the height values z(x,y).
        The areal Fourier transform is given by:\n
        $
        \\begin{equation}
        F_z(f_x,f_y)=\\lim_{L_x \\to \\infty}\\lim_{Ly \\to \\infty} \\frac{1}{L_x L_y}\int_{-L_y/2}^{L_y/2}\\int_{-L_x/2}^{L_x/2}z(x,y)e^{[-i2\\pi(f_xx+f_yy)]}dxdy
        \\end{equation}
        $\n
        The Power Spectral Density is then calculated as:\n
        $
        \\begin{equation}
            P(f_x,f_y)=\\frac{d}{df_x}\\frac{d}{df_y}F_z^*(f_x,f_y)F_z(f_x,f_y)
        \\end{equation}
        $
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
        Calculate the spectra in polar coordinates

        Parameters
        ----------
        df_fct: float
            The frequency resolution
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
        fr = np.linspace(0.1 * fr0, frmax, 512)
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
        Calculate the angle integrated spectra

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
        if self.psd is None: raise Exception('Angle integrated spectra failed: psd has not been evaluated')

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
            
            plt.show()

        return (self.fx[self.fx >= 0], PSDxmean[self.fx >= 0]), (self.fy[self.fy >= 0], PSDymean[self.fy >= 0])


@dataclass
class Roi:
    X: np.array
    Z: np.array


class Parameters:
    @staticmethod
    def calc(obj: profile.Profile, rem: geometry.FormEstimator = None, fil: filter.Filter = None, bplt=False):
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
        border = 1

        if rem is not None:
            rem.applyFit(obj)
        if fil is not None:
            fil.applyFilter(obj, bplt=False)
            cutoff = fil.cutoff
            nsample_cutoff = cutoff // (np.nanmax(obj.X) / np.size(obj.X))
            border = int(nsample_cutoff // 2)

        roi = Roi(obj.X[border: -border], obj.Z[border: -border])
        # print(roi.X, roi.Z)

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(obj.X, obj.Z, alpha=0.5)
            ax.plot(roi.X, roi.Z)
            plt.show()

        RA = np.nansum(abs(roi.Z)) / np.size(roi.Z)
        RQ = np.sqrt(np.nansum(abs(roi.Z ** 2)) / np.size(roi.Z))
        RP = abs(np.nanmax(roi.Z))
        RV = abs(np.nanmin(roi.Z))
        RT = RP + RV
        RSK = (np.nansum(roi.Z ** 3) / np.size(roi.Z)) / (RQ ** 3)
        RKU = (np.nansum(roi.Z ** 4) / np.size(roi.Z)) / (RQ ** 4)
        return RA, RQ, RP, RV, RT, RSK, RKU


def __makeFacesVectorized(shape):
    Nr = shape[0]
    Nc = shape[1]

    out = np.empty((Nr - 1, Nc - 1, 2, 3), dtype=int)

    r = np.arange(Nr * Nc).reshape(Nr, Nc)

    out[:, :, 0, 0] = r[:-1, :-1]
    out[:, :, 1, 0] = r[:-1, 1:]
    out[:, :, 0, 1] = r[:-1, 1:]

    out[:, :, 1, 1] = r[1:, 1:]
    out[:, :, :, 2] = r[1:, :-1, None]

    out.shape = (-1, 3)
    return out


def __appendSpherical_np(xyz):
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    # rs = np.sqrt(xy + xyz[:, 2] ** 2)
    thetas = np.rad2deg(np.arctan2(np.sqrt(xy), xyz[:, 2]))  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    phis = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    phis[phis < 0] += 360
    return thetas, phis


@funct.options(bplt=True, csvPath='out\\')
def slopeDistribution(obj: surface.Surface, structured=False, theta_res=1, phi_res=1, adaptive_hist=False, bplt=False):
    """
    Calculates the slope distribution in angles theta and phi

    Parameters
    ----------
    obj: surface.Surface
        The surface n wich the slope distribution is calculated
    structured: bool
        If true the method assumes equal spacing along x-axis (dx) and equal spacing along
        y-axis
        If false the method uses a generalized triangle approach (recommended)
    theta_res: float
        The angle resolution for the theta angle
    phi_res: float
        The angle resolution for the phi angle
    adaptive_hist: bool
        If true the bins are calculated between 0 and the max angles
        If false the bins are calculated between 0 - 90 for theta, 0 - 360 for phi
    bplt: bool
        If True plots the calculated histograms

    Returns
    -------
    (hist_theta, edges_theta): tuple
        The theta histogram
    (hist_phi, edges_phi): tuple
        The phi histogram
        
    Notes
    -----
    This method performs a slope distribution analysis on the surface by calculating all 
    the normal vectors to the triangles constructed on the adjacent points of the topography. 
    The normal vectors are then expressed in polar coordinates and for the two angles 
    (elevation, azimuth) a histogram plot is generated. The number of bins for each angle can 
    be specified by the user by setting the two parameters res and adaptive_hist. 
    """
    x = obj.X.ravel()
    y = obj.Y.ravel()
    z = obj.Z.ravel()

    ind = __makeFacesVectorized(obj.Z.shape)

    def unstructuredMesh():
        A = np.array([x[ind[:, 0]], y[ind[:, 0]], z[ind[:, 0]]]).T
        B = np.array([x[ind[:, 1]], y[ind[:, 1]], z[ind[:, 1]]]).T
        C = np.array([x[ind[:, 2]], y[ind[:, 2]], z[ind[:, 2]]]).T

        u = B - A
        v = C - B

        return np.cross(u, v)

    def structuredMesh():
        dx, dy = np.max(obj.x) / np.size(obj.x), np.max(obj.y) / np.size(obj.y)

        eve_fun = lambda a: np.array([(z[a[0]] - z[a[1]]) * dy, (z[a[0]] - z[a[2]]) * dx, dx * dy])
        odd_fun = lambda a: np.array([(z[a[2]] - z[a[1]]) * dy, (z[a[0]] - z[a[1]]) * dx, dx * dy])

        ind_eve = ind[::2]
        n_eve = np.apply_along_axis(eve_fun, 1, ind_eve)

        ind_odd = ind[1::2]
        n_odd = np.apply_along_axis(odd_fun, 1, ind_odd)

        return np.vstack((n_eve, n_odd))

    if structured: normals = structuredMesh()
    else:          normals = unstructuredMesh()

    thetas, phis = __appendSpherical_np(normals)

    print(f'Slope Distribution:\nTheta:\t{st.describe(thetas)}\nPhi:\t{st.describe(phis)}')

    if not adaptive_hist:
        max_theta = 90
        max_phi = 360
        thetas = np.append(thetas, max_theta)  # add only one value to max the space
        phis = np.append(phis, max_phi)

    else:
        max_theta = np.max(thetas)
        max_phi = np.max(phis)

    theta_bins = int(np.ceil(max_theta / theta_res))
    phi_bins = int(np.ceil(max_phi / phi_res))

    hist_theta, edges_theta = np.histogram(thetas, bins=theta_bins)
    hist_phi, edges_phi = np.histogram(phis, bins=phi_bins)

    if bplt:
        fig = plt.figure()
        (ax_ht, bx_ht) = fig.subplots(nrows=2, ncols=1)
        ax_ht.hist(edges_theta[:-1], bins=edges_theta, weights=hist_theta / len(ind) * 100, color='darksalmon')
        bx_ht.hist(edges_phi[:-1], bins=edges_phi, weights=hist_phi / len(ind) * 100, color='darkturquoise')
        funct.persFig(
            [ax_ht, bx_ht],
            gridcol='grey',
            xlab='slope distribution [deg]',
            ylab='percentage [%]'
        )
        ax_ht.set_title(obj.name + ' Theta')
        bx_ht.set_title(obj.name + ' Phi')

        plt.figure()
        cx, dx = plt.subplot(121, polar=True), plt.subplot(122, polar=True)
        cx.bar(np.deg2rad(edges_theta[:-1]), hist_theta / max(hist_theta), width=(2*np.pi) / theta_bins,
               bottom=0, color='darksalmon', alpha=0.5, label='Theta')
        dx.bar(np.deg2rad(edges_phi[:-1]), hist_phi / max(hist_phi), width=(2 * np.pi) / phi_bins,
               bottom=0, color='darkturquoise', alpha=0.5, label='Phi')
        cx.legend()
        dx.legend()
        cx.set_title(obj.name)
        # plt.show()

    return (edges_theta[:-1], hist_theta / len(ind) * 100), (edges_phi[:-1], hist_phi / len(ind) * 100)
