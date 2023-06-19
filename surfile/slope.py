"""
'surfile.slope'
- analysis of slope distribution for:
    - Surfaces

@author: Andrea Giura
"""
from matplotlib import pyplot as plt

from surfile import surface, funct

import numpy as np
import scipy.stats as st


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


def slopeDistribution(obj: surface.Surface, structured=False, theta_bins=90, phi_bins=360, bplt=False):
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
    theta_bins: int
        The number of bins for the Theta distribution
    phi_bins: int
        The number of bins for the Phi distribution
    bplt: bool
        If True plots the calculated histograms

    Returns
    -------
    (hist_theta, edges_theta): tuple
        The theta histogram
    (hist_phi, edges_phi): tuple
        The phi histogram
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
        plt.show()

    return (hist_theta, edges_theta), (hist_phi, edges_phi)
