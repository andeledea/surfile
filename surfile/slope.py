"""
'surfile.slope'
- analysis of slope distribution for:
    - Surfaces

@author: Andrea Giura
"""
from matplotlib import pyplot as plt

from surfile import surface, funct

import numpy as np
from dataclasses import dataclass
from alive_progress import alive_bar


@dataclass
class Triangle:
    A: np.array
    B: np.array
    C: np.array

    def triangleNormal(self):
        u = self.B - self.A
        v = self.C - self.B

        n = np.cross(u, v)
        return n


def __makeFacesVectorized1(shape):
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


def slopeDistribution(obj: surface.Surface, equalSpacing=True, theta_bins=90, phi_bins=360, bplt=False):
    """
    Calculates the slope distribution in angles theta and phi

    Parameters
    ----------
    obj: surface.Surface
        The surface n wich the slope distribution is calculated
    equalSpacing: bool
        If true the method assumes equal spacing along x-axis (dx) and equal spacing along
        y-axis, this speeds up a lot the calculation
        If false the method uses a generalized triangle approach which takes longer to process
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

    ind = __makeFacesVectorized1(obj.Z.shape)

    def triangleMethod():
        nor = []
        with alive_bar(len(ind), force_tty=True,
                       title='Triangles', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for i1, i2, i3 in ind:
                t = Triangle(np.array([x[i1], y[i1], z[i1]]),
                             np.array([x[i2], y[i2], z[i2]]),
                             np.array([x[i3], y[i3], z[i3]]))

                nor.append(t.triangleNormal())

                bar()
        return np.array(nor)

    def fastMethod():
        dx, dy = np.max(obj.x) / np.size(obj.x), np.max(obj.y) / np.size(obj.y)

        eve_fun = lambda a: np.array([(z[a[0]] - z[a[1]]) * dy, (z[a[0]] - z[a[2]]) * dx, dx * dy])
        odd_fun = lambda a: np.array([(z[a[2]] - z[a[1]]) * dy, (z[a[0]] - z[a[1]]) * dx, dx * dy])

        ind_eve = ind[::2]
        n_eve = np.apply_along_axis(eve_fun, 1, ind_eve)

        ind_odd = ind[1::2]
        n_odd = np.apply_along_axis(odd_fun, 1, ind_odd)

        return np.vstack((n_eve, n_odd))

    if equalSpacing:
        normals = fastMethod()
    else:
        normals = triangleMethod()

    thetas, phis = __appendSpherical_np(normals)

    hist_theta, edges_theta = np.histogram(thetas, bins=theta_bins)
    hist_phi, edges_phi = np.histogram(phis, bins=phi_bins)

    if bplt:
        fig = plt.figure()
        (ax_ht, bx_ht) = fig.subplots(nrows=2, ncols=1)
        ax_ht.hist(edges_theta[:-1], bins=edges_theta, weights=hist_theta / len(ind) * 100, color='darkturquoise')
        bx_ht.hist(edges_phi[:-1], bins=edges_phi, weights=hist_phi / len(ind) * 100, color='darkturquoise')
        funct.persFig(
            [ax_ht, bx_ht],
            gridcol='grey',
            xlab='slope distribution [deg]',
            ylab='percentage [%]'
        )
        ax_ht.set_title(obj.name + ' Theta')
        bx_ht.set_title(obj.name + ' Phi')
        plt.show()

        return (hist_theta, edges_theta), (hist_phi, edges_phi)
