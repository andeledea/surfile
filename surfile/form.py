from alive_progress import alive_bar
from matplotlib import cm
from scipy import ndimage, sparse, special, signal, integrate
from abc import ABC, abstractmethod
import numpy as np

from surfile import profile, surface, funct

import matplotlib.pyplot as plt


class Form(ABC):
    @staticmethod
    def plotForm(x, z, coeff):
        form = np.polyval(coeff, x)
        fig, ax = plt.subplots()
        ax.plot(x, z, x, form)
        ax.set_ylim(min(z), max(z))

        plt.show()

    @staticmethod
    def removeForm(x, z, coeff):
        form = np.polyval(coeff, x)
        z_final = z - form
        return z_final

    @staticmethod
    def plot3DForm(x, y, z, coeff):
        form = np.polynomial.polynomial.polyval2d(x, y, coeff)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_surface(x, y, z, cmap=cm.Greys, alpha=0.4)
        ax.plot_surface(x, y, form, cmap=cm.rainbow)

        plt.show()

    @staticmethod
    def remove3DForm(x, y, z, coeff):
        form = np.polynomial.polynomial.polyval2d(x, y, coeff)
        z_final = z - form
        return z_final


class ProfileLSLine(Form):
    @staticmethod
    def form(obj: profile.Profile(), bplt=False):
        """
        Least square line fit implementation
        Parameters
        ----------
        obj: profile.Profile()
            The profile object on wich the form is removed
        bplt: bool
            If True plots the line overimposed on the profile
        return (m, q): (float, ...)
            The line equation coefficients
        """
        # create matrix and Z vector to use lstsq
        XZ = np.vstack([obj.X.reshape(np.size(obj.X)),
                        obj.Z.reshape(np.size(obj.Z))]).T
        (rows, cols) = XZ.shape
        G = np.ones((rows, 2))
        G[:, 0] = XZ[:, 0]  # X
        Z = XZ[:, 1]
        (m, q), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)  # calculate LS plane

        if bplt:
            Form.plotForm(obj.X, obj.Z, [m, q])

        obj.Z = Form.removeForm(obj.X, obj.Z, [m, q])
        return m, q


class ProfilePolynomial(Form):
    @staticmethod
    def form(obj: profile.Profile(), degree, comp=lambda a, b: a < b, bound=None, bplt=False):
        # TODO: add a cutter in the arguments??
        """
        Plynomial fit implementation
        Parameters
        ----------
        obj: profile.Profile()
            The profile object on wich the form is removed
        comp: funct / lambda
            comparison method between bound and profile
        degree: int
            polynomial degree
        bound: float
            if not set the fit uses all points,
            if set the fit uses all points below the values,
            if set to True the fit uses only the values below the average value of the profile
        bplt: bool
            If True plots the polynomial form overimposed on the profile
        return coeff: np.array()
            The polynomial form coefficients
        """
        if bound is None:
            coeff = np.polyfit(obj.X, obj.Z, degree)
        elif bound is True:
            bound = np.mean(obj.Z)
            ind = np.argwhere(comp(obj.Z, bound)).ravel()
            coeff = np.polyfit(obj.X[ind], obj.Z[ind], degree)
        else:
            ind = np.argwhere(comp(obj.Z, bound)).ravel()
            coeff = np.polyfit(obj.X[ind], obj.Z[ind], degree)

        if bplt:
            Form.plotForm(obj.X, obj.Z, coeff)

        obj.Z = Form.removeForm(obj.X, obj.Z, coeff)
        return coeff


class ProfileHistogram(Form):  # Still not working well
    @staticmethod
    def form(obj: profile.Profile(), final_m, bplt=False):
        """
        Alligns the profile using the recursive histogram method, stops when
        the tilt correction is less than the parameter final_m
        Parameters
        ----------
        obj: profile.Profile()
            The profile object on wich the form is removed
        final_m: float
            The end slope correction
        bplt: bool
            Plots the histogram evolution over the iterations
        """
        m, q = obj.fitLineLS()  # preprocess inclination
        tot_bins = int(np.size(obj.X) / 20)
        # threshold = 50 / tot_bins

        line_m = m / 10  # start incline
        print(f'Hist method -> Start slope  {line_m}')

        fig = plt.figure()
        ax_h = fig.add_subplot(211)
        bx_h = fig.add_subplot(212)

        def calcNBUT():
            hist, edges = obj.histMethod(bins=tot_bins, bplt=False)  # make the hist
            weights = hist / np.size(obj.Z) * 100
            threshold = np.max(weights) / 20
            n_bins_under_threshold = np.size(np.where(weights < threshold)[0])  # how many bins under th

            if bplt:
                ax_h.clear()
                ax_h.hist(edges[:-1], bins=edges, weights=weights, color='red')
                ax_h.plot(edges[:-1], integrate.cumtrapz(hist / np.size(obj.Z) * 100, edges[:-1], initial=0))
                ax_h.text(.25, .75, f'NBUT = {n_bins_under_threshold} / {tot_bins}, line_m = {line_m:.3f} -> {final_m}',
                          horizontalalignment='left', verticalalignment='bottom', transform=ax_h.transAxes)
                ax_h.axhline(y=threshold, color='b')

                bx_h.clear()
                bx_h.plot(obj.X, obj.Z)
                plt.draw()
                plt.pause(0.05)

            return n_bins_under_threshold

        param = calcNBUT()
        n_row = 0
        # until I have enough bins < th keep loop
        while np.abs(line_m) > final_m:  # nbut < (tot_bins - tot_bins / 20):
            obj.Z = obj.Z - obj.X * line_m
            param_old = param
            param = calcNBUT()

            if param < param_old:  # invert rotation if we are going the wrong way
                line_m = -line_m / 2

            if param == param_old:
                n_row += 1
                if n_row >= 15: break  # we got stuck for too long
            else:
                n_row = 0
        print(f'Hist method -> End slope {line_m}')


class SurfaceLSPlane(Form):
    @staticmethod
    def form(obj: surface.Surface(), bplt=False):
        """
        Least square plane fit implementation
        Parameters
        ----------
        obj: surface.Surface()
            The surface object on wich the LS plane is applied
        bplt: bool
            If True plots the plane overimposed on the surface
        return sol: np.ndarray
            Array of polynomial coefficients.
        """
        # create matrix and Z vector to use lstsq
        XYZ = np.vstack([obj.X.reshape(np.size(obj.X)),
                         obj.Y.reshape(np.size(obj.Y)),
                         obj.Z.reshape(np.size(obj.Z))]).T
        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  # X
        G[:, 1] = XYZ[:, 1]  # Y
        Z = XYZ[:, 2]
        sol, resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)  # calculate LS plane

        if bplt:
            Form.plot3DForm(obj.X, obj.Y, obj.Z, sol.reshape((2, 2)))

        Form.remove3DForm(obj.X, obj.Y, obj.Z, sol.reshape((2, 2)))
        return sol


class SurfacePolynomial(Form):
    @staticmethod
    def form(obj: surface.Surface(), kx=3, ky=3, comp=lambda a, b: a < b, bound=None, order=None, bplt=False):
        # TODO: add a cutter??
        """
        Least square polynomial fit implementation
        Parameters
        ----------
        obj: surface.Surface()
            The surface object on wich the polynomial fit is applied
        kx, ky: int
            Polynomial order in x and y, respectively.
        comp: funct / lambda
            comparison method between bound and profile
        bound: float
            if not set the fit uses all points,
            if set the fit uses all points below the values,
            if set to True the fit uses only the values below the average value of the surface
        order: int or None, default is None
            If None, all coefficients up to maxiumum kx, ky, i.e. up to and including x^kx*y^ky, are considered.
            If int, coefficients up to a maximum of kx+ky <= order are considered.
        bplt: bool
            If True plots the plane overimposed on the surface
        return sol: np.ndarray
            Array of polynomial coefficients.
        """
        if bound is True:
            bound = np.mean(obj.Z)  # set the bound to the mean point

        # coefficient array, up to x^kx, y^ky
        coeffs = np.ones((kx + 1, ky + 1))

        # solve array
        a = np.zeros((coeffs.size, obj.X.size))

        # for each coefficient produce array x^i, y^j
        for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
            # do not include powers greater than order
            if order is not None and i + j > order:
                arr = np.zeros_like(obj.X)
            else:
                arr = coeffs[i, j] * obj.X ** i * obj.Y ** j
            a[index] = arr.ravel()

        z = obj.Z
        if bound is not None:
            where = np.argwhere(comp(z.reshape(np.size(z)), bound))
            z = np.delete(z, where, 0)
            a = np.delete(a, where, 1)  # check if 1 is correct

        # do leastsq fitting and return leastsq result
        sol, resid, rank, s = np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

        if bplt:
            Form.plot3DForm(obj.X, obj.Y, obj.Z, sol.reshape((kx+1, ky+1)))

        Form.remove3DForm(obj.X, obj.Y, obj.Z, sol.reshape((kx+1, ky+1)))
        return sol
