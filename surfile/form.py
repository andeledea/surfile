import itertools

from matplotlib import cm
from scipy import integrate
import numpy as np

from surfile import profile, surface, cutter as cutr

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector


def polyval2d(x, y, coeffs):
    # https://sofia-usra.github.io/sofia_redux/_modules/sofia_redux/toolkit/fitting/polynomial.html#poly2d
    """
    Evaluate 2D polynomial coefficients
    ONLY USED INTERNALLY FOR SURFACE RECONSTRUCTION

    Parameters
    ----------
    x : array_like of float
        (shape1) x-coordinate independent interpolants
    y : array_like of float
        (shape1) y-coordinate independent interpolants
    coeffs : numpy.ndarray
        (y_order + 1, x_order + 1) array of coefficients output by
        `polyfit2d`.

    Returns
    -------
    numpy.ndarray
        (shape1) polynomial coefficients evaluated at (y,x).
    """
    s = x.shape
    coeffs = np.array(coeffs)
    ny, nx = coeffs.shape
    z = np.zeros(s)
    for c, (j, i) in zip(coeffs.ravel(), itertools.product(range(ny), range(nx))):
        if c == 0:
            continue
        z += c * (x ** i) * (y ** j)
    return z


class Form:
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
        form = polyval2d(x, y, coeff)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_surface(x, y, z, cmap=cm.Greys, alpha=0.4)
        ax.plot_surface(x, y, form, cmap=cm.rainbow)

        plt.show()

    @staticmethod
    def remove3DForm(x, y, z, coeff):
        form = polyval2d(x, y, coeff)
        z_final = z - form
        return z_final


class ProfileLSLine(Form):
    @staticmethod
    def form(obj: profile.Profile, bplt=False):
        """
        Least square line fit implementation

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the form is removed
        bplt: bool
            If True plots the line overimposed on the profile

        Returns
        ----------
        (m, q): (float, ...)
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
    def form(obj: profile.Profile, degree, comp=lambda a, b: a < b, bound=None, cutter=None, bplt=False):
        """
        Plynomial fit implementation

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the form is removed
        comp: funct / lambda
            comparison method between bound and profile
        degree: int
            polynomial degree
        bound: float
            -if not set the fit uses all points,
            -if set the fit uses all points below the values,
            -if set to True the fit uses only the values below the average value of the profile
        cutter: cutter.Cutter
            -if not set, the fit uses all points
            -if true allows the user to select manually the region of interest
            -if a cutter obj is passed the fit is done only on the cutted profile points
             and then applied on the whole profile
        bplt: bool
            If True plots the polynomial form overimposed on the profile

        Returns
        ----------
        coeff: np.array()
            The polynomial form coefficients
        """

        if cutter is True:  # if the fit is only on part of the profile the user chooses
            _, (x, z) = cutr.ProfileCutter.cut(obj, finalize=False)
        elif cutter is not None:
            x, z = cutter.applyCut(obj, finalize=False)
        else:  # use the whole profile
            x = obj.X
            z = obj.Z

        if bound is None:
            coeff = np.polyfit(x, z, degree)
        elif bound is True:
            bound = np.mean(z)
            ind = np.argwhere(comp(z, bound)).ravel()
            coeff = np.polyfit(x[ind], x[ind], degree)
        else:
            ind = np.argwhere(comp(z, bound)).ravel()
            coeff = np.polyfit(x[ind], z[ind], degree)

        if bplt:
            Form.plotForm(obj.X, obj.Z, coeff)

        obj.Z = Form.removeForm(obj.X, obj.Z, coeff)
        return coeff


# TODO
class ProfileHistogram(Form):  # Still not working well
    @staticmethod
    def form(obj: profile.Profile, final_m, bplt=False):
        """
        Alligns the profile using the recursive histogram method, stops when
        the tilt correction is less than the parameter final_m

        Parameters
        ----------
        obj: profile.Profile
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
    def form(obj: surface.Surface, bplt=False):
        """
        Least square plane fit implementation

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the LS plane is applied
        bplt: bool
            If True plots the plane overimposed on the surface

        Returns
        ----------
        sol: np.ndarray
            Array of polynomial coefficients.
        """

        return SurfacePolynomial.form(obj, kx=1, ky=1, bplt=bplt)


class SurfacePolynomial(Form):
    @staticmethod
    def form(obj: surface.Surface, kx=3, ky=3, full=False, comp=lambda a, b: a < b, bound=None, cutter=None, bplt=False):
        """
        Least square polynomial fit implementation

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the polynomial fit is applied
        kx, ky: int
            Polynomial order in x and y, respectively.
        full : bool, optional
            If True, will solve using the full polynomial matrix.  Otherwise,
            will use the upper-left triangle of the matrix.  See
            `polyinterp2d` for further details.  Note that if kx != ky, then
            the full matrix will be solved for. (@sofia_redux/toolkit)
        comp: funct / lambda, optional
            comparison method between bound and surface
        bound: float, optional
            if not set the fit uses all points,
            if set the fit uses all points below the values,
            if set to True the fit uses only the values below the average value of the surface
        cutter: cutter.Cutter
            -if not set, the fit uses all points
            -if true allows the user to select manually the region of interest
            -if a cutter obj is passed the fit is done only on the cutted profile points
             and then applied on the whole profile
        bplt: bool, optional
            If True plots the plane overimposed on the surface

        Returns
        ----------
        sol: np.ndarray
            Array of polynomial coefficients.
        """
        if cutter is True:
            _, (x, y, z) = cutr.SurfaceCutter.cut(obj, finalize=False)
        elif cutter is not None:
            x, y, z = cutter.applyCut(obj, finalize=False)
        else:
            x, y, z = obj.X, obj.Y, obj.Z

        x, y, z,  = x.ravel(), y.ravel(), z.ravel(),

        if bound is True:
            bound = np.mean(obj.Z)  # set the bound to the mean point

        nx, ny = kx + 1, ky + 1
        full |= kx != ky  # if they are different -> full matrix
        loop = list(itertools.product(range(ny), range(nx)))

        if not full:  # calculate only upper left part of matrix
            loop2 = []
            for (j, i) in loop:
                if i <= (ky - j):
                    loop2.append((j, i))
            loop = loop2
        a = np.zeros((x.size, len(loop)))
        for k, (j, i) in enumerate(loop):
            a[:, k] = (x ** i) * (y ** j)

        # remove nan values
        indexes = ~np.isnan(z)
        z = z[indexes]
        a = a[indexes]

        if bound is not None:  # remove z limited values in comp direction
            where = np.argwhere(comp(z.reshape(np.size(z)), bound))
            z = np.delete(z, where, 0)
            a = np.delete(a, where, 1)  # check if 1 is correct

        sol, _, _, _ = np.linalg.lstsq(a, z, rcond=None)

        if full:
            coeffs = sol.reshape((ny, nx))
        else:
            coeffs = np.zeros((ny, nx))
            for c, (j, i) in zip(sol, loop):
                coeffs[j, i] = c

        if bplt:
            Form.plot3DForm(obj.X, obj.Y, obj.Z, coeffs)

        obj.Z = Form.remove3DForm(obj.X, obj.Y, obj.Z, coeffs)
        return sol


class Surface3Points(Form):
    @staticmethod
    def form(obj: surface.Surface, bplt=False):
        """
        3 points plane fit implementation
        Opens a plot figure to choose the 3 points and fids the plane for those points

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the polynomial fit is applied
        bplt: bool
            If True plots the plane overimposed on the surface
        """
        def onClose(event):  # when fig is closed calculate plane parameters
            po = []
            for (a, b) in selector.verts:
                xind = np.where(obj.X[0, :] <= a)[0][-1]
                yind = np.where(obj.Y[:, 0] <= b)[0][-1]

                po.append([a, b, obj.Z[yind, xind]])

            print(f"Collected points: {po}")
            a1 = po[1][0] - po[0][0]  # x2 - x1;
            b1 = po[1][1] - po[0][1]  # y2 - y1;
            c1 = po[1][2] - po[0][2]  # z2 - z1;
            a2 = po[2][0] - po[0][0]  # x3 - x1;
            b2 = po[2][1] - po[0][1]  # y3 - y1;
            c2 = po[2][2] - po[0][2]  # z3 - z1;
            a = b1 * c2 - b2 * c1
            b = a2 * c1 - a1 * c2
            c = a1 * b2 - b1 * a2
            d = 0  # (- self.a * po[0][0] - self.b * po[0][1] - self.c * po[0][2])

            sol = np.array([-d/c, -a/c, -b/c, 0]).reshape((2, 2))

            if bplt:
                Form.plot3DForm(obj.X, obj.Y, obj.Z, sol)

            obj.Z = Form.remove3DForm(obj.X, obj.Y, obj.Z, sol)
            plt.close(fig)

        fig, ax = plt.subplots()
        ax.pcolormesh(obj.X, obj.Y, obj.Z, cmap=cm.rainbow)
        ax.set_title('3 Points plane fit')
        fig.canvas.mpl_connect('close_event', onClose)

        selector = PolygonSelector(ax, lambda *args: None)

        plt.show()


class sphere(Form):
    @staticmethod
    def form(obj: surface.Surface, finalize=True,  bplt=False):
        """
        Calculates the least square sphere

        Parameters
        ----------
        obj: surface.Surface
            The surface object on wich the polynomial fit is applied
        finalize: bool
            If set to False the fit will not alter the surface,
            the method will only return the center and the radius
        bplt: bool
            Plots the sphere fitted to the data points

        Returns
        ----------
        (radius, C): (float, [xc, yc, zc])
            Radius and sphere center coordinates
        """
        #   Assemble the A matrix
        spZ = obj.Z.flatten()
        spX = obj.X.flatten()
        spY = obj.Y.flatten()

        nanind = ~np.isnan(spZ)

        spZ = spZ[nanind]
        spX = spX[nanind]
        spY = spY[nanind]

        A = np.zeros((len(spX), 4))
        A[:, 0] = spX * 2
        A[:, 1] = spY * 2
        A[:, 2] = spZ * 2
        A[:, 3] = 1

        #   Assemble the f matrix
        f = np.zeros((len(spX), 1))
        f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
        C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)

        #   solve for the radius
        t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
        radius = np.sqrt(t)

        if finalize:
            sph = np.sqrt(t - (obj.X-C[0])**2 - (obj.Y-C[1])**2) + C[2]
            obj.Z = obj.Z - sph

        if bplt:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = np.cos(u) * np.sin(v) * radius
            y = np.sin(u) * np.sin(v) * radius
            z = np.cos(v) * radius
            x = x + C[0]
            y = y + C[1]
            z = z + C[2]

            #   3D plot of Sphere bfv
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(obj.X, obj.Y, obj.Z, cmap=cm.rainbow)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.2)
            plt.show()

        return radius[0], C

