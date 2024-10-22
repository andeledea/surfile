"""
'surfile.geometry'
- form fit operations, implements:
    - circle fit
    - least square polynomial fits with bounded domain
    - 3 point plane
    - spherical fit
    - cylinder fit
    
Notes
-----
Measurements made by microscopy techniques involve sample loading,
and specifically profilometers involve the housing of samples above a tilt x-y table.
Before carrying out the measurements, the samples must be levelled with respect to 
the instrument plane, but since the images recorded have distortions mainly due 
to misalignments (since the plane which contains the sample surface is not perfectly 
parallel to the plane of the image), it is therefore essential to implement methods 
for image levelling.  Moreover, other image distortions which can degrade the quality 
of the surface reconstruction are (i) bow, which appears as a false curvature superposed 
on the real sample topography, and (ii) edge effects, that can enlarge or shrink 
features present in the image borders.

@author: Andrea Giura
"""

import itertools

from matplotlib import cm
from scipy import integrate, optimize
import numpy as np
import circle_fit

from surfile import funct, profile, surface, cutter as cutr

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector


def _polyval2d(x, y, coeffs):
    """
    Evaluate 2D polynomial coefficients
    Private method: ONLY USED INTERNALLY FOR FORM RECONSTRUCTION
    
    # https://sofia-usra.github.io/sofia_redux/license.html

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
    z : numpy.ndarray
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


class FormEstimator:
    """
    Base class for form fitting operations contains the methods 
    common to all the derived classes:
    - Profile Line
    - Profile Poly
    - Circle
    - Surface Plane
    - Surface Poly
    - Surface 3 Points
    - Sphere
    - Cylinder
    
    Notes
    -----
    The form or geometry parameters are estimated by minimizing the sum of the squares
    of the residuals: $\\varepsilon_i \\; = \\; f(\\mathbf{p}, x_i, y_i, z_i)$
    where $(x_i, y_i)$ are the sampling positions, $z_i = z(x_i, y_i)$ are the height values 
    of the topography map, $f$ is the model function, and $\mathbf{p} = [p_0, p_1, \\dots, p_m]$ 
    is the tuple of geometry parameters or polynomial coefficients to be estimated by optimization.
    
    $\\min_{\\mathbf{p}} \\{\\sum\\limits_{i = 0}^{n-1}  f(\\mathbf{p}, x_i, y_i, z_i)^2 \\}$
    """
    def applyFit(self, obj, bplt=False):
        """
        Applies the fit operation to the obj with the
        parameters of the self object

        Parameters
        ----------
        obj : surfile.container
            The data structure
        bplt : bool, optional
            If true plots the removed form, by default False
        """
        pass

    @staticmethod
    def plotForm(x, z, coeff):
        """
        Plots the fitted 1d function and the profile

        Parameters
        ----------
        x : np.array
            The x values of the profile
        z : np.array
            The z values of the profile
        coeff : np.array
            The coefficients of the polynomial fit
        """
        form = np.polyval(coeff, x)
        fig, ax = plt.subplots()
        ax.plot(x, z, x, form)
        ax.set_ylim(np.nanmin(z), np.nanmax(z))

        plt.show()

    @staticmethod
    def removeForm(x, z, coeff):
        """
        Calculates the polynomial function given the
        coefficients and subtracts the form from the profile

        Parameters
        ----------
        x : np.array
            The x values of the profile
        z : np.array
            The z values of the profile
        coeff : np.array
            The coefficients of the polynomial fit

        Returns
        -------
        z_final : np.array
            The final z values of the profile after the form is removed
        """
        form = np.polyval(coeff, x)
        z_final = z - form
        return z_final

    @staticmethod
    def plot3DForm(x, y, z, coeff):
        """
        Plots the fitted 2d function and the surface

        Parameters
        ----------
        x : np.array
            The x values of the surface
        y : np.array
            The y values of the surface
        z : np.array
            The z values of the surface
        coeff : np.array
            The coefficients of the polynomial fit
        """
        form = _polyval2d(x, y, coeff)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot_surface(x, y, z, cmap=cm.Greys, alpha=0.7)
        ax.plot_surface(x, y, form, cmap=cm.rainbow)

        plt.show()

    @staticmethod
    def remove3DForm(x, y, z, coeff):
        """
        Removes the fitted 2d function from the surface

        Parameters
        ----------
        x : np.array
            The x values of the surface
        y : np.array
            The y values of the surface
        z : np.array
            The z values of the surface
        coeff : np.array
            The coefficients of the polynomial fit
            
        Returns
        -------
        z_final : np.array
            The final z values of the surface after the form is removed
        """
        form = _polyval2d(x, y, coeff)
        z_final = z - form
        return z_final


class ProfileLSLine(FormEstimator):
    """
    Class derived from FormEstimator for the degree 1 line 
    fit on a profile data.
    """
    # shortcut class to profile poly of degree 1 (implemented for simmetry)
    def applyFit(self, obj: profile.Profile, bplt=False):
        return self.formFit(obj, bplt=bplt)

    @staticmethod
    def formFit(obj: profile.Profile, bplt=False):
        """
        Least square line fit implementation on a profile

        Parameters
        ----------
        obj : profile.Profile
            The profile object on wich the form is removed
        bplt : bool
            If True plots the line overimposed on the profile

        Returns
        -------
        (m, q) : (float, ...)
            The line equation coefficients
            
        Notes
        -----
        $\\min_{\\mathbf{p}} \\{\\sum\\limits_{i = 0}^{n-1}  f(\\mathbf{p}, x_i, z_i)^2 \\}$
        
        In this particular case of line fitting $f(\\mathbf{p}, x_i, z_i)=z_i-z(\\mathbf{p}, x_i)$
        where $z(\\mathbf{p}, x)=p_0 x + p_1$ and $\\mathbf{p}=[m, q]$.
        """
        # create matrix and Z vector to use lstsq
        XZ = np.vstack([obj.X.reshape(np.size(obj.X)),
                        obj.Z.reshape(np.size(obj.Z))]).T
        XZ = XZ[~np.isnan(obj.Z)]
        (rows, cols) = XZ.shape
        G = np.ones((rows, 2))
        G[:, 0] = XZ[:, 0]  # X
        Z = XZ[:, 1]
        (m, q), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)  # calculate LS line

        if bplt:
            FormEstimator.plotForm(obj.X, obj.Z, [m, q])

        obj.Z = FormEstimator.removeForm(obj.X, obj.Z, [m, q])
        return m, q


class ProfilePolynomial(FormEstimator):
    """
    Class derived from FormEstimator for the degree n polynomial 
    fit on a profile data.
    """
    def __init__(self, degree, comp=lambda a, b: a < b, bound=None, cutter=None):
        """
        Creates a ProfilePolynomial object with the parameters provided

        Parameters
        ----------
        degree: int
            polynomial degree
        comp: funct / lambda
            comparison method between bound and profile
        bound: float
            -if not set the fit uses all points,
            -if set the fit uses all points below the value,
            -if set to True the fit uses only the values below the average value of the profile
        cutter: cutter.Cutter
            -if not set, the fit uses all points
            -if true allows the user to select manually the region of interest
            -if a cutter obj is passed the fit is done only on the cutted profile points
             and then applied on the whole profile
             
        Examples
        --------
        >>> profPolyFit = ProfilePolynomial(2, bound=True, cutter=None)
        >>> texture.Parameters.calc(prf, rem=profPolyFit, bplt=True)
        """
        self.degree = degree
        self.comp = comp
        self.bound = bound
        self.cutter = cutter

    def applyFit(self, obj: profile.Profile, bplt=False):
        return self.formFit(obj, self.degree, self.comp, self.bound, self.cutter, bplt=bplt)

    @staticmethod
    def formFit(obj: profile.Profile, degree, comp=lambda a, b: a < b, bound=None, cutter=None, bplt=False):
        """
        Plynomial fit implementation on a profile using the least square method

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the form is removed
        degree: int
            polynomial degree
        comp: funct / lambda
            comparison method between bound and profile
        bound: float
            -if not set the fit uses all points,
            -if set the fit uses all points below the value,
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
            
        Notes
        -----
        $\\min_{\\mathbf{p}} \\{\\sum\\limits_{i = 0}^{n-1}  f(\\mathbf{p}, x_i, z_i)^2 \\}$
        
        In this particular case of polynomial fitting $f(\\mathbf{p}, x_i, z_i)=z_i-z(\\mathbf{p}, x_i)$
        where $z(\\mathbf{p}, x)=p_0 x^n + \\cdots + p_{n-1} x + p_n$ and $\\mathbf{p}=[p_0, \\cdots, p_n]$.
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
            FormEstimator.plotForm(obj.X, obj.Z, coeff)

        obj.Z = FormEstimator.removeForm(obj.X, obj.Z, coeff)
        return coeff


class Circle(FormEstimator):
    """
    Class derived from FormEstimator for the circle 
    fit on a profile data.
    """
    @staticmethod
    def formFit(obj: profile.Profile, cutter=None, finalize=True, bplt=False):
        """
        Circle fit implementation using the least square method

        Parameters
        ----------
        obj: profile.Profile
            The profile object on wich the form is removed
        cutter: cutter.Cutter
            -if not set, the fit uses all points
            -if true allows the user to select manually the region of interest
            -if a cutter obj is passed the fit is done only on the cutted profile points
             and then applied on the whole profile
        finalize: bool
            If set to False the fit will not alter the surface,
            the method will only return the radius and the form deviation
        bplt: bool
            Plots the sphere fitted to the data points

        Returns
        -------
        r: float
            The radius of the fitted circle
        dev: float
            The form deviation of the points
        (xc, zc) : tuple
            Centre coordinates
            
        Notes
        -----
        The method used is from circle-fit python package hyperLSQ()
        which implements the algorithm in [1]
        
        References
        ----------
        [1] Rangarajan, Prasanna & Kanatani, Kenichi & Niitsuma, Hirotaka & Sugaya, Yasuyuki. (2010). 
        Hyper Least Squares and Its Applications. 5-8. 10.1109/ICPR.2010.10.
        """
        if cutter is True:  # if the fit is only on part of the profile the user chooses
            _, (x, z) = cutr.ProfileCutter.cut(obj, finalize=False)
        elif cutter is not None:
            x, z = cutter.applyCut(obj, finalize=False)
        else:  # use the whole profile
            x = obj.X
            z = obj.Z

        cords = np.vstack((x, z)).T
        xc, zc, r, dev = (circle_fit.hyperLSQ(cords))

        if bplt:
            fig, ax = plt.subplots()
            ax.axis('equal')
            ax.plot(obj.X, obj.Z, 'r')
            circle = plt.Circle((xc, zc), r, alpha=0.6, fill=False)
            ax.add_patch(circle)

            funct.persFig([ax], 'x [um]', 'y [um]')

            ax.set_title(obj.name)
            plt.show()

        if finalize:
            if np.mean(obj.Z) > zc:  # convex case
                cirz = np.sqrt(r**2 - (obj.X - xc)**2) + zc
            else:
                cirz = - np.sqrt(r**2 - (obj.X - xc)**2) - zc

            obj.Z -= cirz
        return r, dev, (xc, zc)


class ProfileStitchError(FormEstimator):
    @staticmethod
    def remove(obj: profile.Profile, stitchPos, bplt=False):
        """
        Adjust the profile to preserve the derivative in the stitching.py points

        Parameters
        ----------
        obj: profile.Profile
            The profile on wich the correction is calculated
        stitchPos: np.array
            The x positions of the stitching.py
        bplt: bool
            Plots the comparison between the corrected profile and the original
        """
        posind = np.array([np.argwhere(obj.X == pos)[0] for pos in stitchPos])
        for i in posind:
            i = i[0]
            dl = obj.Z[i-1] - obj.Z[i]
            dr = obj.Z[i+1] - obj.Z[i+2]
            di = obj.Z[i] - obj.Z[i+1]

            err = (dl + dr) / 2 - di
            obj.Z[i+1:-1] -= err  # make the derivative the same

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(obj.X, obj.Z, label='Corrected stitching.py')
            ax.plot(obj.X, obj.Z0, label='Orignal data')
            # ax.plot(obj.X[posind], obj.Z[posind], 'o')

            ax.legend()
            funct.persFig([ax], 'x [um]', 'y[um]')
            plt.show()


# TODO: this class does not work, is it even useful??
class ProfileHistogram(FormEstimator):
    @staticmethod
    def remove(obj: profile.Profile, final_m, bplt=False):
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


class SurfaceLSPlane(FormEstimator):
    """
    Class derived from FormEstimator for the degree 1 plane 
    fit on surface data.
    """
    # shortcut class to surface poly of degree 1 (implemented for simmetry)
    def applyFit(self, obj: surface.Surface, bplt=False):
        return self.formFit(obj, bplt=bplt)

    @staticmethod
    def formFit(obj: surface.Surface, bplt=False):
        """
        Least square plane fit implementation

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the LS plane is applied
        bplt : bool
            If True plots the plane overimposed on the surface

        Returns
        ----------
        sol : np.ndarray
            Array of polynomial coefficients.
            
        Notes
        -----
        $\\min_{\\mathbf{p}} \\{\\sum\\limits_{i = 0}^{n-1}  f(\\mathbf{p}, x_i, y_i, z_i)^2 \\}$
        
        In this particular case of plane fitting $f(\\mathbf{p}, x_i, y_i, z_i)=z_i-z(\\mathbf{p}, x_i, y_i)$
        where $z(\\mathbf{p}, x, y)=p_{0,1} x + p_{1,0} y + p_{0,0}$ and 
        $\\mathbf{p}=[[p_{0,0}, p_{0,1}][p_{1,0}, \\textcolor{red}{p_{1,1}}]]$.
        """
        return SurfacePolynomial.formFit(obj, kx=1, ky=1, bplt=bplt)


class SurfacePolynomial(FormEstimator):
    """
    Class derived from FormEstimator for the degree kx, ky polynomial 
    fit on surface data.
    """
    def __init__(self, kx=3, ky=3, full=False, comp=lambda a, b: a < b, bound=None, cutter=None):
        """
        Creates a SurfacePolynomial object with the parameters provided

        Parameters
        ----------
        kx, ky : int
            Polynomial order in x and y, respectively.
        full : bool, optional
            If True, will solve using the full polynomial matrix.  Otherwise,
            will use the upper-left triangle of the matrix.  See
            `polyinterp2d` for further details.  Note that if kx != ky, then
            the full matrix will be solved for. (@sofia_redux/toolkit)
        comp : funct / lambda, optional
            comparison method between bound and surface
        bound: float, optional
            if not set the fit uses all points,
            if set the fit uses all points below the values,
            if set to True the fit uses only the values below the average value of the surface
        cutter : cutter.Cutter
            -if not set, the fit uses all points
            -if true allows the user to select manually the region of interest
            -if a cutter obj is passed the fit is done only on the cutted profile points
             and then applied on the whole profile
        """
        self.kx = kx
        self.ky = ky
        self.full = full
        self.comp = comp
        self.bound = bound
        self.cutter = cutter

    def applyFit(self, obj: surface.Surface, bplt=False):
        return self.formFit(obj, self.kx, self.ky, self.full, self.comp, self.bound, self.cutter)

    @staticmethod
    def formFit(obj: surface.Surface, kx=3, ky=3, full=False, comp=lambda a, b: a < b, bound=None, cutter=None, bplt=False):
        """
        Least square polynomial fit implementation on surface data

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the polynomial fit is applied
        kx, ky : int
            Polynomial order in x and y, respectively.
        full : bool, optional
            If True, will solve using the full polynomial matrix.  Otherwise,
            will use the upper-left triangle of the matrix.  See
            `polyinterp2d` for further details.  Note that if kx != ky, then
            the full matrix will be solved for. (@sofia_redux/toolkit)
        comp : funct / lambda, optional
            comparison method between bound and surface
        bound : float, optional
            if not set the fit uses all points,
            if set the fit uses all points below the values,
            if set to True the fit uses only the values below the average value of the surface
        cutter : cutter.Cutter
            -if not set, the fit uses all points
            -if true allows the user to select manually the region of interest
            -if a cutter obj is passed the fit is done only on the cutted profile points
             and then applied on the whole profile
        bplt: bool, optional
            If True plots the plane overimposed on the surface

        Returns
        ----------
        sol : np.ndarray
            Array of polynomial coefficients.
            
        Notes
        -----
        $\\min_{\\mathbf{p}} \\{\\sum\\limits_{i = 0}^{n-1}  f(\\mathbf{p}, x_i, y_i, z_i)^2 \\}$
        
        In this particular case of polynomial fitting $f(\\mathbf{p}, x_i, y_i, z_i)=z_i-z(\\mathbf{p}, x_i, y_i)$
        where $z(\\mathbf{p}, x, y)=\\sum_{i,j}p_{i,j}\\cdot x^i \\cdot y^j$ and 
        $\\mathbf{M}=[
        [m_{0,0},  \\cdots,  m_{0,kx}]
        [\\vdots,  \\ddots,  \\textcolor{red}{\\vdots}]
        [m_{ky,0},  \\textcolor{red}{\cdots},  \\textcolor{red}{m_{kx,ky}}]
        ]$.
        """
        if cutter is True:
            _, (x, y, z) = cutr.SurfaceCutter.cut(obj, finalize=False)
        elif cutter is not None:
            x, y, z = cutter.applyCut(obj, finalize=False)
        else:
            x, y, z = obj.X, obj.Y, obj.Z

        x, y, z,  = x.ravel(), y.ravel(), z.ravel()

        if bound is True:
            bound = np.nanmean(obj.Z)  # set the bound to the mean point

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
            keep = comp(z, bound)
            z = z[keep]
            a = a[keep]

        sol, _, _, _ = np.linalg.lstsq(a, z, rcond=None)

        if full:
            coeffs = sol.reshape((ny, nx))
        else:
            coeffs = np.zeros((ny, nx))
            for c, (j, i) in zip(sol, loop):
                coeffs[j, i] = c

        if bplt:
            FormEstimator.plot3DForm(obj.X, obj.Y, obj.Z, coeffs)

        obj.Z = FormEstimator.remove3DForm(obj.X, obj.Y, obj.Z, coeffs)
        return coeffs


class Surface3Points(FormEstimator):
    """
    Class derived from FormEstimator for the removal of the
    plane defined by three points selected by the user.
    """
    # TODO : add radius of points to pick with a single click
    @staticmethod
    def remove(obj: surface.Surface, bplt=False):
        """
        3 points plane fit implementation
        Opens a plot figure to choose the 3 points and finds the equation
        of the plane passing through those 3 points.

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the polynomial fit is applied
        bplt : bool
            If True plots the plane overimposed on the surface
            
        Notes
        -----
        Once the choice has been confirmed, the program calculates the coefficients 
        a, b and c of the equation of the plane $z_{plane}(x,y)$  passing through the 
        three specified positions Pa, Pb and Pc.
        
        Points coordinates:
        $P_a=(x_a, y_a, z_a)$
        $P_b=(x_b, y_b, z_b)$
        $P_c=(x_c, y_c, z_c)$
        
        Plane coefficients:
        $a=(y_b-y_a)(z_c-z_a)-(y_c-y_a)(z_b-z_a)$
        $b=(x_c-x_a)(z_b-z_a)-(x_b-x_a)(z_c-z_a)$
        $c=(x_b-x_a)(y_c-y_a)-(y_b-y_a)(x_c-x_a)$
        
        Plane equation:
        $z_{plane}(x,y)=\\frac{(-ax-by)}{c}$
        
        NOTE: in order to let the software perform a robust and affordable calculation, 
        the chosen points Pa, Pb and Pc must be as far apart as possible 
        (if possible, as vertices of an equilateral triangle)
        """
        def onClose(event):  # when fig is closed calculate plane parameters
            po = []
            for (a, b) in selector.verts:
                xind = np.where(obj.X[0, :] <= a)[0][-1]
                yind = np.where(obj.Y[:, 0] <= b)[0][-1]

                po.append([a, b, obj.Z[yind, xind]])

            # print(f"Collected points: {po}")
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
                FormEstimator.plot3DForm(obj.X, obj.Y, obj.Z, sol)

            obj.Z = FormEstimator.remove3DForm(obj.X, obj.Y, obj.Z, sol)
            plt.close(fig)

        fig, ax = plt.subplots()
        ax.pcolormesh(obj.X, obj.Y, obj.Z, cmap=cm.rainbow)
        ax.set_title('3 Points plane fit')
        fig.canvas.mpl_connect('close_event', onClose)

        selector = PolygonSelector(ax, lambda *args: None)

        plt.show()


class Sphere(FormEstimator):
    """
    Class derived from FormEstimator for the removal of the
    least square sphere
    """
    @staticmethod
    def formFit(obj: surface.Surface, finalize=True, radius=None, concavity=None, bplt=False):
        """
        Calculates the least square sphere

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the polynomial fit is applied
        finalize : bool
            If set to False the fit will not alter the surface,
            the method will only return the center and the radius
        radius : float
            If None the method will use the best fit radius
            If a radius is passed then the program will use it
        concavity : str
            Can be either 'convex' or 'concave'
            If set to None the program will find the concavity of the sample
        bplt : bool
            Plots the sphere fitted to the data points

        Returns
        -------
        (radius, C): (float, [xc, yc, zc])
            Radius and sphere center coordinates
            
        Notes
        -----
        The general equation of a sphere is: $(x-x_c)^2+(y-y_c)^2+(z-z_c)^2=r^2$
        
        Expanding the equation we get: $x^2+y^2+z^2=2xx_c+2yy_c+2zz_c+r^2-x_c^2-y_c^2-z_c^2$        
        We can solve for $x_c, y_c, z_c, r$
        """
        if concavity not in [None, 'concave', 'convex']: raise Exception('Concavity is not valid')
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
        C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)

        # radius = [radius]
        # solve for the radius
        if radius is None:
            t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
            radius = np.sqrt(t)[0]

        if finalize:
            sph = np.sqrt(radius**2 - (obj.X-C[0])**2 - (obj.Y-C[1])**2)
            if C[2][0] <= np.nanmean(obj.Z0): concavity = 'convex'
            if C[2] > np.nanmean(obj.Z0): concavity = 'concave'
            print(f'{concavity=}')
            obj.Z = obj.Z - sph - C[2] if concavity == 'convex' else obj.Z + sph - C[2]

        if bplt:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
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

        return radius, C


def _evalCyl(obj: surface.Surface, est_p, concavity):
    """
    Evaluates the cylinder points given the 5 parameters

    Parameters
    ----------
    obj: surface.Surface
        The surface object on wich the cylinder is calculated
    est_p: np.array
        P[0] = r, radius of the cylinder
        p[1] = Yc, y coordinate of the cylinder centre
        P[2] = Zc, z coordinate of the cylinder centre
        P[3] = alpha_z, rotation angle (radian) about the z-axis
        P[4] = alpha_y, rotation angle (radian) about the y-axis
    concavity:
        Either 'concave' or 'convex'

    Returns
    -------
    z_cyl: np.ndarray
        The calculated points
    """
    l = np.cos(est_p[3]) * np.cos(est_p[4])
    m = np.sin(est_p[3])
    n = np.cos(est_p[3]) * np.sin(est_p[4])

    A = 1 - n ** 2
    B = -2 * n * (l * obj.X + m * (obj.Y - est_p[1]))
    C = (1 - l ** 2) * obj.X ** 2 + \
        (1 - m ** 2) * (obj.Y - est_p[1]) ** 2 - \
        2 * obj.X * l * m * (obj.Y - est_p[1]) - est_p[0] ** 2

    delta = np.sqrt(B ** 2 - 4 * A * C) if concavity == 'convex' else -np.sqrt(B ** 2 - 4 * A * C)

    return (-B + delta) / (2 * A) + est_p[2]


class Cylinder(FormEstimator):
    """
    Class derived from FormEstimator for the calculation of the
    least square cylinder
    """
    @staticmethod
    def formFit(obj: surface.Surface, radius, alphaZ=0, alphaY=0, concavity='convex', base=False, finalize=True, bplt=False):
        """
        This is a fitting for a horizontal along x cylinder fitting
        uses the following parameters to find the best cylinder fit

        Parameters
        ----------
        obj : surface.Surface
            The surface object on wich the cylinder fit is applied
        radius : float
            The cylinder nominal radius
        alphaY : float
            An estimate ot the cylinder rotation about the y-axis (radian)
        alphaZ : float
            An estimate ot the cylinder rotation about the z-axis (radian)
        concavity : str
            Can be either 'convex' or 'concave'
        base : bool
            If true removes the points at the base of the cylinder
        finalize : bool
            If set to False the fit will not alter the surface,
            the method will only return the center and the radius
        bplt : bool
            Plots the sphere fitted to the data points

        Returns
        ----------
        est_p : np.array
            P[0] = r, radius of the cylinder\n
            p[1] = Yc, y coordinate of the cylinder centre\n
            P[2] = Zc, z coordinate of the cylinder centre\n
            P[3] = alpha_z, rotation angle (radian) about the z-axis\n
            P[4] = alpha_y, rotation angle (radian) about the y-axis
            
        Notes
        -----
        The cylinder fit is done according to the general cylider equation with 5
        degrees of freedom to accomodate for the axis rotation and translation.
        The general equation of the cylinder is $(x-x_c)^2+(y-y_c)^2+(z-z_c)^2-[l(x-x_c)+m(y-y_c)+n(z_{cyl}-z_c)]^2-R^2=0$
        Where $x,y,z_{cyl}$ are the coordinates of the cylinder points, $l, m, n$ 
        are the three components of the versor that represent the cylinder axis direction,
        $x_c, y_c, z_c$ are the coordinate of the centre and $R$ is the radius of the circular base.
        """
        # TODO : try masking the points instead of removing
        if base:  # remove base points
            if concavity == 'convex':
                th = np.nanmax(obj.Z) - 9 / 10 * radius
                obj.Z[obj.Z < th] = np.nan
            elif concavity == 'concave':
                th = np.nanmin(obj.Z) + 9 / 10 * radius
                obj.Z[obj.Z > th] = np.nan
            else:
                raise Exception('Concavity is not valid')

        X = obj.X.flatten()
        Y = obj.Y.flatten()
        Z = obj.Z.flatten()

        nanind = ~np.isnan(Z)

        X = X[nanind]
        Y = Y[nanind]
        Z = Z[nanind]

        def fitfunc(p, x, y, z):
            l = np.cos(p[3]) * np.cos(p[4])
            m = np.sin(p[3])
            n = np.cos(p[3]) * np.sin(p[4])

            return x ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2 - (l * x + m * (y - p[1]) + n * (z - p[2])) ** 2

        errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[0] ** 2  # error function

        p_init = np.array([radius, 0, 0, alphaZ, alphaY])
        est_p, success = optimize.leastsq(errfunc, p_init, args=(X, Y, Z))

        # print(f'Cylinder fit: {est_p}')

        z_cyl = _evalCyl(obj, est_p, concavity)

        if bplt:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(obj.X, obj.Y, obj.Z, cmap=cm.Reds, alpha=0.8)

            ax.plot_surface(obj.X, obj.Y, z_cyl, cmap=cm.rainbow, alpha=0.3)
            ax.set_box_aspect((np.ptp(obj.X), np.ptp(obj.Y), np.ptp(z_cyl[~np.isnan(z_cyl)])))

            funct.persFig([ax], 'x[um]', 'y[um]', 'z[um]')
            plt.show()

        if finalize:
            obj.Z = obj.Z - z_cyl

        return est_p
