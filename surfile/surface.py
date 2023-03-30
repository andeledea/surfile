import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, lines
import os
from matplotlib.widgets import RectangleSelector, PolygonSelector
from scipy import interpolate, ndimage
from alive_progress import alive_bar

from surfile import profile as prf
from surfile import funct
from surfile import measfile_io


# plu classes

class Surface:
    # TODO: remove meshgrids for x and y.
    def __init__(self):
        self.X0, self.Y0, self.Z0 = None, None, None
        self.Z = None
        self.Y = None
        self.X = None

        self.y = None
        self.x = None
        self.rangeY = None
        self.rangeX = None

        self.name = 'Figure'

    def openTxt(self, fname, bplt):
        with open(fname, 'r') as fin:
            self.name = os.path.basename(fin.name)

            line = fin.readline().split()
            sx = int(line[0])  # read number of x points
            sy = int(line[1])  # read number of y points
            print(f'Pixels: {sx} x {sy}')

            spacex = float(line[2])  # read x spacing
            spacey = float(line[3])  # read y spacing

        plu = np.loadtxt(fname, usecols=range(sy), skiprows=1)
        plu = (plu - np.mean(plu)) * (10 ** 6)  # 10^6 from mm to nm

        self.rangeX = sx * spacex
        self.rangeY = sy * spacey
        self.x = np.linspace(0, sx * spacex, num=sx)
        self.y = np.linspace(0, sy * spacey, num=sy)

        # create main XYZ and backup of original points in Z0
        self.X0, self.Y0 = self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z0 = self.Z = np.transpose(plu)

        if bplt: self.pltC()

    def openFile(self, fname, bplt):
        self.name = os.path.basename(fname)

        userscalecorrections = [1.0, 1.0, 1.0]
        dx, dy, z_map, weights, magnification, measdate = \
            measfile_io.read_microscopedata(fname, userscalecorrections, 0)
        (n_y, n_x) = z_map.shape
        self.rangeX = n_x * dx
        self.rangeY = n_y * dy
        self.x = np.linspace(0, self.rangeX, num=n_x)
        self.y = np.linspace(0, self.rangeY, num=n_y)

        # create main XYZ and backup of original points in Z0
        self.X0, self.Y0 = self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z0 = self.Z = z_map

        if bplt: self.pltC()

    def fitPlane3P(self):
        # TODO: define radius of points and avg
        # TODO: expand using polygonSelector https://matplotlib.org/stable/gallery/widgets/polygon_selector_simple.html
        """
        3 points plane fit implementation
        Opens a plot figure to choose the 3 points and fids the plane for those points
        """

        # po = []

        # def pointPick(point, mouseevent):  # called when pick event happens on fig
        #     if mouseevent.xdata is None:
        #         return False, dict()
        #     xdata = mouseevent.xdata
        #     ydata = mouseevent.ydata
        #
        #     # find indexes corresponding to picked values
        #     xind = np.where(self.X[0, :] <= xdata)[0][-1]
        #     yind = np.where(self.Y[:, 0] <= ydata)[0][-1]
        #
        #     if len(po) < 3:  # find the 3 points to apply the method
        #         po.append([xdata, ydata, self.Z[yind, xind]])
        #     return True, dict(pickx=xdata, picky=ydata)

        def onClose(event):  # when fig is closed calculate plane parameters
            po = []
            for (a, b) in selector.verts:
                xind = np.where(self.X[0, :] <= a)[0][-1]
                yind = np.where(self.Y[:, 0] <= b)[0][-1]

                po.append([a, b, self.Z[yind, xind]])

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
            plt.close(fig)

            self.__removePlane(a, b, c, d)

        fig, ax = plt.subplots()
        ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.rainbow)
        ax.set_title('3 Points plane fit')
        fig.canvas.mpl_connect('close_event', onClose)

        selector = PolygonSelector(ax, lambda *args: None)

        plt.show()

    def fitPlaneLS(self):
        """
        Least square plane fit implementation
        """
        # create matrix and Z vector to use lstsq
        XYZ = np.vstack([self.X.reshape(np.size(self.X)),
                         self.Y.reshape(np.size(self.Y)),
                         self.Z.reshape(np.size(self.Z))]).T
        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  # X
        G[:, 1] = XYZ[:, 1]  # Y
        Z = XYZ[:, 2]
        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)  # calculate LS plane

        print(f'Plane Params: a={a}, b={b}')
        self.__removePlane(a, b, -1, 0)

    def fitPlaneLS_bound(self, comp, bound=None):
        """
        Least square plane only top/bottom points
        :param comp: comparation method (ex. labda points, bound: points > bound)
        :param bound: bound level for comparison
        """
        if bound is None:
            bound = np.mean(self.Z)  # set the bound to the mean point

        XYZ = np.vstack([self.X.reshape(np.size(self.X)),
                         self.Y.reshape(np.size(self.Y)),
                         self.Z.reshape(np.size(self.Z))]).T

        where = np.argwhere(comp(self.Z.reshape(np.size(self.Z)), bound))
        XYZ = np.delete(XYZ, where, 0)  # remove unwanted points from fit

        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  # X
        G[:, 1] = XYZ[:, 1]  # Y
        Z = XYZ[:, 2]
        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)

        print(f'Plane Params: a={a}, b={b}')
        self.__removePlane(a, b, -1, 0)

    def histMethod(self, bplt, bins=100):
        """
        histogram method implementation
        :return: histogram values, bin edges values
        """
        hist, edges = np.histogram(self.Z, bins)
        if bplt: self.__pltHist(hist, edges)
        return hist, edges

    def __removePlane(self, a, b, c, d):
        # TODO: put this method at the end of the fit functions
        """
        removes the fitted plane from the points
        """
        z_plane = (- a * self.X - b * self.Y - d) * 1. / c
        self.Z = self.Z - z_plane + np.mean(z_plane)

    def cutSurface(self, extents):
        # TODO: check x and y coord on surface matrix
        xmin, xmax, ymin, ymax = extents

        print(extents)

        i_near = lambda arr, val: (np.abs(arr - val)).argmin()  # find the closest index
        start_x, end_x = i_near(self.x, xmin), i_near(self.x, xmax)
        start_y, end_y = i_near(self.y, ymin), i_near(self.y, ymax)

        self.X = self.X[start_y: end_y, start_x: end_x]
        self.Y = self.Y[start_y: end_y, start_x: end_x]
        self.Z = self.Z[start_y: end_y, start_x: end_x]

        self.x = self.x[start_x: end_x]
        self.y = self.y[start_y: end_y]

    def cutSurfaceRectangle(self):
        def onSelect(eclick, erelease):
            pass

        def onClose(event):
            self.cutSurface(RS.extents)

        fig, ax = plt.subplots()
        RS = RectangleSelector(ax, onSelect,
                               drawtype='box', useblit=True,
                               button=[1, 3],  # don't use middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)

        ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.rainbow)
        ax.set_title('Choose cut region')
        fig.canvas.mpl_connect('close_event', onClose)

        plt.show()
        return RS.extents

    def extractProfile(self) -> prf.Profile:
        """
        extracts a profile along x or y
        :return: profile object extracted (use p = copy.copy())
        """
        po = {'x': 0, 'y': 0}
        profile = prf.Profile()

        def pointPick(point, mouseevent):  # called when pick event on fig
            if mouseevent.xdata is None:
                return False, dict()
            xdata = mouseevent.xdata
            ydata = mouseevent.ydata

            xind = np.where(self.X[0, :] <= xdata)[0][-1]
            yind = np.where(self.Y[:, 0] <= ydata)[0][-1]
            print(f'Chosen point {xind}, {yind}')  # print the point the user chose

            po['x'] = xind
            po['y'] = yind
            return True, dict(pickx=xind, picky=yind)

        def onClose(event):  # called when fig is closed
            choice = input('Extract [x/y] profile?')
            if choice == 'x':  # the user chooses the direction of extraction
                profile.setValues(self.X[po[choice]], self.Z[po[choice]], False)
            else:
                profile.setValues(self.Y[po[choice]], self.Z[po[choice]], False)  # TODO: extract y profile
            plt.close(fig)

        fig, ax = plt.subplots()  # create the fig for profile selection
        ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.rainbow, picker=pointPick)
        ax.set_title('Extract profile')
        fig.canvas.mpl_connect('close_event', onClose)

        plt.show()
        return profile

    def extractMidProfile(self, direction='x') -> prf.Profile:
        profile = prf.Profile()
        if direction == 'x':
            profile.setValues(self.x, self.Z[int(np.size(self.x) / 2), :], False)

        if direction == 'y':
            profile.setValues(self.y, self.Z[:, int(np.size(self.y) / 2)], False)

        return profile

    def meanProfile(self, direction='x') -> prf.Profile:
        """
        extracts the mean profile along x or y
        :param direction: 'x' or 'y'
        :return: profile object extracted (use p = copy.copy())
        """
        profile = prf.Profile()
        if direction == 'x':
            profile.setValues(self.x, np.mean(self.Z, axis=0), False)

        if direction == 'y':
            profile.setValues(self.y, np.mean(self.Z, axis=1), False)

        return profile

    def sphereMaxProfile(self, start, bplt) -> prf.Profile:
        """
        Returns the profile starting from the max of the topo and
        on the positive x direction
        """
        # TODO : check if xind and yind are inverted
        profile = prf.Profile()
        if start == 'max':
            raveled = np.nanargmax(self.Z)
            unraveled = np.unravel_index(raveled, self.Z.shape)
            xind = unraveled[0]
            yind = unraveled[1]
        elif start == 'fit':
            r, C = self.sphereFit(bplt=False)
            yc = C[0][0]
            xc = C[1][0]
            xind = np.argwhere(self.x > xc)[0][0]
            yind = np.argwhere(self.y > yc)[0][0]
        elif start == 'center':
            xind = int(len(self.x) / 2)
            yind = int(len(self.y) / 2)
        elif start == 'local':
            maxima = (self.Z == ndimage.filters.maximum_filter(self.Z, 5))
            mid = np.asarray(self.Z.shape) / 2
            maxinds = np.argwhere(maxima)  # find all maxima indices
            center_max = maxinds[np.argmin(np.linalg.norm(maxinds - mid, axis=1))]

            xind = center_max[0]
            yind = center_max[1]
        else:
            raise Exception(f'{start} is not a valid start method')

        if bplt:
            fig, ax = plt.subplots()
            ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.viridis)  # hot, viridis, rainbow
            funct.persFig(
                [ax],
                gridcol='grey',
                xlab='x [um]',
                ylab='y [um]'
            )
            plt.hlines(self.y[xind], self.x[yind], self.x[-1])
            plt.show()

        profile.setValues(self.x[yind:-1], self.Z[xind][yind:-1], bplt=bplt)

        return profile

    def rotate(self, angle):
        """
        rotates the original topography by the specified angle
        """
        self.Z = ndimage.rotate(self.Z0, angle, order=0, reshape=False, cval=np.nan)

    def maxMeasSlope(self, angleStepSize, bplt, start='center'):
        """
        Returns the maximum measurable slope in every direction
        (directions are sampled every angleStepSize)
        """
        meas_slope1, meas_slope2 = [], []
        with alive_bar(int(360 / angleStepSize), force_tty=True,
                       title='Slope', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for a in range(0, 360, angleStepSize):
                self.rotate(a)
                slopeprofile = copy.copy(self.sphereMaxProfile(start=start, bplt=False))
                ms1, ms2 = slopeprofile.findMaxArcSlope(350)  # 350 um radius
                meas_slope1.append(np.rad2deg(ms1))
                meas_slope2.append(np.rad2deg(ms2))
                bar()

        if bplt:
            fig, ax = plt.subplots()
            ax.plot(range(0, 360, angleStepSize), meas_slope1, 'r', label='Max slope Rms1')
            ax.plot(range(0, 360, angleStepSize), meas_slope2, 'b', label='Max slope Rms2')
            ax.legend()

            fig2, bx = plt.subplots(subplot_kw={'projection': 'polar'})
            bx.plot(np.deg2rad(range(0, 360, angleStepSize)), meas_slope1, 'r', label='Max slope Rms1')
            bx.plot(np.deg2rad(range(0, 360, angleStepSize)), meas_slope2, 'b', label='Max slope Rms2')
            bx.legend()
            plt.show()

        return meas_slope1, meas_slope2

    def sphereRadius(self, angleStepSize, bplt, start='local'):
        rs = []
        zs = []
        fig, ax = plt.subplots()

        with alive_bar(int(360 / angleStepSize), force_tty=True,
                       title='Radius', theme='smooth',
                       elapsed_end=True, stats_end=True, length=30) as bar:
            for a in range(0, 360, angleStepSize):
                self.rotate(a)
                slopeprofile = copy.copy(self.sphereMaxProfile(start=start, bplt=False))
                r, z = slopeprofile.arcRadius(bplt=False)  # 350 um radius
                rs.append(r)
                zs.append(z)
                if bplt: ax.plot(z, r, alpha=0.2)
                bar()

        yr, error = funct.tolerant_mean(rs)
        yz, error = funct.tolerant_mean(zs)
        ax.plot(yz, yr, color='red')
        ax.set_ylim(0, max(yr))
        if bplt: plt.show()

        return yr, yz

    def sphereFit(self, bplt):
        #   Assemble the A matrix
        spZ = self.Z.flatten()
        spX = self.X.flatten()
        spY = self.Y.flatten()

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

        if bplt:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = np.cos(u) * np.sin(v) * radius
            y = np.sin(u) * np.sin(v) * radius
            z = np.cos(v) * radius
            x = x + C[0]
            y = y + C[1]
            z = z + C[2]

            #   3D plot of Sphere
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.rainbow)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.2)
            ax.set_xlabel('$x$', fontsize=16)
            ax.set_ylabel('\n$y$', fontsize=16)
            zlabel = ax.set_zlabel('\n$z$', fontsize=16)
            plt.show()

        return radius[0], C

    @funct.timer
    def resample(self, newXsize, newYsize):
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

    # TODO: implementare parametri S
    # TODO: implementare filtri 2D

    #####################################################################################################
    #                                       PLOT SECTION                                                #
    #####################################################################################################
    # TODO: move graphics to respective method
    def plt3D(self):
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.plot_surface(self.X, self.Y, self.Z, cmap=cm.rainbow)  # hot, viridis, rainbow
        funct.persFig(
            [ax_3d],
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]',
            zlab='z [nm]'
        )
        ax_3d.set_title(self.name)
        # ax_3d.colorbar(p)
        plt.show()

    def pltCompare(self):
        fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
        ax.pcolormesh(self.X0, self.Y0, self.Z0, cmap=cm.jet)  # hot, viridis, rainbow
        bx.pcolormesh(self.X, self.Y, self.Z, cmap=cm.jet)  # hot, viridis, rainbow
        funct.persFig(
            [ax, bx],
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]'
        )
        plt.show()

    def pltC(self):
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
        plt.show()

    # def __planePlot(self):
    #     maxx = np.max(self.X)
    #     maxy = np.max(self.Y)
    #     minx = np.min(self.X)
    #     miny = np.min(self.Y)
    #
    #     # compute needed points for plane plotting
    #     xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    #     z_plane = (-self.a * xx - self.b * yy - self.d) * 1. / self.c
    #
    #     # plot original points
    #     ax_pl = self.fig.add_subplot(self.gs[1, 2], projection='3d')
    #     ax_pl.plot_surface(self.X, self.Y, self.Z0, alpha=0.5, cmap=cm.Greys_r)  # hot, viridis, rainbow
    #     funct.persFig(
    #         [ax_pl],
    #         gridcol='grey',
    #         xlab='x [um]',
    #         ylab='y [um]',
    #         zlab='z [nm]'
    #     )
    #
    #     # plot plane
    #     ax_pl.plot_surface(xx, yy, z_plane - np.mean(z_plane), alpha=0.8, cmap=cm.viridis)

    def __pltHist(self, hist, edges):
        fig = plt.figure()
        ax_ht = fig.add_subplot(111)
        ax_ht.hist(edges[:-1], bins=edges, weights=hist / np.size(self.Z) * 100, color='red')
        funct.persFig(
            [ax_ht],
            gridcol='grey',
            xlab='z [nm]',
            ylab='pixels %'
        )
        plt.show()
