import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RectangleSelector
from scipy import signal, ndimage, interpolate

import profile as prf
import funct


# plu classes

class Surface:
    def __init__(self):
        self.Z0 = None
        self.Z = None
        self.Y = None
        self.X = None

        self.y = None
        self.x = None
        self.rangeY = None
        self.rangeX = None
        self.gs = None
        self.fig = None

        self.a, self.b, self.c, self.d = 0, 0, 0, 0  # plane parameters

    def openTxt(self, fname):
        with open(fname, 'r') as fin:
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
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z0 = self.Z = np.transpose(plu)

    def fitPlane3P(self):
        """
        3 points plane fit implementation
        Opens a plot figure to choose the 3 points and fids the plane for those points
        """
        po = []

        def pointPick(point, mouseevent):  # called when pick event happens on fig
            if mouseevent.xdata is None:
                return False, dict()
            xdata = mouseevent.xdata
            ydata = mouseevent.ydata

            # find indexes corresponding to picked values
            xind = np.where(self.X[0, :] <= xdata)[0][-1]
            yind = np.where(self.Y[:, 0] <= ydata)[0][-1]

            if len(po) < 3:  # find the 3 points to apply the method
                po.append([xdata, ydata, self.Z[yind, xind]])
            return True, dict(pickx=xdata, picky=ydata)

        def onClose(event):  # when fig is closed calculate plane parameters
            print(f"Collected points: {po}")
            a1 = po[1][0] - po[0][0]  # x2 - x1;
            b1 = po[1][1] - po[0][1]  # y2 - y1;
            c1 = po[1][2] - po[0][2]  # z2 - z1;
            a2 = po[2][0] - po[0][0]  # x3 - x1;
            b2 = po[2][1] - po[0][1]  # y3 - y1;
            c2 = po[2][2] - po[0][2]  # z3 - z1;
            self.a = b1 * c2 - b2 * c1
            self.b = a2 * c1 - a1 * c2
            self.c = a1 * b2 - b1 * a2
            self.d = 0  # (- self.a * po[0][0] - self.b * po[0][1] - self.c * po[0][2])
            plt.close(fig)

        fig, ax = plt.subplots()
        ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.rainbow, picker=pointPick)
        ax.set_title('3 Points plane fit')
        fig.canvas.mpl_connect('close_event', onClose)

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

        print(f'Params: a={a}, b={b}')

        self.d = 0  # discard d value to keep plane in center
        self.a = a
        self.b = b
        self.c = -1  # z coefficient in plane eq.

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

        print(f'Params: a={a}, b={b}')

        self.d = 0
        self.a = a
        self.b = b
        self.c = -1

    def histMethod(self, bins=100):
        """
        histogram method implementation
        :return: histogram values, bin edges values
        """
        hist, edges = np.histogram(self.Z, bins)
        return hist, edges

    def removePlane(self):
        """
        removes the fitted plane from the points
        """
        z_plane = (-self.a * self.X - self.b * self.Y - self.d) * 1. / self.c
        self.Z = self.Z - z_plane + np.mean(z_plane)

    def cutSurface(self, radius):
        # TODO: check x and y coord on surface matrix

        n_radius_x = int(radius / (self.rangeX / len(self.x)))
        n_radius_y = int(radius / (self.rangeY / len(self.y)))

        n_mid_x = int(len(self.x) / 2)
        n_mid_y = int(len(self.y) / 2)

        start_x, end_x = n_mid_x - n_radius_x, n_mid_x + n_radius_x
        start_y, end_y = n_mid_y - n_radius_y, n_mid_y + n_radius_y

        print(f'x len: {len(self.x)}, cutting from {start_x} to {end_x}')
        print(f'y len: {len(self.y)}, cutting from {start_y} to {end_y}')

        print(np.shape(self.Z))

        self.X = self.X[start_y: end_y, start_x: end_x]
        self.Y = self.Y[start_y: end_y, start_x: end_x]
        self.Z = self.Z[start_y: end_y, start_x: end_x]

        print(np.shape(self.Z))

    def cutSurfaceRectangle(self):
        def onSelect(eclick, erelease):
            print('Choose cut region')

        def toggle_selector(event):
            print(' Key pressed.')
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print(' RectangleSelector activated.')
                toggle_selector.RS.set_active(True)

        def onClose(event):
            xmin, xmax, ymin, ymax = toggle_selector.RS.extents

            print(toggle_selector.RS.extents)

            i_near = lambda arr, val: (np.abs(arr - val)).argmin()
            start_x, end_x = i_near(self.x, xmin), i_near(self.x, xmax)
            start_y, end_y = i_near(self.y, ymin), i_near(self.y, ymax)

            self.X = self.X[start_y: end_y, start_x: end_x]
            self.Y = self.Y[start_y: end_y, start_x: end_x]
            self.Z = self.Z[start_y: end_y, start_x: end_x]

        fig, ax = plt.subplots()
        toggle_selector.RS = RectangleSelector(ax, onSelect,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)

        ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.rainbow)
        ax.set_title('3 Points plane fit')
        fig.canvas.mpl_connect('close_event', onClose)

        plt.show()

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
                profile.setValues(self.X[po[choice]], self.Z[po[choice]])
            else:
                profile.setValues(self.Y[po[choice]], self.Z[po[choice]])  # TODO: extract y profile
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
            profile.setValues(self.x, self.Z[int(np.size(self.x) / 2), :])

        if direction == 'y':
            profile.setValues(self.y, self.Z[:, int(np.size(self.y) / 2)])

        return profile

    def meanProfile(self, direction='x') -> prf.Profile:
        """
        extracts the mean profile along x or y
        :param direction: 'x' or 'y'
        :return: profile object extracted (use p = copy.copy())
        """
        profile = prf.Profile()
        if direction == 'x':
            profile.setValues(self.x, np.mean(self.Z, axis=0))

        if direction == 'y':
            profile.setValues(self.y, np.mean(self.Z, axis=1))

        return profile

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
    def init_graphics(self):
        """
        call this function before starting to plot
        """
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(3, 3)

    def pltPlot(self, fname):
        ax_3d = self.fig.add_subplot(self.gs[0:-1, 0:-1], projection='3d')
        ax_3d.plot_surface(self.X, self.Y, self.Z, cmap=cm.rainbow)  # hot, viridis, rainbow
        funct.persFig(
            ax_3d,
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]',
            zlab='z [nm]'
        )
        ax_3d.set_title('Surface' + fname)
        # ax_3d.colorbar(p)

    def pltCplot(self):
        ax_2d = self.fig.add_subplot(self.gs[0, 2])
        ax_2d.pcolormesh(self.X, self.Y, self.Z, cmap=cm.jet)  # hot, viridis, rainbow
        funct.persFig(
            ax_2d,
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]'
        )

    def planePlot(self):
        maxx = np.max(self.X)
        maxy = np.max(self.Y)
        minx = np.min(self.X)
        miny = np.min(self.Y)

        # compute needed points for plane plotting
        xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
        z_plane = (-self.a * xx - self.b * yy - self.d) * 1. / self.c

        # plot original points
        ax_pl = self.fig.add_subplot(self.gs[1, 2], projection='3d')
        ax_pl.plot_surface(self.X, self.Y, self.Z0, alpha=0.5, cmap=cm.Greys_r)  # hot, viridis, rainbow
        funct.persFig(
            ax_pl,
            gridcol='grey',
            xlab='x [um]',
            ylab='y [um]',
            zlab='z [nm]'
        )

        # plot plane
        ax_pl.plot_surface(xx, yy, z_plane - np.mean(z_plane), alpha=0.8, cmap=cm.viridis)

    def histPlot(self, hist, edges):
        ax_ht = self.fig.add_subplot(self.gs[2, :])
        ax_ht.hist(edges[:-1], bins=edges, weights=hist / np.size(self.Z) * 100, color='red')
        funct.persFig(
            ax_ht,
            gridcol='grey',
            xlab='z [nm]',
            ylab='pixels %'
        )
