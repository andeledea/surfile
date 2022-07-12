import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# plu classes
def findHfromHist(hist, edges):
    ml = 0
    mh = 0
    binl = 0
    binh = 0
    i = 0
    for edge in edges[:-1]:
        if edge < 0:
            binl = edge if hist[i] > ml else binl
            ml = max(ml, hist[i])

        else:
            binh = edge if hist[i] > mh else binh
            mh = max(mh, hist[i])

        i = i + 1

    print(f'Max left {ml} @ {binl} \nMax right {mh} @ {binh}')
    print(f'Height: {binh - binl}')

    return binh - binl


def histPlot(edges, hist):
    fig = plt.figure()
    bx = plt.axes()
    bx.plot(edges[0:-1], hist, color='red')
    persFig(
        bx,
        gridcol='grey',
        title='Histogram',
        xlab='z [nm]',
        ylab='pixels'
    )


def persFig(figure, gridcol, title, xlab, ylab, zlab=None):
    figure.set_title(title)
    figure.set_xlabel(xlab)
    figure.set_ylabel(ylab)
    if zlab is not None:
        figure.set_zlabel(zlab)
    figure.grid(color=gridcol)


class Plu:
    def __init__(self, name):
        self.a, self.b, self.c, self.d = 0, 0, 0, 0
        with open(name, 'r') as fin:
            line = fin.readline().split()
            sx = int(line[0])
            sy = int(line[1])
            print(f'Pixels: {sx} x {sy}')

            spacex = float(line[2])
            spacey = float(line[3])

        plu = np.loadtxt(name, usecols=range(sy), skiprows=1)
        plu = (plu - np.mean(plu)) * (10 ** 6)

        x = np.linspace(0, sx * spacex, num=sx)
        y = np.linspace(0, sy * spacey, num=sy)

        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.transpose(plu)

    def fitPlane3P(self):
        po = []

        def pointPick(point, mouseevent):
            if mouseevent.xdata is None:
                return False, dict()
            xdata = mouseevent.xdata
            ydata = mouseevent.ydata

            xind = np.where(self.X[0, :] <= xdata)[0][-1]
            yind = np.where(self.Y[:, 0] <= ydata)[0][-1]

            if len(po) < 3:
                po.append([xdata, ydata, self.Z[yind, xind]])
            return True, dict(pickx=xdata, picky=ydata)

        def onClose(event):
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
            plt.close()

        fig, ax = plt.subplots()
        points = ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.rainbow, picker=pointPick)
        ax.set_title('3 Points plane fit')
        fig.canvas.mpl_connect('close_event', onClose)

        plt.show()

    def fitPlaneLS(self):
        XYZ = np.vstack([self.X.reshape(np.size(self.X)),
                         self.Y.reshape(np.size(self.Y)),
                         self.Z.reshape(np.size(self.Z))]).T
        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  # X
        G[:, 1] = XYZ[:, 1]  # Y
        Z = XYZ[:, 2]
        print(Z)
        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)

        print(f'Params: a={a}, b={b}')

        self.d = 0
        self.a = a
        self.b = b
        self.c = -1

    def fitPlaneLS_bound(self, comp, bound=None):
        if bound is None:
            bound = np.mean(self.Z)

        XYZ = np.vstack([self.X.reshape(np.size(self.X)),
                         self.Y.reshape(np.size(self.Y)),
                         self.Z.reshape(np.size(self.Z))]).T
        print(np.shape(XYZ), XYZ)

        where = np.argwhere(comp(self.Z.reshape(np.size(self.Z)), bound))
        XYZ = np.delete(XYZ, where, 0)
        print(len(where), np.shape(XYZ))

        plt.show()
        plt.close()

        (rows, cols) = XYZ.shape
        G = np.ones((rows, 3))
        G[:, 0] = XYZ[:, 0]  # X
        G[:, 1] = XYZ[:, 1]  # Y
        Z = XYZ[:, 2]
        print(Z)
        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)

        print(f'Params: a={a}, b={b}')

        self.d = 0
        self.a = a
        self.b = b
        self.c = -1

    def histMethod(self, bins=100):
        hist, edges = np.histogram(self.Z, bins)
        return hist, edges

    def removePlane(self):
        z_plane = (-self.a * self.X - self.b * self.Y - self.d) * 1. / self.c
        self.Z = self.Z - z_plane + np.mean(z_plane)

    #####################################################################################################
    #                                       PLOT SECTION                                                #
    #####################################################################################################

    def pltPlot(self, fname):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.rainbow)  # hot, viridis, rainbow
        persFig(
            ax,
            gridcol='grey',
            title='Surface ' + fname,
            xlab='x [um]',
            ylab='y [um]',
            zlab='z [nm]'
        )
        fig.colorbar(p)

    def pltCplot(self, fname):
        fig = plt.figure()
        ax = plt.axes()
        p = ax.pcolormesh(self.X, self.Y, self.Z, cmap=cm.rainbow)  # hot, viridis, rainbow
        persFig(
            ax,
            gridcol='grey',
            title='Surface ' + fname,
            xlab='x [um]',
            ylab='y [um]'
        )
        fig.colorbar(p)

    def planePlot(self):
        maxx = np.max(self.X)
        maxy = np.max(self.Y)
        minx = np.min(self.X)
        miny = np.min(self.Y)

        # plot original points
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = ax.plot_surface(self.X, self.Y, self.Z, alpha=0.5, cmap=cm.Greys_r)  # hot, viridis, rainbow
        persFig(
            ax,
            gridcol='grey',
            title='Plane fit',
            xlab='x [um]',
            ylab='y [um]',
            zlab='z [nm]'
        )
        fig.colorbar(p)

        # compute needed points for plane plotting
        xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
        z_plane = (-self.a * xx - self.b * yy - self.d) * 1. / self.c

        # plot plane
        ax.plot_surface(xx, yy, z_plane - np.mean(z_plane), alpha=0.8, cmap=cm.viridis)
