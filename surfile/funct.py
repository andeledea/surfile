import time

import numpy as np
import matplotlib.pyplot as plt


class Bcol:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def findHfromHist(hist, edges):
    """
    Finds the 2 maximum values in the histogram and calculates the distance of
    the peaks -> gives info about sample step height

    Parameters
    ----------
    hist: np.array
        histogram y values
    edges: np.array
        histogram bins

    Returns
    ----------
    h: float
        Height of sample
    """
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


def persFig(figures, gridcol, xlab, ylab, zlab=None):
    """
    Personalize an axis object or multiple
    Parameters
    ----------
    figures: list
        The list of ax objects to be customized
    gridcol: str
        The color of the grid
    xlab: str
    ylab: str -> labels
    zlab: str
    """
    for figure in figures:
        figure.set_xlabel(xlab)
        figure.set_ylabel(ylab)
        if zlab is not None:
            figure.set_zlabel(zlab)
        figure.grid(color=gridcol)


def options(**param):
    """
    Decorator that implements global configurations

    Parameters
    ----------
    save: bool
        If True saves the figure
    bplt: bool
        If True shows the image
    chrono: bool
        If true times the duration of the decorated function

    Notes
    ----------
    Use this decorator only on methods that do not call plt.show
    """
    # TODO now every function asks the user to define a bplt parameter
    # TODO this decorator implements global options, how can we implement both solutions
    def outer(func):
        def inner(*args, **kwargs):
            init = time.time()
            ret = func(*args, **kwargs)

            try:
                if param['save']:  # save the figure
                    print(Bcol.OKCYAN + f'Saving image from function {func.__name__}' + Bcol.ENDC)
                    plt.savefig(f'{func.__name__}.png')
            except KeyError:
                print(Bcol.WARNING + 'Missing save in options' + Bcol.ENDC)

            try:
                if param['bplt']:  # plot the figure
                    print(Bcol.OKCYAN + f'Plotting image from function {func.__name__}' + Bcol.ENDC)
                    plt.show()
            except KeyError:
                print(Bcol.WARNING + 'Missing save in options' + Bcol.ENDC)

            try:
                if param['chrono']:  # time the function
                    print(Bcol.OKCYAN +
                          f"Function {func.__name__} took: {(time.time() - init):.2f} seconds"
                          + Bcol.ENDC)
            except KeyError:
                print(Bcol.WARNING + 'Missing chrono in options' + Bcol.ENDC)
            return ret
        return inner
    return outer


def tolerant_mean(arrs: list):
    """
    Calculates the average between multiple arrays of different length

    Parameters
    ----------
    arrs: list
        The arrays to be processed

    Returns
    -------
    mean: np.array
        The mean calculated
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)
