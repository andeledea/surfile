"""
'surfile.funct'
- utility functions
- decorators for automation and code reusage

@author: Andrea Giura
"""

import time
import os

import matplotlib.pyplot as plt
import csv
import json

import numpy as np


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


def persFig(figures, xlab, ylab, zlab=None, gridcol='k'):
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


def options(csvPath=None, save=None, bplt=False, chrono=False):
    """
    Decorator that implements global configurations

    Parameters
    ----------
    csvPath: str
        If not none saves the return parameters of the decorated function to a csv
    save: str
        If not none saves the figures in the path
    bplt: bool
        If True shows the images
    chrono: bool
        If true times the duration of the decorated function

    Notes
    ----------
    Use this decorator only on methods that do not call plt.show
    """
    # TODO now every function asks the user to define a bplt parameter
    # TODO this decorator implements global options, how can we implement both solutions
    # Solution: the bplt can be defaulted to False in every function, the plt.show()
    # should not be called in the function but only in the decorator and only if there
    # are pending graphs to be shown plt.get_fignums(). Try this and see...
    def outer(func):
        def inner(*args, **kwargs):                
            init = time.time()
            # exec the function
            ret = func(*args, **kwargs)

            if csvPath is not None:  # save the return parameters
                try:
                    for i, xy in enumerate(ret):
                        np.savetxt(f"{csvPath}{func.__name__}_{rcs.currentImage}_{str(i)}.csv", np.c_[xy])
                except TypeError:
                    with open(f"{csvPath}.csv", 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=range(len(ret)), dialect='excel')
                        writer.writerow(dict(enumerate(ret)))

            fig_nums = plt.get_fignums()
            figs = [plt.figure(n) for n in fig_nums]
            if save is not None:  # save the figures
                if len(plt.get_fignums()) > 0:
                    print(Bcol.OKCYAN + f'Saving images from function {func.__name__}' + Bcol.ENDC)

                    for i, fig in enumerate(figs):
                        fig.savefig(f'{save}{func.__name__}_{rcs.currentImage}_{str(i)}.png', format='png')
                else:
                    print(Bcol.WARNING + f'Function {func.__name__} has no active figures' + Bcol.ENDC)

            if bplt:  # plot the figure
                if len(plt.get_fignums()) > 0:
                    print(Bcol.OKCYAN + f'Plotting image from function {func.__name__}' + Bcol.ENDC)
                    plt.show()
                else:
                    print(Bcol.WARNING + f'Function {func.__name__} has no active figures' + Bcol.ENDC)
            else:
                plt.close('all')

            if chrono:  # time the function
                print(Bcol.OKCYAN +
                      f"Function {func.__name__} took: {(time.time() - init):.2f} seconds"
                      + Bcol.ENDC)
            return ret
        return inner
    return outer


def classOptions(decorator):
    """
    Class decorator uset to apply the same @dec to all methods
    """
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


class Rc:
    """
    Class used to define the options of the decorators

    Parameters convention: each parameter has:
    - 1 letter indentifying the option
        - c: csv
        - s: save the image
        - b: bplt
        - t: chrono
    - 1 letter identifying the type
        - s: surface
        - p: profile
    - 3 chars identifying the decorated function
    """
    # TODO: i think this is not the best way, I tried to emulate matplotlib's RcParams
    # I don't really understand how mpl' Rcs work ... maybe I can define all params in a file (like mpl does)
    # and then read the file (json) with this class using only a dictionary

    params: dict
    currentImage: str = 'Image'

    def __init__(self):
        import surfile
        rcfile = os.path.join(os.path.dirname(surfile.__file__), 'Rcs.json')
        with open(rcfile, 'r') as fin:
            self.params = json.load(fin)

    def load(self, js_fin):
        """
        Loads a user defined rc parameters file

        Parameters
        ----------
        js_fin: str
            The json file name
        """
        with open(js_fin, 'r') as fin:
            self.params = json.load(fin)

    def store(self, js_fout):
        """
        Saves the current parameters to a file

        Parameters
        ----------
        js_fout: str
            The json file name
        """
        with open(js_fout, 'w') as fout:
            json.dump(self.params, fout)

    def setCurrentImage(self, name):
        self.currentImage = name


rcs = Rc()  # define global Rcs
