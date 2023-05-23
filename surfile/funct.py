"""
'surfile.funct'
- utility functions
- decorators for automation and code reusage

@author: Andrea Giura
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import csv


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


def options(csvPath=None, save=False, bplt=False, chrono=False):
    """
    Decorator that implements global configurations

    Parameters
    ----------
    csvPath: str
        If not none saves the return parameters of the decorated function to a csv
    save: bool
        If True saves the figures
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
                with open(f"{csvPath}.csv", 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=range(len(ret)), dialect='excel')
                    writer.writerow(dict(enumerate(ret)))

            if save:  # save the figures
                if len(plt.get_fignums()) > 0:
                    print(Bcol.OKCYAN + f'Saving images from function {func.__name__}' + Bcol.ENDC)
                    fig_nums = plt.get_fignums()
                    figs = [plt.figure(n) for n in fig_nums]
                    for i, fig in enumerate(figs):
                        fig.savefig(f'img\\{func.__name__}_{fig.axes[0].get_title()}_{str(i)}.png', format='png')
                else:
                    print(Bcol.WARNING + f'Function {func.__name__} has no active figures' + Bcol.ENDC)

            if bplt:  # plot the figure
                if len(plt.get_fignums()) > 0:
                    print(Bcol.OKCYAN + f'Plotting image from function {func.__name__}' + Bcol.ENDC)
                    plt.show()
                else:
                    print(Bcol.WARNING + f'Function {func.__name__} has no active figures' + Bcol.ENDC)

            if chrono:  # time the function
                print(Bcol.OKCYAN +
                      f"Function {func.__name__} took: {(time.time() - init):.2f} seconds"
                      + Bcol.ENDC)
            return ret
        return inner
    return outer

