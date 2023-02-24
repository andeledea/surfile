import time


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
    :param hist: histogram y values
    :param edges: histogram bins
    :return: height of sample
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
    for figure in figures:
        figure.set_xlabel(xlab)
        figure.set_ylabel(ylab)
        if zlab is not None:
            figure.set_zlabel(zlab)
        figure.grid(color=gridcol)


def timer(func):  # wrapper function
    def wrapper(*args, **kwargs):
        init = time.time()
        ret = func(*args, **kwargs)
        print(Bcol.OKCYAN + f"Function {func.__name__} took: {(time.time() - init):.2f} seconds" + Bcol.ENDC)
        return ret
    return wrapper

