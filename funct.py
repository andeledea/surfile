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


def persFig(figure, gridcol, xlab, ylab, zlab=None):
    figure.set_xlabel(xlab)
    figure.set_ylabel(ylab)
    if zlab is not None:
        figure.set_zlabel(zlab)
    figure.grid(color=gridcol)
