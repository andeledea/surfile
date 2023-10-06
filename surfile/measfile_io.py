# -*- coding: utf-8 -*-
"""
'surfile.measfile_io'
Module to read data output of microscopes:
- Olympus Lext
- Sensofar plu
- Asylum Cypher ibw
- Digital Surf sur
- SPIP bcr and bcrf
- ISO 25178-71 sdf
- NMM (Gaoliang)

Created 2019-10-03
last edit 2021-04-02
@author: Dorothee Hueser
"""

import numpy as np
from PIL import Image
import pathlib
import struct
import matplotlib.pyplot as plt

withigor = 0
try:
    from igor import binarywave

    withigor = 1
except ImportError:
    print('igor not found: use')
    print('pip install --proxy=http://webproxy.bs.ptb.de:8080 igor')
    print('to install it, this command also works in spyder console I/A')
try:
    from csaps import csaps
except ImportError:
    print('csaps not found: use')
    print('pip install --proxy=http://webproxy.bs.ptb.de:8080 csaps')
    print('to install it, this command also works in spyder console I/A')


#
def str2float(s):
    try:
        return float(s)
    except ValueError:
        return np.NaN


#
def extract_tag(sdata, tagkey):
    i1 = sdata.find(b''.join([b'<', tagkey, b'>']))
    i2 = sdata.find(b''.join([b'</', tagkey, b'>']))
    tagcontent = sdata[i1 + len(tagkey) + 2:i2]
    return tagcontent

# TXT file readings ############################

def read_spaceZtxt(fname):
    with open(fname, 'r') as fin:
        line = fin.readline().split()
        sx = int(line[0])  # read number of x points
        sy = int(line[1])  # read number of y points
        print(f'Pixels: {sx} x {sy}')

        spacex = float(line[2])  # read x spacing
        spacey = float(line[3])  # read y spacing

    plu = np.loadtxt(fname, usecols=range(sy), skiprows=1)
    plu = (plu - np.mean(plu)) * (10 ** 6)  # 10^6 from mm to nm

    rangeX = sx * spacex
    rangeY = sy * spacey
    x = np.linspace(0, sx * spacex, num=sx)
    y = np.linspace(0, sy * spacey, num=sy)

    # create main XYZ and backup of original points in Z0
    X, Y = np.meshgrid(self.x, self.y)
    Z = np.transpose(plu)

    return X, Y, Z, x, y

def read_xyztxt(fname):
    X, Y, Z = np.genfromtxt(fname, unpack=True, usecols=(0, 1, 2), delimiter=';')

    # find size of array
    i = np.argwhere(Y > 0)[0][0]

    X = np.reshape(X, (i, i))
    Y = np.reshape(Y, (i, i))
    Z = np.reshape(Z, (i, i))

    x = X[0,:]
    y = Y[:,0]

    return X, Y, Z, x, y
################################################

#
def read_asc(filecontent: str):
    lines = filecontent.decode('utf-8').splitlines(keepends=False)
    name = lines[0][lines[0].find(':') + 1 :]
    lx = str2float(lines[1][lines[1].find(':') + 1 :])
    ly = str2float(lines[2][lines[2].find(':') + 1 :])
    nx = int(lines[3][lines[3].find(':') + 1 :])
    ny = int(lines[4][lines[4].find(':') + 1 :])

    dx = lx / nx
    dy = ly / ny

    height_map = np.ndarray(shape=(nx, ny))
    for i, l in enumerate(lines[6:]):
        height_map[:, i] = np.fromstring(l, sep='\t', dtype=float)

    return nx, ny, dx, dy, height_map.T, ''

#
def read_lextinfo(filecontent):
    """read factor to convert raw data values into values in micron

    parameter:
    filecontent: binary content of complete file

    return:
    nx, ny: number of data points/ pixel for each direction
    endheightmap: byte position of end of height map
    heightperpix: factor to be multiplied to raw data to obtain micron
    magnification: of objective lens
    measdate: date an time when data where aquired with instrument
    """
    keywords = [b'HeightDataPerPixel', b'HEIGHT', \
                b'ObjectiveLenseMagnification', b'LEXT 1.0.2.1', \
                b'ImageWidth', b'ImageHeight', \
                b'MakerCalibrationValue', b'HeightDataUnitZ', \
                b'HeightDataUnitX', b'HeightDataUnitY', \
                b'HeightMaxValue', b'HeightMaxScalingValue']
    axes = [b'X', b'Y', b'Z']
    #
    #   according to format description of Olympus, HeightperPixel[X,Y] is
    #   always in Picometer
    #    print('filecontent: ', filecontent)
    infostring7 = extract_tag(filecontent, keywords[7])
    infostring8 = extract_tag(filecontent, keywords[8])
    infostring9 = extract_tag(filecontent, keywords[9])
    print('unit  z: ', infostring7)
    print('unit  x: ', infostring8)
    print('unit  y: ', infostring9)
    tomicron = np.array([(0.001 ** int(infostring8)), (0.001 ** int(infostring9)), \
                         (0.001 ** int(infostring7))])
    infostring10 = extract_tag(filecontent, keywords[10])
    infostring11 = extract_tag(filecontent, keywords[11])
    print('HeightMaxValue: ', int(infostring10), ', HeightMaxScalingValue: ', int(infostring11))
    if int(infostring10) * int(infostring11) == 0:
        print('attention - this is interpreted not to have the correct HeightDataUnit')
        print('from conversion of pior to lext format, therefore correction from mm to nm')
        print('finally delivered in micron, as usual')
        tomicron *= 1e-6
    #    unittomicron
    heightperpix = np.zeros(3)
    for k in range(0, 3):
        infostring = extract_tag(filecontent, b''.join([keywords[0], axes[k]]))
        calibstring = extract_tag(filecontent, b''.join([keywords[6], axes[k]]))
        print('---\n tomicron[k]: ', tomicron[k])
        heightperpix[k] = str2float(infostring) * \
                          (1e-6 * str2float(calibstring)) * tomicron[k]
    #
    endheightmap = filecontent.find(keywords[1])
    #
    infostring = extract_tag(filecontent, keywords[2])
    magnification = str2float(infostring)
    #
    infostring = extract_tag(filecontent, keywords[4])
    nx = int(infostring)
    infostring = extract_tag(filecontent, keywords[5])
    ny = int(infostring)
    #
    i1 = filecontent.find(keywords[3])
    measdate = filecontent[i1 + len(keywords[3]) + 2:i1 + len(keywords[3]) + 22].decode('ascii')
    return nx, ny, endheightmap, heightperpix, magnification, measdate


#
def read_lextimg(filename, nx, ny, endheightmap, heightflag):
    imgs = Image.open(filename)
    frameflag = 1
    iframe = 0
    iimg = 0
    npimg = []
    cumbytes = 0
    posbyte = []
    #    for iframe in range(0, 4):
    while frameflag:
        print('iframe: ', iframe)
        try:
            imgs.seek(iframe)
            print(imgs.mode)
            if imgs.mode.find('RGB') > -1:
                rgb = np.array(Image.Image.convert(imgs))
                (nr, nc, nn) = rgb.shape
                cumbytes = cumbytes + nr * nc * nn
            if imgs.mode.find('I;16') > -1:
                hlp = np.array(Image.Image.convert(imgs))
                (nr, nc) = hlp.shape
                print('nr ', nr, ' ny ', ny)
                print('nc ', nc, ' nx ', nx)
                if (nr == ny) and (nc == nx):
                    cumbytes = cumbytes + nr * nc * 2
                    posbyte.append(cumbytes)
                    npimg.append(hlp)
                    iimg = iimg + 1

            iframe = iframe + 1
        except:
            print('geht nicht')
            frameflag = 0
    #    for jimg in range(0, iimg):
    #        kimg = iimg - jimg - 1
    #        if endheightmap > posbyte[kimg]:
    #            height_map = npimg[kimg]
    #            break
    #    print('iimg: ', iimg, ' heightflag ', heightflag)
    if (heightflag == 1):
        the_map = npimg[iimg - 1]
        print(the_map.shape)
    else:
        if (iimg == 2):
            the_map = npimg[iimg - 2]
        else:
            print('there does not exist any intenstiy map!')
            the_map = []
    return the_map

#
# Sensofar
#
def read_plu(filecontent):
    DATE_SIZE = 128
    COMMENT_SIZE = 256
    measdate = filecontent[0:DATE_SIZE].decode('ascii', errors='ignore').replace("\x00", "")
    print('--plu measdate: ', measdate)
    i1 = DATE_SIZE + COMMENT_SIZE + 4
    npix = struct.unpack('2I', filecontent[i1:i1 + 8])
    pix_size = struct.unpack('2f', filecontent[i1 + 16:i1 + 24])
    i2 = i1 + 10 * 4 + 10 * 4 + 8 + 3 * 4 + 1 + 7 + 4
    num_height = npix[0] * npix[1]
    data_height = struct.unpack(str(num_height) + 'f', filecontent[i2:i2 + 4 * num_height])
    height_map = np.reshape(data_height, (npix[0], npix[1]))
    inan = np.where(height_map > 1e6)
    height_map[inan] = np.NaN
    dx = pix_size[1]
    dy = pix_size[0]
    return npix[1], npix[0], dx, dy, height_map, measdate


#
# Cypher
#
def read_ibw(filename):
    filepath = pathlib.Path(filename)
    ibw = binarywave.load(filepath)
    larsdat = ibw['wave']['wData'].T
    (nch, ny, nx) = larsdat.shape
    note_afmmeta = ibw['wave']['note'].decode('ascii', 'ignore').split('\r')
    numinfo = len(note_afmmeta)
    meas_Date_str = ''
    meas_Time_str = ''
    for i in range(0, numinfo):
        meta_pair = note_afmmeta[i].split(':')
        #        print(meta_pair)
        if (meta_pair[0] == 'FastScanSize'):
            dx = str2float(meta_pair[1]) * 1e3
        elif (meta_pair[0] == 'SlowScanSize'):
            dy = str2float(meta_pair[1]) * 1e3
        elif (meta_pair[0] == 'ScanSpeed'):
            velostr = meta_pair[1]
        elif (meta_pair[0] == 'Date'):
            meas_Date_str = meta_pair[1]
        elif (meta_pair[0] == 'Time'):
            meas_Time_str = meta_pair[1] + ':' + meta_pair[2]
    date_time = meas_Date_str + ', ' + meas_Time_str

    zsensorstr = ibw['wave']['labels'][2][5].decode('ascii', 'ignore')
    zpiezostr = ibw['wave']['labels'][2][1].decode('ascii', 'ignore')
    zsensor_map = larsdat[4] * 1e6  # micron
    zpiezo_map = larsdat[0] * 1e6
    print(date_time)
    print(zsensorstr)
    print(velostr)
    return nx, ny, dx, dy, zsensor_map, zpiezo_map, date_time, zsensorstr, zpiezostr, velostr


#
def read_asciisdf(content):
    xpixels = 0
    ypixels = 0
    dx = 0
    dy = 0
    zmap2D = []
    measdate = ''
    sdfrecords = content.decode('ascii', errors='ignore').split('*\r\n')
    headlines = sdfrecords[0].split('\r\n')
    for k in range(0, len(headlines)):
        print(k, '.)', headlines[k])
        headinfo = headlines[k].split('=')
        print(headinfo)
        if (headinfo[0].find('ManufacID') > -1):
            measdate = measdate + headinfo[1].strip()
        elif (headinfo[0].find('CreateDate') > -1):
            print(headinfo[1])
            listinfo = list(headinfo[1].strip())
            listnew = [',', ' ', listinfo[4], listinfo[5], \
                       listinfo[6], listinfo[7], '-', listinfo[2], \
                       listinfo[3], '-', listinfo[0], listinfo[1], ', ', \
                       listinfo[8], listinfo[9], ':', listinfo[10], listinfo[11], ' ', 'h']
            for char in listnew:
                measdate = measdate + char
        elif headinfo[0].find('NumPoints') > -1:
            xpixels = int(headinfo[1])
        elif headinfo[0].find('NumProfiles') > -1:
            ypixels = int(headinfo[1])
        elif (headinfo[0].find('Xscale') > -1) or (headinfo[0].find('X-scale') > -1):
            dx = str2float(headinfo[1]) * 1e6
        elif (headinfo[0].find('Yscale') > -1) or (headinfo[0].find('Y-scale') > -1):
            dy = str2float(headinfo[1]) * 1e6
        elif (headinfo[0].find('Zscale') > -1) or (headinfo[0].find('Z-scale') > -1):
            z2micron = str2float(headinfo[1]) * 1e6
        elif (headinfo[0].find('DataType') > -1) or (headinfo[0].find('Data type') > -1):
            datatype = int(headinfo[1])
    zmap2D = np.zeros((ypixels, xpixels))
    str_profiles = sdfrecords[1].split('\r\n')
    for i_profile in range(0, ypixels):
        hlplist = str_profiles[i_profile].split()
        for i_pt in range(0, xpixels):
            zmap2D[i_profile][i_pt] = z2micron * str2float(hlplist[i_pt])
    return xpixels, ypixels, dx, dy, zmap2D, measdate


#
def read_sdf(content):
    versionnumber = content[0:8].decode('ascii', errors='ignore')
    print(versionnumber)
    if (versionnumber.find('aISO') > -1):
        xpixels, ypixels, dx, dy, zmap2D, measdate = read_asciisdf(content)
    else:
        print('no implementation for binary sdf existing so far')
        xpixels = 0
        ypixels = 0
        dx = 0
        dy = 0
        zmap2D = []
        measdate = ''
    return xpixels, ypixels, dx, dy, zmap2D, measdate


#
def read_bcrf(content):
    hsize_strtry = '2048'
    hsize_n = int(hsize_strtry)
    header = content[0:hsize_n].decode('utf8', errors='ignore')
    headlines = header.split('\n')
    for k in range(0, len(headlines)):
        headinfo = headlines[k].split('=')
        if headinfo[0].find('xpixels') > -1:
            xpixels = int(headinfo[1])
        elif headinfo[0].find('ypixels') > -1:
            ypixels = int(headinfo[1])
        elif headinfo[0].find('xlength') > -1:
            xlength = str2float(headinfo[1])
        elif headinfo[0].find('ylength') > -1:
            ylength = str2float(headinfo[1])
        elif headinfo[0].find('scanspeed') > -1:
            scanspeed = str2float(headinfo[1])
        elif headinfo[0].find('bit2nm') > -1:
            bit2nm = str2float(headinfo[1])
            print(headinfo[1], 'bit2nm', bit2nm)
        elif headinfo[0].find('intelmode') > -1:
            intelmode = int(headinfo[1])
        elif headinfo[0].find('xunit') > -1:
            xunit = 1
            if (headinfo[1].find('nm') > -1):
                xunit = 1e-3
            elif (headinfo[1].find('mm') > -1):
                xunit = 1e3
        elif headinfo[0].find('yunit') > -1:
            yunit = 1
            if (headinfo[1].find('nm') > -1):
                yunit = 1e-3
            elif (headinfo[1].find('mm') > -1):
                yunit = 1e3
        elif headinfo[0].find('zunit') > -1:
            zunit = 1
            if (headinfo[1].find('nm') > -1):
                zunit = 1e-3
            elif (headinfo[1].find('mm') > -1):
                zunit = 1e3
        elif headinfo[0].find('fileformat') > -1:
            bcrf_str = headinfo[1]
        elif headinfo[0].find('headersize') > -1:
            hsize_n = int(headinfo[1])
    # 32 bit floating point
    if (bcrf_str.find('bcr') > -1):
        print('headersize: ', hsize_n)
        if (intelmode == 1):
            #           big endian
            if (bcrf_str.find('bcrf') > -1):
                zmapf32 = np.frombuffer(content, dtype=np.float32, count=xpixels * ypixels, offset=hsize_n)
            else:
                zmapf32 = np.frombuffer(content, dtype=np.int16, count=xpixels * ypixels, offset=hsize_n)
        else:
            #           little endian
            num32 = xpixels * ypixels
            zmapf32 = np.zeros(num32)
            buf = np.frombuffer(content, dtype=np.uint8, count=-1, offset=hsize_n)
            #            print('len(buf)', len(buf))
            #            print('4*num32', 4*num32)
            for k in range(0, num32):
                k4 = 4 * k
                data_bytes = np.array([buf[k4 + 3], buf[k4 + 2], buf[k4 + 1], buf[k4 + 0]], dtype=np.uint8)
                zmapf32[k] = data_bytes.view(dtype=np.float32)
        zmap2D = zunit * bit2nm * zmapf32.reshape((ypixels, xpixels))
        dx = xunit * xlength / xpixels
        dy = yunit * ylength / ypixels
    else:
        xpixels = 0
        ypixels = 0
        dx = 0
        dy = 0
        zmap2D = []
    return xpixels, ypixels, dx, dy, zmap2D


#
def read_sur(content):
    little_endian = 1

    dx = 0
    dy = 0
    zmap2D = []
    measdate = ''
    header_size = 512
    #
    intsize_char = np.frombuffer(content, dtype=np.uint8, count=2, offset=98)
    if (int(intsize_char[1]) == 0):
        little_endian = 0
    intsize = int(intsize_char[0]) + int(intsize_char[1])
    if (little_endian == 0):
        print('big')
        num_data = np.frombuffer(content, dtype=np.uint32, count=3, offset=108)
        dx = np.frombuffer(content, dtype=np.float32, count=1, offset=120)[0]
        dy = np.frombuffer(content, dtype=np.float32, count=1, offset=124)[0]
        dz = np.frombuffer(content, dtype=np.float32, count=1, offset=128)[0]
    else:
        print('little')
        num_char = np.frombuffer(content, dtype=np.uint8, count=12, offset=108)
        d_char = np.frombuffer(content, dtype=np.uint8, count=12, offset=120)
        num_data = [
            np.array([num_char[3], num_char[2], num_char[1], num_char[0]], dtype=np.uint8).view(dtype=np.int)[0], \
            np.array([num_char[7], num_char[6], num_char[5], num_char[4]], dtype=np.uint8).view(dtype=np.int)[0], \
            np.array([num_char[11], num_char[10], num_char[9], num_char[8]], dtype=np.uint8).view(dtype=np.int)[0]]
        dx = np.array([d_char[3], d_char[2], d_char[1], d_char[0]], dtype=np.uint8).view(dtype=np.float32)[0]
        dy = np.array([d_char[7], d_char[6], d_char[5], d_char[4]], dtype=np.uint8).view(dtype=np.float32)[0]
        dz = np.array([d_char[11], d_char[10], d_char[9], d_char[8]], dtype=np.uint8).view(dtype=np.float32)[0]
    #    print(dx, dy, dz)
    #
    if (int(content[132 + 3 * 16]) == 109):
        dx = 1e3 * dx
    elif (int(content[132 + 3 * 16]) == 110):
        dx = 1e-3 * dx
    #
    if (int(content[132 + 4 * 16]) == 109):
        dy = 1e3 * dy
    elif (int(content[132 + 4 * 16]) == 110):
        dy = 1e-3 * dy
    #
    if (int(content[132 + 5 * 16]) == 109):
        dz = 1e3 * dz
    elif (int(content[132 + 5 * 16]) == 110):
        dz = 1e-3 * dz
    #
    if (intsize == 32):
        data_type = np.int32
    elif (intsize == 16):
        data_type = np.int16
    else:
        print('invalid data type', intsize)
    numbyte = int(intsize / 8)
    if (little_endian == 0):
        zmap_array = np.frombuffer(content, dtype=data_type, count=num_data[2], offset=header_size)
    else:
        buf = np.frombuffer(content, dtype=np.uint8, count=numbyte * num_data[2], offset=header_size)
        zmap_array = np.zeros(0, dtype=data_type)
        for k in range(0, num_data[2]):
            k4 = numbyte * k
            if numbyte == 4:
                data_bytes = np.array([buf[k4 + 3], buf[k4 + 2], buf[k4 + 1], buf[k4 + 0]], dtype=np.uint8)
            else:
                data_bytes = np.array([buf[k4 + 1], buf[k4 + 0]], dtype=np.uint8)
            zmap_array[k] = data_bytes.view(dtype=data_type)
    zmap2D = dz * zmap_array.reshape((num_data[1], num_data[0]))
    #    print(dx, dy, dz)
    return num_data[0], num_data[1], dx, dy, zmap2D, measdate


#
# NMM
#
def read_NMMgaoliang(content):
    dx = 0
    dy = 0
    zmap2D = []

    scanspeed_str = '?'
    sensortype = '?'
    head_data = content.split('//DATA')
    headlines = head_data[0].split('\r\n')
    data_strlist = head_data[1].split()
    num_headlines = len(headlines)
    num_data = len(data_strlist)
    print('=====================\n', num_data, '--\n', data_strlist[0])
    for k in range(0, num_headlines):
        headinfo = headlines[k].split('=')
        #        print('headinfo: ', headinfo)
        if headinfo[0].find('//Pixels per Row') > -1:
            headinfoB = headinfo[1].split()
            nx = int(headinfoB[0])
        elif headinfo[0].find('//Scan Lines') > -1:
            ny = int(headinfo[1])
        elif headinfo[0].find('//xp') > -1:
            print('headinfo xp', headinfo[1])
            dy = str2float(headinfo[1]) * 1e3
            print('dy ', dy)
        elif headinfo[0].find('//yp') > -1:
            dx = str2float(headinfo[1]) * 1e3
            print('dx ', dx)
        elif headinfo[0].find('//ScanSpeed') > -1:
            scanspeed_str = headinfo[1]
            print('scanspeed ', scanspeed_str)
        elif headinfo[0].find('//Detection Head') > -1:
            headinfoB = headinfo[0].split(':')
            sensortype = headinfoB[1]
            print('sensortype ', sensortype)
    measinfo = 'Sensor: ' + sensortype + ' scan speed: ' + scanspeed_str
    print('measinfo: ', measinfo)
    print(data_strlist[0:10])
    zmap2D = 1e-3 * np.array(data_strlist).astype(np.float).reshape(ny, nx)
    print(zmap2D[0][0:10])
    return nx, ny, dx, dy, zmap2D, measinfo


#
def identifybadborderlines(zval2d, ny):
    istart = 0
    in_valid_pts = np.where(np.isnan(zval2d[istart]))[0]
    num_cut = ny // 3
    while len(in_valid_pts) > num_cut:
        istart += 1
        in_valid_pts = np.where(np.isnan(zval2d[istart]))[0]
    iend = ny - 1
    in_valid_pts = np.where(np.isnan(zval2d[iend]))[0]
    while len(in_valid_pts) > num_cut:
        iend -= 1
        in_valid_pts = np.where(np.isnan(zval2d[iend]))[0]
    return istart, iend


def interpol_csaps(zmatrix, wmatrix, dx, dy, smoothparam):
    pltflag = 0
    (ny, nx) = zmatrix.shape
    istart_y, iend_y = identifybadborderlines(zmatrix, ny)
    istart_x, iend_x = identifybadborderlines(zmatrix.T, nx)
    z_xdir = zmatrix[istart_y:iend_y, istart_x:iend_x]
    wfinal = wmatrix[istart_y:iend_y, istart_x:iend_x]
    print('istart_y:iend_y,istart_x:iend_x', istart_y, ':', iend_y, ',', istart_x, ':', iend_x)
    (ny, nx) = z_xdir.shape
    print('nx', nx, ' - ', iend_x - istart_x)
    x_knots = np.arange(0, nx) * dx
    y_knots = np.arange(0, ny) * dy
    itotalline = []
    for iy in range(0, ny):
        i_nans = np.where(np.isnan(z_xdir[iy]))[0]
        if (len(i_nans) > 0):
            if (len(i_nans) <= ny // 3):
                i_valid = np.where(np.isnan(z_xdir[iy]) == False)[0]
                z_fill = csaps(x_knots[i_valid], z_xdir[iy][i_valid], \
                               x_knots[i_nans], smooth=smoothparam)
                z_xdir[iy][i_nans] = z_fill
            else:
                itotalline.append(iy)

    z_ydir = z_xdir.T
    for ix in range(0, nx):
        i_nans = np.where(np.isnan(z_ydir[ix]))[0]
        if (len(i_nans) > 0):
            if (len(i_nans) <= ny // 3):
                i_valid = np.where(np.isnan(z_ydir[ix]) == False)[0]
                z_fill = csaps(y_knots[i_valid], z_ydir[ix][i_valid], \
                               y_knots[i_nans], smooth=smoothparam)
                z_ydir[ix][i_nans] = z_fill
    zfinal = z_ydir.T

    if pltflag:
        plt.figure(1)
        plt.imshow(zmatrix)
        plt.title('zmatrix')
        plt.figure(10)
        plt.imshow(z_xdir)
        plt.title('z_xdir')
    if 0:
        ny_begin = 0
        ny_end = ny - 1
        if len(itotalline) > 0:
            print('ny_begin: ', ny_begin)
            print(itotalline)
            whileflag = 1
            while (itotalline[ny_begin] == ny_begin) and whileflag:
                if (ny_begin < len(itotalline) - 1):
                    ny_begin += 1
                else:
                    whileflag = 0
                print('this ', ny_begin)
            n_last = len(itotalline) - 1
            while itotalline[n_last] == ny_end:
                ny_end -= 1
                n_last -= 1

        z_xdir_crop = np.copy(z_xdir[ny_begin:ny_end])
        wmatrix_crop = np.copy(wmatrix[ny_begin:ny_end])
        (ny, nx) = z_xdir_crop.shape
        y_knots = np.arange(0, ny) * dy
        if pltflag:
            plt.figure(2)
            plt.imshow(z_xdir_crop)
            plt.title('z_xdir_crop')

        i_nans = np.where(np.isnan(z_xdir))[0]
        if len(i_nans) > 0:
            print('height map has too many invalid points: ', len(i_nans))
            z_ydir = z_xdir_crop.T
            for ix in range(0, nx):
                i_nans = np.where(np.isnan(z_ydir[ix]))[0]
                if (len(i_nans) > 0):
                    if (len(i_nans) < ny // 10):
                        i_valid = np.where(np.isnan(z_ydir[ix]) == False)[0]
                        z_fill = csaps(y_knots[i_valid], z_ydir[ix][i_valid], \
                                       y_knots[i_nans], smooth=smoothparam)
                        z_ydir[ix][i_nans] = z_fill
            zfinal = z_ydir.T
        else:
            zfinal = z_xdir_crop
    if pltflag:
        plt.figure(3)
        plt.imshow(zfinal)
        plt.title('zfinal')
        plt.show()
    return zfinal, wfinal


#
def invalid_data_weight(height_map):
    (ny, nx) = height_map.shape
    weights = np.ones((ny, nx))
    i_nans = np.where(np.isnan(height_map))
    weights[i_nans] = 0
    return weights, len(i_nans[0])


#
def read_lextintensity(filename):
    dx = 0
    dy = 0

    magnification = 0
    measdate = ''
    f = open(filename, "rb")
    content = f.read()
    f.close()
    nx, ny, endheightmap, heightperpix, magnification, measdate = read_lextinfo(content)
    intensity_map = read_lextimg(filename, nx, ny, endheightmap, 0)
    dx = heightperpix[0]
    dy = heightperpix[1]
    return dx, dy, intensity_map


#
def read_microscopedata(filename, userscalecorr, interpolflag):
    #
    # dx, dy: sampling/pixel distances in micron
    # height_map in micron
    #
    fnlen = len(filename)
    dx = 0
    dy = 0
    height_map = 0
    magnification = 0
    measdate = ''
    f = open(filename, "rb")
    content = f.read()
    f.close()
    if filename[fnlen - 4:fnlen].find('ibw') > -1:
        if (withigor == 1):
            # returns all data in micron
            nx, ny, dx, dy, height_map, zpiezo_map, measdate, \
            heightstr, zpiezostr, velostr = read_ibw(filename)
            measdate = measdate + '\n' + heightstr + ', v=' + 'Scanspeed: ' + velostr
        else:
            print('ibw not readable: package igor required but not installed')
            print('pip install igor')
            print('also with this command in spyder console I/A')
    elif filename[fnlen - 4:fnlen].find('plu') > -1:
        # returns all data in micron
        nx, ny, dx, dy, height_map, measdate = read_plu(content)
    elif filename[fnlen - 5:fnlen].find('bcr') > -1:
        # returns all data in micron
        measdate = ''
        nx, ny, dx, dy, height_map = read_bcrf(content)
    elif filename[fnlen - 4:fnlen].find('sdf') > -1:
        # returns all data in micron
        nx, ny, dx, dy, height_map, measdate = read_sdf(content)
    elif filename[fnlen - 4:fnlen].find('sur') > -1:
        # returns all data in micron
        nx, ny, dx, dy, height_map, measdate = read_sur(content)
    elif filename[fnlen - 4:fnlen].find('asc') > -1:
        # returns all data in micron
        nx, ny, dx, dy, height_map, measdate = read_asc(content)
    else:
        if content.find(b'LEXT') > -1:
            # returns all data in micron
            nx, ny, endheightmap, heightperpix, magnification, measdate = read_lextinfo(content)
            height_map = read_lextimg(filename, nx, ny, endheightmap, 1)
            # use n/(n-1) to get pixel distance from pixel width (='height')
            # the it agrees with gwyddion's result, but which we reject to use!
            # HeightperPixel is already the distance!!
            dx = heightperpix[0]
            dy = heightperpix[1]
            height_map = height_map * heightperpix[2]
        elif content[0:15].find(b'LRSPM') > -1:
            nx, ny, dx, dy, height_map, measdate = read_NMMgaoliang(content.decode('ascii'))

    weight_map, num_invalid = invalid_data_weight(height_map)
    print(f'num_invalid: {num_invalid} / {height_map.size}')
    if (num_invalid > 0) and interpolflag:
        height_map, weight_map = interpol_csaps(height_map, weight_map, dx, dy, 0.7)
    print('userscalecorr: ', userscalecorr)
    dx *= userscalecorr[0]
    dy *= userscalecorr[1]
    height_map *= userscalecorr[2]
    return dx, dy, height_map, weight_map, magnification, measdate


#
def read_HRTS(content):
    contentlines = content.split('\n')
    ix = -1
    iy = -1
    iz = -1
    header = 1
    iline = 0
    while header == 1:
        if (contentlines[iline].find('GROUP3.BEI.CurrentPosition') > -1):
            columnlabels = contentlines[iline].split()
            ncol = len(columnlabels)
            header = 0
            for k in range(0, ncol):
                if (columnlabels[k].find('GROUP3.BEI.CurrentPosition') > -1):
                    ix = k
                elif (columnlabels[k].find('GROUP5.BEI.CurrentPosition') > -1):
                    iy = k
                elif (columnlabels[k].find('GROUP2.MAXON.CurrentPosition') > -1):
                    iz = k
        iline += 1
    print(ix, iy, iz)
    nline = len(contentlines)
    if len(contentlines[nline - 1].split()) < ix:
        nline -= 1
    data = np.zeros((3, nline - iline))
    for k in range(iline, nline):
        hlpstr = contentlines[k].split()
        data[0][k - iline] = str2float(hlpstr[ix])
        data[1][k - iline] = str2float(hlpstr[iy])
        data[2][k - iline] = str2float(hlpstr[iz])
    return data


#
def read_pointcloud(filename):
    f = open(filename, 'rb')
    content = f.read().decode('ascii', errors='ignore')
    f.close()
    if content.find('MAXON.CurrentPosition') > -1:
        print('found MAXON')
        data = read_HRTS(content)  # millimeter
        data *= 1e3  # micron
    return data


#
def lateral_axis(data_lat):
    len_lat = np.max(data_lat) - np.min(data_lat)
    diff = np.diff(data_lat)
    delta = np.abs(np.mean(diff))
    s_delta = np.std(diff, ddof=1)
    num = int(np.round(len_lat / delta) + 1)
    print(len_lat, delta, s_delta)
    return len_lat, delta, s_delta, num


#
def pointcloud_to_heightmap(dat_x, dat_y, dat_z):
    len_x, delta_x, s_delta_x, n_x = lateral_axis(dat_x)
    len_y, delta_y, s_delta_y, n_y = lateral_axis(dat_y)
    n_all = len(dat_z)
    print(dat_z.shape, n_x, n_y)
    if delta_x > s_delta_x:
        print('scan is in x dir')
        n2 = n_x
        n1 = n_all // n_x
    elif delta_y > s_delta_y:
        print('scan is in y dir')
        n2 = n_y
        n1 = n_all // n_y
    z_map = dat_z.reshape(n1, n2)
    return delta_x, delta_y, z_map
#
