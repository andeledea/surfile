"""
'surfile.stitcher'
- implementation of surface stitching methods

@author: Andrea Giura
"""
import copy

from matplotlib import patches, cm

from surfile import surface, funct
from scipy import optimize, signal, ndimage

import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d


def _composeFigure(alpha, beta, dx, dy, sp):
    """
    Compose the stitched image (stitch in x direction)

    Parameters
    ----------
    alpha : np.array
        The left figure
    beta : np.array
        The right figure
    dx : int
        The optimal x direction displacement
    dy : int
        The optimal y direction displacement
    sp : int
        The overlap of the 2 images %

    Returns
    -------
    composed : np.array
        The composed array
    """
    # patches creation
    sp = int(alpha.shape[1] * (sp / 100))
    print(f'{dx=} {dy=}')
    alpha = np.roll(alpha, shift=(dx, dy), axis=(1, 0))
    left = alpha[:, :-sp // 2]
    right = beta[:, sp // 2:]

    st = np.hstack((left, right))

    fig, (ax, bx) = plt.subplots(nrows=2, ncols=1)
    ax.imshow(alpha[:, -sp:])
    bx.imshow(beta[:, :sp])

    fig2, cx = plt.subplots(nrows=1, ncols=1)
    cx.imshow(st, cmap=cm.viridis)
    plt.show()


class SurfaceStitcher:
    @staticmethod
    def stitchMinimizeNorm(surl, surr, stitchPrc=20, pixelScan=40, bplt=False):
        """
        Given 2 surfaces finds the best allignement
        by minimizing the norm2 of the difference

        Parameters
        ----------
        surl : surface.Surface
            The left image to be stitched
        surr : surface.Surface
            The right image to be stitched
        stitchPrc : int
            the percentage of the image overlapping
        pixelScan: int
            the number of pixel the method tryes to displace the images
            an higher number results in longer excution time
        bplt : bool
            If true plots the stitching process, limits the radius scan to 5 pixels
            Use this only to see graphically and very slowly what this function does.

        Returns
        -------
        surface.Surface
            The stitched image
        """
        # Given starting displacement (0, 0) in x and y
        # move surr % of stitching.py over the other % surl
        # and minimize surr(x - a; y - b) - surl(x, y)

        # We need a function that given a, b moves surr
        # and subtracts surr moved from surl

        # find the interested zones to be stitched
        lZone = surl.Z[:, 1 + int(surl.Z.shape[1] * (1 - stitchPrc / 100)):]
        rZone = surl.Z[:, :int(surl.Z.shape[1] * (stitchPrc / 100))]
        # rZone = np.roll(lZone, 30, axis=1)  # used for testing

        ny, nx = lZone.shape

        if bplt:
            fig, (ax, bx, cx) = plt.subplots(nrows=1, ncols=3)
            plot_data = ax.imshow(lZone)
            bx.imshow(lZone)
            cx.imshow(rZone)
            rectb = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
            rectc = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
            bx.add_patch(rectb)
            cx.add_patch(rectc)

            funct.persFig([ax, bx, cx], xlab='x [pixels]', ylab='y [pixels]')

            plt.show(block=False)

        def move(disp):
            a, b = disp[0], disp[1]
            print(a, b)
            if a >= 0:
                laa, raa, lab, rab = a, nx, 0, nx - a
            else:
                a = -a
                laa, raa, lab, rab = 0, nx - a, a, nx

            if b >= 0:
                lba, rba, lbb, rbb = b, ny, 0, ny - b
            else:
                b = -b
                lba, rba, lbb, rbb = 0, ny - b, b, ny

            alpha_patch = lZone[lba: rba, laa: raa]
            beta_patch = rZone[lbb: rbb, lab: rab]

            diff = alpha_patch - beta_patch

            if bplt:
                plot_data.set_data(diff)

                rectb.set_xy((laa, lba))
                rectb.set_width(raa - laa)
                rectb.set_height(rba - lba)
                bx.add_patch(rectb)
                rectc.set_xy((lab, lbb))
                rectc.set_width(rab - lab)
                rectc.set_height(rbb - lbb)

                fig.canvas.draw()
                plt.pause(0.05)

            return np.linalg.norm(diff)  # maybe an ssd (sum of square difference with a gaussian kernel is better)

        # a and b are ints since they rapresent pixel translations
        nPixelMaxDisp = pixelScan if not bplt else 5
        bestTranslation = optimize.brute(
            move,
            ranges=((slice(-nPixelMaxDisp, nPixelMaxDisp, 1),) * 2),
            disp=True,
            finish=None
        )

        print(bestTranslation)

    @staticmethod
    def stitchCorrelation(surl, surr, stitchPrc=20, samplingPrc=50, bplt=False):
        """
        Finds the best allignment between surl and surr
        by calculating the maximum of the cross correlation
        
        Parameters
        ----------
        samplingPrc : int
            The percentage of the points of the overimposed
            surfaces that is sampled from the arrays
        surl : surface.Surface
            The left image to be stitched
        surr : surface.Surface
            The right image to be stitched
        stitchPrc : int
            the percentage of the image overlapping
        bplt : bool
            If true plots the stitched image
        """
        # find the interested zones to be stitched
        # TODO: what if the lZone and rZone have different sizes ?? (often they dont)
        lZone = surl.Z[:, 1 + int(surl.Z.shape[1] * (1 - stitchPrc / 100)):]
        rZone = surr.Z[:, :int(surr.Z.shape[1] * (stitchPrc / 100))]

        # lZone = np.diff(lZone)
        # rZone = np.diff(rZone)

        # take a central patch from the second image
        center_x, center_y = lZone.shape[0] // 2, lZone.shape[1] // 2
        size_x, size_y = lZone.shape[0] * samplingPrc // 200, lZone.shape[1] * samplingPrc // 200
        sampleL = lZone[center_x - size_x // 2: center_x + size_x // 2,
                  center_y - size_y // 2: center_y + size_y // 2]
        sampleR = rZone[center_x - size_x // 2: center_x + size_x // 2,
                  center_y - size_y // 2: center_y + size_y // 2]

        # correlate the patch with the first image to find its position
        ccL = signal.correlate2d(lZone, sampleR, mode='valid')  # , boundary='fill', fillvalue=np.average(lZone))
        ccR = signal.correlate2d(rZone, sampleL, mode='valid')  # , boundary='fill', fillvalue=np.average(lZone))

        ML = np.argmax(ccL)
        yML, xML = np.unravel_index(ML, ccL.shape)
        print(f'{ML=} {xML=} {yML=}')

        MR = np.argmax(ccR)
        yMR, xMR = np.unravel_index(MR, ccR.shape)
        print(f'{MR=} {xMR=} {yMR=}')

        bestLTranslation = [ccL.shape[1] // 2 - xML, ccL.shape[0] // 2 - yML]
        bestRTranslation = [ccR.shape[1] // 2 - xMR, ccR.shape[0] // 2 - yMR]

        meanTranslation = [(bestLTranslation[i] - bestRTranslation[i]) // 2 for i in [0, 1]]
        print(f'{bestLTranslation=}\n{bestRTranslation=}\n{meanTranslation=}')

        flippedccR = np.flip(ccR)
        cross_cc = ccL * flippedccR
        M = np.argmax(cross_cc)
        yM, xM = np.unravel_index(M, cross_cc.shape)
        bestMeanTranslation = [cross_cc.shape[1] // 2 - xM, cross_cc.shape[0] // 2 - yM]
        print(f'\n\n{M=} {xM=} {yM=}')
        print(f'{bestMeanTranslation=}')

        _composeFigure(surl.Z, surr.Z,
                       dx=bestMeanTranslation[0], dy=bestMeanTranslation[1],
                       sp=stitchPrc)

        if bplt:
            fig, ((ax, bx, cx), (dx, ex, fx)) = plt.subplots(nrows=2, ncols=3)
            ax.imshow(ccL)
            ax.plot(xML, yML, 'ro', ms=5)
            bx.imshow(lZone)
            cx.imshow(sampleR)

            dx.imshow(ccR)
            dx.plot(xMR, yML, 'ro', ms=5)
            ex.imshow(rZone)
            fx.imshow(sampleL)
            funct.persFig([ax, bx, cx, dx, ex, fx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')

            plt.get_current_fig_manager().full_screen_toggle()

            fig2, (lx, mx, nx) = plt.subplots(nrows=1, ncols=3)
            lx.imshow(ccL)
            mx.imshow(np.flip(ccR))
            nx.imshow(ccL * np.flip(ccR))
            nx.plot(xM, yM, 'r.', ms=5)

            plt.show()

    # @staticmethod
    # def stitchSSDminimize(surl, surr, stitchPrc=20, bplt=False):
    #     """
    #     Match image locations using SSD minimization.
    #
    #     Areas from `surl` are matched with areas from `surr`. These areas
    #     are defined as patches located around pixels with Gaussian
    #     weights.
    #
    #     https://scikit-image.org/docs/stable/auto_examples/registration/plot_stitching.html
    #
    #     Parameters
    #     ----------
    #     surl : surface.Surface
    #         The left image to be stitched
    #     surr : surface.Surface
    #         The right image to be stitched
    #     stitchPrc : int
    #         the percentage of the image overlapping
    #     bplt : bool
    #         If true plots the stitched image
    #
    #     Returns
    #     -------
    #     match_coords: (2, m) array
    #         The points in `coordsR` that are the closest corresponding matches to
    #         those in `coordsL` as determined by the (Gaussian weighted) sum of
    #         squared differences between patches surrounding each point.
    #     """
    #     lZone = surl.Z[:, 1 + int(surl.Z.shape[1] * (1 - stitchPrc / 100)):]
    #     # rZone = surr.Z[:, :int(surr.Z.shape[1] * (stitchPrc / 100))]
    #     rZone = lZone
    #
    #     samplingPxls = 40
    #     sampleSpacing = 50
    #     samplingSdev = 5
    #
    #     startx = starty = samplingPxls
    #     stopx = lZone.shape[0] - samplingPxls
    #     stopy = lZone.shape[1] - samplingPxls
    #
    #     coordsL = np.mgrid[startx:stopx:sampleSpacing, starty:stopy:sampleSpacing].reshape(2, -1).T
    #     coordsR = np.mgrid[startx:stopx:sampleSpacing, starty:stopy:sampleSpacing].reshape(2, -1).T
    #
    #     y, x = np.mgrid[-samplingPxls:samplingPxls + 1, -samplingPxls:samplingPxls + 1]
    #     weights = np.exp(-0.5 * (x ** 2 + y ** 2) / samplingSdev ** 2)
    #     weights /= 2 * np.pi * samplingSdev * samplingSdev
    #
    #     match_list = []
    #     for rL, cL in coordsL:
    #         roiL = lZone[rL - samplingPxls:rL + samplingPxls + 1, cL - samplingPxls:cL + samplingPxls + 1]
    #         roiR_list = [rZone[rR - samplingPxls:rR + samplingPxls + 1,
    #                      cR - samplingPxls:cR + samplingPxls + 1] for rR, cR in coordsR]
    #         # sum of squared differences
    #         ssd_list = [np.sum(weights * (roiL - roiR) ** 2) for roiR in roiR_list]
    #         match_list.append(coordsL[np.argmin(ssd_list)])
    #
    #     print(match_list)
    #     return np.array(match_list)

    @staticmethod
    def stitchICP(surl, surr, stitchPrc=20):
        """
        Finds the best allignment between surl and surr
        by first registering approximatively the 2 images
        using a FGR feature matcher, and then improves
        the result by refining with an ICP registration

        see: http://www.open3d.org/docs/0.9.0/python_api/open3d.registration.html
        see: http://vladlen.info/papers/fast-global-registration.pdf

        Parameters
        ----------
        surl : surface.Surface
            The left image to be stitched
        surr : surface.Surface
            The right image to be stitched
        stitchPrc : int
            the percentage of the image overlapping
        """
        # find the interested zones to be stitched
        lZone = surl.Z[:, 1 + int(surl.Z.shape[1] * (1 - stitchPrc / 100)):]
        rZone = surr.Z[:, :int(surr.Z.shape[1] * (stitchPrc / 100))]

        # scale max dimention
        scale = 1
        if lZone.shape[1] > lZone.shape[0]:
            scaley = lZone.shape[0] * scale / lZone.shape[1]
            scalefactor = lZone.shape[1]

            X = np.linspace(0, scale, lZone.shape[1])
            Y = np.linspace(0, scaley, lZone.shape[0])
        else:
            scalex = lZone.shape[1] * scale / lZone.shape[0]
            scalefactor = lZone.shape[0]

            X = np.linspace(0, scalex, lZone.shape[1])
            Y = np.linspace(0, scale, lZone.shape[0])
        mesh_x, mesh_y = np.meshgrid(X, Y)

        def toPC(zone):
            xyz = np.zeros((np.size(mesh_x), 3))
            xyz[:, 0] = np.reshape(mesh_x, -1)
            xyz[:, 1] = np.reshape(mesh_y, -1)
            xyz[:, 2] = np.reshape(zone, -1)

            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            return pcd

        pcd_l = toPC(lZone)
        pcd_r = toPC(rZone)

        o3d.visualization.draw_geometries([pcd_l])
        o3d.visualization.draw_geometries([pcd_r])

        def draw_registration_result(source, target, transformation):
            source_temp = copy.deepcopy(source)
            target_temp = copy.deepcopy(target)
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
            source_temp.transform(transformation)
            o3d.visualization.draw_geometries([source_temp, target_temp])

        def preprocess_point_cloud(pcd, voxel_size):
            print(":: Downsample with a voxel size %.3f." % voxel_size)
            pcd_down = pcd.voxel_down_sample(voxel_size)

            radius_normal = voxel_size * 2
            print(":: Estimate normal with search radius %.3f." % radius_normal)
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

            radius_feature = voxel_size * 5
            print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            return pcd_down, pcd_fpfh

        def prepare_dataset(voxel_size):
            source = pcd_l
            target = pcd_r

            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
            return source, target, source_down, target_down, source_fpfh, target_fpfh

        def execute_global_registration(source_down, target_down, source_fpfh,
                                        target_fpfh, voxel_size):
            distance_threshold = voxel_size * 1.5
            print(":: RANSAC registration on downsampled point clouds.")
            print("   Since the downsampling voxel size is %.3f," % voxel_size)
            print("   we use a liberal distance threshold %.3f." % distance_threshold)
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold)
            )
            return result

        voxel_size = 0.02  # means 5cm for the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(voxel_size)

        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        print(result_ransac)
        print("Transformation is:")
        print(result_ransac.transformation * scalefactor)
        draw_registration_result(source_down, target_down,
                                 result_ransac.transformation)

        print("Apply point-to-point ICP")
        threshold = 0.02
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold,
            init=result_ransac.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation * scalefactor)
        draw_registration_result(source, target, reg_p2p.transformation)

        xtransl = int(reg_p2p.transformation[0, 3] * scalefactor)
        ytransl = int(reg_p2p.transformation[1, 3] * scalefactor)
        _composeFigure(surl.Z, surr.Z,
                       dx=xtransl, dy=ytransl,
                       sp=stitchPrc)
