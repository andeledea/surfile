import numpy as np
import matplotlib.pyplot as plt
import copy

import scipy.interpolate as interpolate


def morph(profile_x, profile_y, radius):
    spacing = profile_x[1] - profile_x[0]
    n_radius = int(radius / spacing)
    n_samples = len(profile_x)

    fig, ax = plt.subplots()
    ax.axis('equal')

    profile_out = profile_y - radius

    for i in range(n_radius, n_samples-n_radius-1):
        loc_x = np.linspace(profile_x[i-n_radius], profile_x[i+n_radius], 1000)
        loc_p = np.interp(loc_x, profile_x[i - n_radius:i + n_radius], profile_y[i - n_radius:i + n_radius])

        alpha = profile_x[i]
        beta = profile_y[i] - radius  # start under the profile

        cerchio = np.sqrt(-(alpha ** 2 - radius ** 2) + 2 * alpha * loc_x - loc_x ** 2) + beta

        dbeta = radius / 500
        disp = 0
        up = len(np.argwhere((cerchio - loc_p) > 0))
        while up > 5:
            cerchio -= dbeta
            disp += dbeta
            up = len(np.argwhere((cerchio - loc_p) > 0))

            # ax.clear()
            # ax.plot(loc_x, cerchio, loc_x, loc_p)
            # plt.axis('equal')
            # plt.draw()
            # plt.pause(0.1)
        profile_out[i] -= disp
        print(f'Processing {i}, disp {disp}')

    ax.plot(profile_x, profile_y, profile_x, profile_out)
    plt.show()
        

if __name__ == '__main__':
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (12, 9)

    x = np.linspace(0, 5, 1000)
    f = 1 * np.exp(-2 * (3 * x - 1.5) ** 2) + np.exp(-2 * (0.4 * x - 2.5) ** 2)

    plt.plot(x, f)
    plt.show()

    morph(x, f, 0.2)

