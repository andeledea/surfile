import numpy as np
import matplotlib.pyplot as plt
import copy

import scipy.interpolate as interpolate
from alive_progress import alive_bar


def morph(profile_x, profile_y, radius):
    spacing = profile_x[1] - profile_x[0]
    n_radius = int(radius / spacing)
    n_samples = len(profile_x)

    fig, ax = plt.subplots()
    ax.axis('equal')

    filler_L = np.ones(n_radius) * profile_y[0]
    filler_R = np.ones(n_radius) * profile_y[-1]

    profile_x_filled = np.arange(start=profile_x[0] - radius, stop=profile_x[-2] + radius, step=spacing)
    profile_y_filled = np.concatenate([filler_L, profile_y, filler_R])

    profile_out = profile_y_filled - radius

    with alive_bar(n_samples, force_tty=True,
                   title='Morph', theme='smooth',
                   elapsed_end=True, stats_end=True, length=30) as bar:
        for i in range(n_radius, n_samples + n_radius):
            loc_x = np.linspace(profile_x_filled[i-n_radius], profile_x_filled[i+n_radius], 1000)
            loc_p = np.interp(loc_x, profile_x_filled[i - n_radius:i + n_radius], profile_y_filled[i - n_radius:i + n_radius])

            alpha = profile_x_filled[i]
            beta = profile_out[i]  # start under the profile

            cerchio = np.sqrt(-(alpha ** 2 - radius ** 2) + 2 * alpha * loc_x - loc_x ** 2) + beta

            dbeta = -radius / 2
            disp = 0

            bar()

            up = len(np.argwhere((cerchio - loc_p) > 0))
            if up > 1:
                while np.abs(dbeta) > radius / 500:
                    cerchio += dbeta
                    disp -= dbeta
                    up = len(np.argwhere((cerchio - loc_p) > 0))

                    if (dbeta < 0 and up == 0) or (dbeta > 0 and up != 0):
                        dbeta = -dbeta / 2

                    # ax.clear()
                    # ax.plot(loc_x, cerchio, loc_x, loc_p)
                    # plt.axis('equal')
                    # plt.draw()
                    # plt.pause(5)
                profile_out[i] -= disp
                # print(f'Processing {i}, disp {disp}')

    profile_out = profile_out[n_radius: -(n_radius)]
    ax.plot(profile_x, profile_y, profile_x_filled, profile_y_filled, '--', profile_x, profile_out)
    plt.show()
        

if __name__ == '__main__':
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (12, 9)

    x = np.linspace(0, 5, 10000)
    f = 1 * np.exp(-2 * (3 * x - 1.5) ** 2) + 0.3 * np.sin(-2 * (1.3 * x - 2.5) ** 2)

    plt.plot(x, f)
    plt.show()

    morph(x, f, 0.1)

