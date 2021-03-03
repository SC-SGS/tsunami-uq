import scipy
from scipy.optimize import fmin
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


##############################################################
##################  auxiliary functions     ##################
##############################################################


def bump(t, c):
    # basically a gaussian pdf. [The 1/(theta*sqrt(2 pi)) is replaced by 'weight']
    theta = c[0]
    position = c[1]
    weight = c[2]
    ybump = weight * np.exp(-.5 * (t - position) ** 2 * theta ** -2)
    return ybump


def plot_all(wave_type, par, args):
    if wave_type == 'heights':
        withResidual = args[0]
        withNormalization = args[1]
        heights = par
        wave_function, c, t = heights_wave(heights, withResidual, withNormalization)

    elif wave_type == 'wave_9D':
        [heights, widths, positions] = par
        withNormalization = args[0]
        wave_function, c, t = wave_9D(heights, widths, positions, withNormalization)

    original_t, original_y = original_wave()

    plt.figure()
    plt.subplot(2, 1, 1)
    plot_wave(wave_function)
    plt.plot(original_t, original_y, 'gray', linestyle='--')
    plt.subplot(2, 1, 2)
    plot_bumps(heights, c, t)


def plot_wave(wave_function):
    T = np.linspace(0, 22.5, 1000)
    W = np.zeros(len(T))
    for i, t in enumerate(T):
        W[i] = wave_function(t)
    plt.plot(T, W)


def plot_bumps(heights, c, t):
    for k in range(len(heights)):
        plt.plot(t, heights[k]*bump(t, c[k, :]), label='bump {}'.format(k+1))

    # plt.gca().set_xlim(xlim)
    # plt.gca().set_ylim(ylim)
    # plt.ylabel('Heigt in m')
    # plt.xlabel('time in s')
    plt.legend()


def extract_bumps(nbump):
    # len(par) many Gauss bumps are fitted to the original wave and subtracted.
    # It remains the residual.

    t, y = original_wave()

    energy = np.trapz(y ** 2, t)

    residual = y.copy()
    c = np.zeros((nbump, 3))
    for k in range(nbump):
        maxid = np.argmax(np.abs(residual))
        c0 = np.array([1.5, t[maxid], residual[maxid]])

        def cost(c):
            ybump = bump(t, c)
            cost = np.sqrt(np.mean((ybump - residual) ** 2))
            return cost

        # fmin minimizes a function (using downhill simplex, no gradients needed)
        c[k, :] = fmin(cost, c0, disp=False)
        residual -= bump(t, c[k, :])
    return c, t, residual, energy

#########################################################
##################  wave functions     ##################
#########################################################


def original_wave():
    # load real wave data
    data = np.loadtxt(
        '/home/rehmemk/git/anugasgpp/Okushiri/data/boundary_wave_original.txt', skiprows=1)
    t = data[:, 0]
    y = data[:, 1]
    return t, y


def original_wave_interpolant():
    t, y = original_wave()
    wave_function = scipy.interpolate.interp1d(t, y, kind='zero')
    return wave_function


def heights_wave(heights, withResidual=0, withNormalization=1):
    # The wave originally used by Steve
    # Each Gauss bump is scaled according to the height parameters then the residual is added again (if withResidual=1)
    # and then the wave is scaled such that it has the same amount of energy (=l2 norm) as the original real
    # wave (if withNormalization=1)
    nbump = len(heights)
    c, t, residual, energy = extract_bumps(nbump)

    # reparametrize Gauss bumps

    # deform wave
    if withResidual == 1:
        ynew = residual.copy()
    else:
        ynew = np.zeros(len(residual))
    for k in range(nbump):
        ynew += heights[k] * bump(t, c[k, :])
    if withNormalization == 1:
        energynew = np.trapz(ynew ** 2, t)
        ynew = np.sqrt(energy / energynew) * ynew
    # elif withNormalization == 0:
    #     print("results are not normalized")

    wave_function = scipy.interpolate.interp1d(t, ynew, kind='zero')
    return wave_function, c, t


def cubic_heights_wave(heights, withResidual=0, withNormalization=1):
    # The wave originally used by Steve except that the wave interpolant is cubic
    # Each Gauss bump is scaled according to the height parameters then the residual is added again (if withResidual=1)
    # and then the wave is scaled such that it has the same amount of energy (=l2 norm) as the original real
    # wave (if withNormalization=1)
    nbump = len(heights)
    c, t, residual, energy = extract_bumps(nbump)

    # reparametrize Gauss bumps

    # deform wave
    if withResidual == 1:
        ynew = residual.copy()
    else:
        ynew = np.zeros(len(residual))
    for k in range(nbump):
        ynew += heights[k] * bump(t, c[k, :])
    if withNormalization == 1:
        energynew = np.trapz(ynew ** 2, t)
        ynew = np.sqrt(energy / energynew) * ynew
    # elif withNormalization == 0:
    #     print("results are not normalized")

    wave_function = scipy.interpolate.interp1d(t, ynew, kind='cubic')
    return wave_function, c, t


if __name__ == "__main__":
    wave_type = 'heights'
    heights = [1.0]*6
    par = heights
    withResidual = 0
    withNormalization = 1
    args = [withResidual, withNormalization]

    plot_all(wave_type, par, args)
    plt.show()
