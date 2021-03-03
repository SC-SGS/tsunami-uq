import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.interpolate import interp1d
import scipy
import os

import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ['Times']})

data = np.loadtxt(
    '/home/rehmemk/git/anugasgpp/Okushiri/data/boundary_wave_original.txt', skiprows=1)
t = data[:, 0]
y = data[:, 1]
energy = np.trapz(y ** 2, t)
original_wave = scipy.interpolate.interp1d(t, y, kind='zero', fill_value='extrapolate')


def bump(c):
    # define bumps [create input wave based on parameters]
    theta = c[0]
    position = c[1]
    weight = c[2]
    ybump = weight * np.exp(-.5 * (t - position) ** 2 * theta ** -2)
    return ybump


def createWave(par, normalization=1):
    nbump = len(par)
    residual = y.copy()
    c = np.zeros((nbump, 3))
    for k in range(nbump):
        maxid = np.argmax(np.abs(residual))
        c0 = np.array([1.5, t[maxid], residual[maxid]])

        def cost(c):
            ybump = bump(c)
            cost = np.sqrt(np.mean((ybump - residual) ** 2))
            return cost

        c[k, :] = fmin(cost, c0, disp=False)
        residual -= bump(c[k, :])

    # deform wave
    y_res = residual.copy()
    y_art = np.zeros(np.shape(residual))
    for k in range(nbump):
        y_res += par[k] * bump(c[k, :])
        y_art += par[k] * bump(c[k, :])
    if normalization == 1:
        energy_res = np.trapz(y_res ** 2, t)
        y_res = np.sqrt(energy / energy_res) * y_res
        energy_art = np.trapz(y_art ** 2, t)
        y_art = np.sqrt(energy / energy_art) * y_art

    # write data
    wave_function_residual = scipy.interpolate.interp1d(t, y_res, kind='zero', fill_value='extrapolate')
    wave_function_artificial = scipy.interpolate.interp1d(t, y_art, kind='zero', fill_value='extrapolate')
    return wave_function_residual, wave_function_artificial, c, residual


def upDownArrow(x, y, length=1.5):
    # x,y coordinates of the midpoint of the two headed arrow
    # up
    plt.arrow(x, y, 0, length/2., head_width=0.05, head_length=0.03,
              linewidth=4, color='k', length_includes_head=True)
    # down
    plt.arrow(x, y, 0, -length/2., head_width=0.05, head_length=0.03,
              linewidth=4, color='k')


def plotBumps(par, c, xlim, ylim):
    colors = ['C1', 'C6', 'C3', 'C4', 'C2', 'C5', 'C7', 'C8', 'C9']
    for k in range(len(par)):
        plt.plot(t, par[k]*bump(c[k, :])*400, label='bump {}'.format(k+1), color=colors[k])
    #plt.title('Gauss Bumps')
    plt.ylabel('Heigt in m', fontsize=labelfontsize)
    plt.xlabel('time in s', fontsize=labelfontsize)
    # plt.gca().set_xlim(xlim)
    # plt.gca().set_ylim(ylim)

    # upDownArrow(6.5, -1.25)
    # upDownArrow(12.35, 6.75)
    # upDownArrow(20, -4.5)
    #upDownArrow(22.2, 1)

    plt.legend(fontsize=legendfontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=tickfontsize)


def plotResidual(residual, xlim, ylim):
    plt.plot(t, residual*400)
    # plt.title('Resiudal')
    plt.ylabel('Heigt in m', fontsize=labelfontsize)
    plt.xlabel('time in s', fontsize=labelfontsize)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.gca().tick_params(axis='both', which='major', labelsize=tickfontsize)


def plotWave(wave_function, xlim, ylim, label=None):
    points = np.linspace(-10, 30, 10000)
    evals = np.zeros(len(points))
    for i in range(len(evals)):
        evals[i] = wave_function(points[i])
    plt.plot(points, evals*400, label=label)
    #plt.title('Okushiri Input Wave')
    plt.ylabel('Heigt in m', fontsize=labelfontsize)
    plt.xlabel('time in s', fontsize=labelfontsize)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.gca().tick_params(axis='both', which='major', labelsize=tickfontsize)


def wave_difference(wave1, wave2):
    numPoints = 10000
    points = np.linspace(-10, 30, numPoints)
    l2_diff = 0
    for i in range(len(points)):
        l2_diff += (wave1(points[i])-wave2(points[i]))**2
    l2_diff = np.sqrt(l2_diff/numPoints)
    return l2_diff


if __name__ == '__main__':
    dim = 6
    # par = [1.0]*dim  # original wave (-residual)
    par = [1.5, 0.5, 1.5, 0.671875, 0.5, 0.5]   # optimal 31.06m wave for R64
    xlim = [0, 22.5]
    ylim = [-6, 8]
    style = 'one'  # 'presentation'/'one'
    saveFig = 1
    savePath = '/home/rehmemk/git/anugasgpp/Okushiri/plots/'

    wave_function_residual, wave_function_artificial, c, residual = createWave(par, normalization=1)

    print(f'diff original wave, artificial wave: {wave_difference(original_wave,wave_function_artificial)}')

    # all three plots in seperate figures
    if style == 'presentation':
        labelfontsize = 16
        legendfontsize = 14
        tickfontsize = 14
        # labelfontsize = 14
        # legendfontsize = 12
        # tickfontsize = 10
        plt.figure(figsize=(6, 4.9))
        plotWave(original_wave, xlim, ylim)
        if saveFig == 1:
            plt.savefig(os.path.join(savePath, 'originalWave.pdf'))
            # plt.savefig('/home/rehmemk/git/okushiripaper/gfx/OptimalWave.pdf')
        plt.figure()
        plotBumps(par, c, xlim, ylim)
        if saveFig == 1:
            plt.savefig(os.path.join(savePath, 'gaussBumps.pdf'))
        plt.figure()
        plotResidual(residual, xlim, ylim)
        if saveFig == 1:
            plt.savefig(os.path.join(savePath, 'residual.pdf'))
        if saveFig != 1:
            plt.show()

    if style == 'one':
        # all three plots in one figure
        labelfontsize = 22
        legendfontsize = 15
        tickfontsize = 16
        plt.figure(figsize=plt.figaspect(1./3.))
        plt.subplot(1, 3, 1)
        plotWave(original_wave, xlim, ylim)
        plt.title('Original Wave', fontsize=labelfontsize)
        plt.subplot(1, 3, 2)
        plotBumps(par, c, xlim, ylim)
        plt.title('Gaussian Bumps', fontsize=labelfontsize)
        plt.subplot(1, 3, 3)
        # plotResidual(residual, xlim, ylim)
        # plt.title('Residual',fontsize=labelfontsize)
        plotWave(wave_function_artificial, xlim, ylim)
        plt.title('Artificial Wave', fontsize=labelfontsize)
        if saveFig == 1:
            plt.tight_layout()
            plt.savefig(os.path.join(savePath, 'InputWave.pdf'))

        # optimum wave w.r.t. maximum run-up
        plt.figure()
        opt_par = [1.5, 0.5, 1.5, 0.671875, 0.5, 0.5]
        _, opt_wave, _, _ = createWave(opt_par)
        plotWave(opt_wave, xlim, ylim)
        if saveFig == 1:
            # plt.tight_layout()
            plt.savefig(os.path.join(savePath, 'OptWave.pdf'), bbox_inches='tight')
        else:
            plt.title('optimum wave')
            plt.show()
