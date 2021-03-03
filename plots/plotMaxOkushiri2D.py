import sys
import pysgpp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')      # nopep8
from sgppOkushiri import maxOkushiri1Out                           # nopep8
sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri/precalc')  # nopep8
from okushiri_precalc_distributed import calculate_missing_values  # nopep8

if __name__ == "__main__":
    dim = 2
    gridResolution = 16
    normalization = 1
    residual = 0
    okushiriFunc = maxOkushiri1Out(dim,  gridResolution, normalization, residual)

    numPoints = 51
    X, Y = np.meshgrid(np.linspace(0.5, 1.5, numPoints), np.linspace(0.5, 1.5, numPoints))
    Z = np.zeros((numPoints, numPoints))
    points = []
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            points.append([X[i, j], Y[i, j]])

    numTimeSteps = 451
    wave_type = 'bumps'
    calculate_missing_values(points, numTimeSteps, gridResolution, normalization, residual,
                             wave_type)

    # reload okushiriFunc with the new precalculated values
    okushiriFunc = maxOkushiri1Out(dim,  gridResolution, normalization, residual)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[i, j] = okushiriFunc.eval([X[i, j], Y[i, j]])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    plt.show()
