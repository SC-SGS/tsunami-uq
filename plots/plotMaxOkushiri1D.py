import sys
import pysgpp
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')      # nopep8
from sgppOkushiri import maxOkushiri1Out

if __name__ == "__main__":
    dim = 1
    numTimeSteps = 451
    gridResolution = 16
    normalization = 1
    residual = 0
    okushiriFunc = maxOkushiri1Out(dim,  gridResolution, normalization, residual)

    X = np.linspace(0.5, 1.5, 11)
    Y = np.zeros(len(X))
    for i, x in enumerate(X):
        Y[i] = okushiriFunc.eval([x])
    okushiriFunc.cleanUp()

    plt.plot(X, Y)
    plt.show()
