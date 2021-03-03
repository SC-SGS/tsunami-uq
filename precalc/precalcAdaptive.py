import pysgpp
import numpy as np
from argparse import ArgumentParser
from okushiri_precalc_distributed import calculate_missing_values
import sys

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from sgppOkushiri import generateKey, getPrecalcFileName, \
                         loadPrecalcData, savePrecalcData,\
                         maxOkushiri1Out, maxOkushiriShape1Out  # nopep8

from max_mean_Okushiri import objFuncSGpp as objFuncSGpp  # nopep8


def checkPrecalc(reSurf, precalculatedValues):
    todoPoints = []
    todoPointsDetermined = False
    grid = reSurf.getGrid()
    gridStorage = grid.getStorage()
    dim = gridStorage.getDimension()
    lb = reSurf.getLowerBounds()
    ub = reSurf.getUpperBounds()
    for n in range(grid.getSize()):
        point = gridStorage.getPoint(n)
        par = np.zeros(dim)
        for d in range(dim):
            par[d] = lb[d] + (ub[d]-lb[d])*point.getStandardCoordinate(d)
        key = generateKey(par)
        if key not in precalculatedValues:
            # if point_py not in todoPoints:
            todoPoints.append(par)
            todoPointsDetermined = True

    return todoPointsDetermined, todoPoints


if __name__ == "__main__":
    # precalc properties
    initialLevel = 1
    numRefine = 10
    maxPoints = 10000
    verbose = False
    gridType = 'nakBsplineBoundary'
    #gridType = 'modNakBspline'
    #gridType = 'nakBsplineExtended'
    #gridType = 'nakPBspline'

    # model set-up
    dim = 6
    gridResolution = 64
    normalization = 1
    residual = 0
    wave_type = 'bumps'
    degree = 3
    boundaryLevel = 1
    numTimeSteps = 451

    precalcValuesFileName = getPrecalcFileName(numTimeSteps, gridResolution, normalization,
                                               residual, wave_type)
    precalculatedValues = loadPrecalcData(precalcValuesFileName)

    #okushiriFunc = maxOkushiriShape1Out(gridResolution, normalization)
    okushiriFunc = maxOkushiri1Out(dim, gridResolution, normalization, residual)

    lb, ub = okushiriFunc.getDomain()
    objFunc = objFuncSGpp(okushiriFunc)

    reSurf = pysgpp.SplineResponseSurface(objFunc, pysgpp.DataVector(lb),
                                          pysgpp.DataVector(ub),
                                          pysgpp.Grid.stringToGridType(gridType),
                                          degree, boundaryLevel)

    reSurf.regular(initialLevel)
    todoPointsDetermined = False
    counter = 0

    while not todoPointsDetermined:
        previousSize = reSurf.getSize()
        if previousSize > maxPoints:
            print(f"nothing to calculate for a maximum of {maxPoints} grid points")
            break
        reSurf.nextSurplusAdaptiveGrid(numRefine, verbose)
        todoPointsDetermined, todoPoints = checkPrecalc(reSurf, precalculatedValues)
        if not todoPointsDetermined:
            counter = counter + 1
            print(f"refining ({counter}), grid size: {reSurf.getSize()}")
            reSurf.refineSurplusAdaptive(numRefine, verbose)

    print(f'Now calculating {len(todoPoints)} evaluations for a grid of {reSurf.getSize()} points')

    calculate_missing_values(todoPoints, numTimeSteps, gridResolution, normalization, residual,
                             wave_type)

    print(f'Calcualted values for an adaptive grid of {reSurf.getSize()} points')
