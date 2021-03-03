import pysgpp
import numpy as np
from argparse import ArgumentParser
from okushiri_precalc_distributed import calculate_missing_values
import sys

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from sgppOkushiri import generateKey, getPrecalcFileName, \
                         loadPrecalcData, savePrecalcData,\
                         okushiri  # nopep8

from max_mean_Okushiri import vectorObjFuncSGpp  # nopep8


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


def getNextTodoPoints(maxPoints, precalculatedValues, dim, gridType, degree, numTimeSteps,
                      gridResolution, normalization, residual, wave_type, distribution,
                      minimum_allowed_height):
    okushiriFunc = okushiri(dim, numTimeSteps, gridResolution, normalization, residual, wave_type,
                            distribution, minimum_allowed_height)
    lb, ub = okushiriFunc.getDomain()
    objFunc = vectorObjFuncSGpp(okushiriFunc)
    pdfs = objFunc.getDistributions()
    reSurf = pysgpp.SplineResponseSurfaceVector(objFunc, pysgpp.DataVector(lb),
                                                pysgpp.DataVector(ub),
                                                pysgpp.Grid.stringToGridType(gridType),
                                                degree)
    reSurf.regular(initialLevel)
    todoPointsDetermined = False
    counter = 0

    while not todoPointsDetermined:
        previousSize = reSurf.getSize()
        if previousSize > maxPoints:
            print(f"nothing to calculate for a maximum of {maxPoints} grid points")
            return todoPointsDetermined, [], reSurf.getSize()

        reSurf.nextSurplusAdaptiveGrid(numRefine, verbose)
        #reSurf.nextDistributionAdaptiveGrid(numRefine, pdfs, verbose)
        todoPointsDetermined, todoPoints = checkPrecalc(reSurf, precalculatedValues)
        if not todoPointsDetermined:
            counter = counter + 1
            print(f"refining ({counter}), grid size: {reSurf.getSize()}")
            reSurf.refineSurplusAdaptive(numRefine, verbose)
            #reSurf.refineDistributionAdaptive(numRefine, pdfs, verbose)
    return todoPointsDetermined, todoPoints, reSurf.getSize()


if __name__ == "__main__":
    parser = ArgumentParser(description='Get a program and run it with input')
    # 'nakBsplineBoundary' / 'modNakBspline' / 'nakBsplineExtended' / 'nakPBspline'
    parser.add_argument('--gridType', default='nakBsplineExtended', type=str, help='grid and basis type')  # nopep8
    args = parser.parse_args()
    gridType = args.gridType

    print(f'Now precalculating for {gridType}')

    # precalc properties
    initialLevel = 1
    numRefine = 10  # 10
    maxPoints = 2500  # 2500
    verbose = False

    # model set-up
    dim = 6
    gridResolution = 64
    normalization = 1
    residual = 0
    wave_type = 'bumps'
    distribution = 'normal'
    degree = 3
    numTimeSteps = 451
    minimum_allowed_height = 1e-5

    precalcValuesFileName = getPrecalcFileName(numTimeSteps, gridResolution, normalization,
                                               residual, wave_type, minimum_allowed_height)

    gridSize = 0
    totalCalculations = 0
    while gridSize < maxPoints:
        precalculatedValues = loadPrecalcData(precalcValuesFileName)
        todoPointsDetermined, todoPoints, gridSize = getNextTodoPoints(maxPoints, precalculatedValues, dim,
                                                                       gridType, degree, numTimeSteps,
                                                                       gridResolution, normalization,
                                                                       residual, wave_type, distribution,
                                                                       minimum_allowed_height)
        if not todoPointsDetermined:
            print('No calcualtion necessary.')
            break
        else:
            print(f'Now calculating {len(todoPoints)} evaluations for a grid of {gridSize} points')

            calculate_missing_values(todoPoints, numTimeSteps, gridResolution, normalization, residual,
                                     wave_type, minimum_allowed_height)

            print(f'Calcualted values for an adaptive grid of {gridSize} points')
            totalCalculations += len(todoPoints)
        print(f'\nIn total calcualted {totalCalculations} new evaluations for an adaptive grid of size {gridSize}')
