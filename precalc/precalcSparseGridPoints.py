import pysgpp
import numpy as np
from argparse import ArgumentParser
from okushiri_precalc_distributed import calculate_missing_values
import sys

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from sgppOkushiri import okushiri  # nopep8

if __name__ == "__main__":
    parser = ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--level', default=3, type=int)
    parser.add_argument('--dim', default=6, type=int)
    parser.add_argument('--gridResolution', default=64, type=int)
    parser.add_argument('--normalization', default=1, type=int)
    parser.add_argument('--residual', default=0, type=int)
    parser.add_argument('--maxNumPoints', default=200, type=int)
    args = parser.parse_args()
    level = args.level
    dim = args.dim
    gridResolution = args.gridResolution
    normalization = args.normalization
    residual = args.residual
    wave_type = 'bumps'
    # wave_type = 'shape'
    #wave_type = 'cubic'
    minimum_allowed_height = 1e-5
    distribution = 'normal'

    numTimeSteps = 451
    okushiriFunc = okushiri(dim, numTimeSteps, gridResolution, normalization,
                            residual, wave_type, distribution, minimum_allowed_height)

    lb, ub = okushiriFunc.getDomain()

    degree = 3
    boundaryLevel = 1

    grid = pysgpp.Grid_createNakBsplineBoundaryGrid(dim, degree, boundaryLevel)
    #grid = pysgpp.Grid_createNakBsplineExtendedGrid(dim, degree)
    grid.getGenerator().regular(level)
    gridStorage = grid.getStorage()

    points = []
    for i in range(grid.getSize()):
        point = gridStorage.getPointCoordinates(i).array()
        for d in range(dim):
            point[d] = lb[d] + (ub[d]-lb[d]) * point[d]
        points.append(point)
    print(f'Grid of dim {dim}, level {level} has {grid.getSize()} points')

    calculate_missing_values(points, numTimeSteps, gridResolution, normalization, residual,
                             wave_type, minimum_allowed_height, args.maxNumPoints)
