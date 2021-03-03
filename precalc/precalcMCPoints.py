import os
import sys
import numpy as np
import pickle
from okushiri_precalc_distributed import precalc_parallel
import sys

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from sgppOkushiri import okushiri, getMCFileName  # nopep8


numMCPoints = 10000
dim = 6
gridResolution = 64
normalization = 1
residual = 0
wave_type = 'bumps'  # 'bumps'/'shape'
distribution = 'uniform'  # 'uniform'/'normal'
minimum_allowed_height = 1e-5
maxNumPoints = 300

numTimeSteps = 451
okushiriFunc = okushiri(dim, numTimeSteps, gridResolution, normalization,
                        residual, wave_type, distribution, minimum_allowed_height)

lb, ub = okushiriFunc.getDomain()

todoPoints = []
if distribution == 'uniform':
    unitpoints = np.random.rand(numMCPoints, dim)
    for point in unitpoints:
        for d in range(dim):
            point[d] = lb[d] + (ub[d]-lb[d])*point[d]
        todoPoints.append(point)
elif distribution == 'normal':
    mu, sigma = 1.0, 0.125  # mean and standard deviation
    for m in range(numMCPoints):
        todoPoints.append(np.random.normal(mu, sigma, dim).tolist())

regular_dict = {}
while len(todoPoints) > 0:
    totalNumPoints = len(todoPoints)
    if maxNumPoints > 0 and totalNumPoints > maxNumPoints:
        # limit the number of points which are processed at once to prevent out of memory and the like
        currentPoints = todoPoints[:maxNumPoints]
        todoPoints = todoPoints[maxNumPoints:]
    else:
        currentPoints = todoPoints
        todoPoints = []
    multiprocessing_dict = precalc_parallel(currentPoints, numTimeSteps, gridResolution, normalization,
                                            residual, wave_type, minimum_allowed_height)
    for key in multiprocessing_dict:
        regular_dict[key] = multiprocessing_dict[key]

mcPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues'
filename = getMCFileName(numMCPoints, dim, numTimeSteps, gridResolution, wave_type, normalization,
                         residual, distribution, minimum_allowed_height)
with open(os.path.join(mcPath, filename), 'wb+') as fp:
    pickle.dump(regular_dict, fp)

print(f'calculated data for {numMCPoints} random points, saved as {filename}')
