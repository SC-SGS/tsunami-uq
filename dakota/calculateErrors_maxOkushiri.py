import sys
import h5py
import numpy as np
import pysgpp
import pickle
import matplotlib.pyplot as plt

dim = 6
numMCPoints = 100
gridResolution = 16
levels = [1, 2, 3]
l2Errors = np.zeros(len(levels))
nrmses = np.zeros(len(levels))
n_points = np.zeros(len(levels))

for l, level in enumerate(levels):
    # L2 Error
    evalPath = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evaluations_maxOkushiri_{dim}D_level{level}.dat'
    points = []
    evals = []
    with open(evalPath, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    points = np.zeros((len(lines)-1, dim))
    evals = np.zeros(len(lines)-1)
    print('calculating error with {} precalculated evaluations'.format(len(points)))
    for i in range(1, len(lines)):
        line = lines[i].split()
        points[i-1, :] = ([float(e) for e in line[2:-1]])
        evals[i-1] = (float(line[-1]))

    l2Error = 0
    realEvals = np.zeros(len(points))
    mcDataPath = f'/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_bumps_{numMCPoints}_{dim}D451T{gridResolution}R_noresidual.pkl'
    with open(mcDataPath, 'rb') as fp:
        realData = pickle.load(fp)
    # for i, point in enumerate(points):
    #     realEvals[i] = np.max(realData[tuple(point)])
    # TODO: The above didn't work because they key/point was different in the ~10th decimal
    # As far as I know in current python3.8 dict orders are deterministic, so the following should work
    for i, key in enumerate(realData):
        print(np.shape(realData[key][0]))
        realEvals[i] = np.max(realData[key][0])
        l2Error += (realEvals[i]-evals[i])**2
    l2Error = np.sqrt(l2Error/len(points))
    nrmse = l2Error/(np.max(realEvals) - np.min(realEvals))
    l2Errors[l] = l2Error
    nrmses[l] = nrmse

    # General Info, num grid points, moments (mean, var)
    h5Path = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/results_maxOkushiri_level{level}.h5'
    with h5py.File(h5Path, 'r') as f:
        mean = f['methods/Polynomial Chaos/results/execution:1/expansion_moments/response_fn_1'][(
            0)]
        var = f['methods/Polynomial Chaos/results/execution:1/expansion_moments/response_fn_1'][(
            1)]
        n_points[l] = len(f['/models/simulation/NO_MODEL_ID/variables/continuous'])


print(f'l2: {l2Errors}')
print(f'NRMSE: {nrmses}')
plt.plot(n_points, l2Errors)
plt.gca().set_yscale('log')
plt.xlabel('# grid points')
plt.ylabel = ('L2 error')
plt.show()
