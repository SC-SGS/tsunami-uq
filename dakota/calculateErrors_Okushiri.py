import sys
import h5py
import numpy as np
import pysgpp
import pickle
import matplotlib.pyplot as plt

# MC_distribution only determines the MC error data.
# The underlying distribution and PCE is so far manually managed in createDakotaInputFile.py
# TODO Change this
dim = 6
MC_distribution = 'uniform'  # 'normal'/'uniform'
gridResolution = 64
levels = [1, 2, 3, 4]
numMCPoints = 10000
average_l2Errors = np.zeros(len(levels))
min_l2Errors = np.zeros(len(levels))
max_l2Errors = np.zeros(len(levels))
average_nrmses = np.zeros(len(levels))
n_points = np.zeros(len(levels))

means = np.zeros((len(levels), 451))
variances = np.zeros((len(levels), 451))
for l, level in enumerate(levels):
    # L2 Error
    if MC_distribution == 'uniform':
        evalPath = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evaluations_Okushiri_{dim}D_R{gridResolution}_level{level}.dat'
    elif MC_distribution == 'normal':
        evalPath = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evaluations_Okushiri_{dim}D_R{gridResolution}_level{level}_normal.dat'

    with open(evalPath, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    actual_numMCPoints = len(lines) - 1
    points = np.zeros((actual_numMCPoints, dim))
    evals = np.zeros((actual_numMCPoints, 451))
    pointwise_l2Errors = np.zeros(451)
    print('calculating error with {} precalculated evaluations'.format(actual_numMCPoints))
    for i in range(1, len(lines)):
        line = lines[i].split()
        points[i-1, :] = [float(e) for e in line[2:2+dim]]
        evals[i-1, :] = [float(e) for e in line[2+dim:]]

    if MC_distribution == 'uniform':
        mcDataPath = f'/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_bumps_{numMCPoints}_{dim}D451T{gridResolution}R_noresidual.pkl'
        h5Path = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/results_Okushiri_{dim}D_R{gridResolution}_level{level}.h5'
    elif MC_distribution == 'normal':
        mcDataPath = f'/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_bumps_{numMCPoints}_{dim}D451T{gridResolution}R_normal_noresidual.pkl'
        h5Path = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/results_Okushiri_{dim}D_R{gridResolution}_level{level}_normal.h5'
    realEvals = np.zeros((numMCPoints, 451))
    with open(mcDataPath, 'rb') as fp:
        realData = pickle.load(fp)
    # for i, point in enumerate(points):
    #     realEvals[i, :] = realData[tuple(point)][0]
    # TODO: The above didn't work because they key/point was different in the ~10th decimal
    # As far as I know in current python3.8 dict orders are deterministic, so the following should work
    for i, key in enumerate(realData):
        realEvals[i, :] = realData[key][0]
        for t in range(451):
            pointwise_l2Errors[t] += (realEvals[i, t]-evals[i, t])**2
    pointwise_l2Errors = [np.sqrt(l2Error/numMCPoints) for l2Error in pointwise_l2Errors]
    min_l2Errors[l] = np.min(pointwise_l2Errors)
    max_l2Errors[l] = np.max(pointwise_l2Errors)
    average_l2Errors[l] = np.average(pointwise_l2Errors)

    # General Info, num grid points, moments (mean, var)
    with h5py.File(h5Path, 'r') as f:
        for t in range(1, 452):
            means[l, t-1] = f['methods/Polynomial Chaos/results/execution:1/expansion_moments/'
                              f'response_fn_{t}'][(0)]*400
            variances[l, t-1] = f['methods/Polynomial Chaos/results/execution:1/expansion_moments'
                                  f'/response_fn_{t}'][(1)]*400
        n_points[l] = len(f['/models/simulation/NO_MODEL_ID/variables/continuous'])

#print(means[3, :])
np.savetxt(
    f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/means/dakota_means_{gridResolution}_{MC_distribution}.txt', means)
# plt.plot(range(451), means[3, :])
# plt.show()

print(f'average l2: {average_l2Errors}')
print(f'min     l2: {min_l2Errors}')
print(f'max     l2: {max_l2Errors}')
plt.plot(n_points, average_l2Errors)
plt.gca().set_yscale('log')
plt.xlabel('# grid points')
plt.ylabel = ('average L2 error')
# plt.show()
