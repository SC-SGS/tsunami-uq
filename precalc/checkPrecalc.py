import cPickle as pickle
import numpy as np
import sys

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from okushiri import run as runOkushiri  # nopep8

# load an evaluation from the precalculations and rerun
# it locally to verify the results are the same.

dim = 2
gridResolution = 16
mc = False
if mc is False:
    precalcFileName = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/precalcValues/sg_precalculations451T%iR.pkl' % (
        gridResolution)
with open(precalcFileName, 'rb') as fp:
    data = pickle.load(fp)

arbitrary_id = 18
arbitraryKey = data.keys()[arbitrary_id]
values = data[arbitraryKey]
par = [arbitraryKey[i] for i in range(dim)]
print('evaluating and comparing at')
print(par)

np.savetxt('/home/rehmemk/git/anugasgpp/Okushiri/data/x.txt', par)
np.savetxt('/home/rehmemk/git/anugasgpp/Okushiri/data/gridsize.txt',
           [gridResolution])
runOkushiri()
y = np.loadtxt('/home/rehmemk/git/anugasgpp/Okushiri/data/y.txt')

print('diff: %.15E' % (np.linalg.norm(values-y)))
