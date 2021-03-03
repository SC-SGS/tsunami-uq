import ipdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

dim = 2
numTimeSteps = 451
gridResolution = 16  # 128
precalcValuesFileName = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/precalcValues/sg_precalculations{}T{}R.pkl'.format(
    numTimeSteps, gridResolution)
with open(precalcValuesFileName, 'rb') as f:
    precalculatedValues = pickle.load(f, encoding='latin1')

print("number of precalculated values: {}".format(
    len(precalculatedValues.keys())))

for point in precalculatedValues:
    plt.scatter(point[0], point[1], color='b', marker='.')
plt.show()
