import ipdb
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from sgppOkushiri import okushiri  # nopep8

dim = 6
gridResolution = 64
numTimeSteps = 451
normalization = 1
residual = 0
distribution = 'normal'
wave_type = 'original'

pyFunc = okushiri(dim, numTimeSteps, gridResolution, normalization, residual, wave_type, distribution)

time = [t/22.5 for t in range(451)]
runup = pyFunc.eval([1]*dim)
pyFunc.cleanUp()
plt.plot(time, runup*400, label='average runup')

tickfontsize = 14
labelfontsize = 14
legendfontsize = 14
plt.xlabel('time in s', fontsize=labelfontsize)
plt.ylabel('height in m', fontsize=labelfontsize)
plt.gca().tick_params(axis='both', which='major', labelsize=tickfontsize)
# plt.legend(fontsize=legendfontsize)
# plt.show()
plt.savefig('/home/rehmemk/git/anugasgpp/Okushiri/plots/OriginalRunup.pdf')
