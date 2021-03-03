#!/usr/bin/env python3.8

import numpy as np
import dakota.interfacing as di
import sys
#import pysgpp

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from sgppOkushiri import maxOkushiri1Out               # nopep8


def maxOkushiri1Out_forDakota(v):
    dim = 6
    okushiri_func = maxOkushiri1Out(dim, gridResolution=64, normalization=1, residual=0)
    dv = np.array(v)
    result = okushiri_func.eval(dv)
    okushiri_func.cleanUp()
    return [-result]


params, results = di.read_parameters_file()
num_params = params.num_variables
continuous_vars = [0]*num_params
for k in range(num_params):
    continuous_vars[k] = params[params.descriptors[k]]

evaluations = maxOkushiri1Out_forDakota(continuous_vars)

for i, r in enumerate(results.responses()):
    if r.asv.function:
        r.function = evaluations[i]

results.write()
