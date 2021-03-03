import numpy as np
import pickle

# To evaluate the Dakota PCE Surrogate the according points must be specified in an input file
# This function takes an MC point file from my standard Okushiri calculations and rewrites it
# to such a Dakota file

dim = 6
numPoints = 10000
distribution = 'uniform'  # 'uniform'/'normal'
gridResolution = 64
if distribution == 'uniform':
    mcFile = f'/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_bumps_{numPoints}_{dim}D451T{gridResolution}R_noresidual.pkl'
    dakotaInputFile = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evalPoints{dim}D_R{gridResolution}_{numPoints}.dat'
elif distribution == 'normal':
    mcFile = f'/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_bumps_{numPoints}_{dim}D451T{gridResolution}R_normal_noresidual.pkl'
    dakotaInputFile = f'/home/rehmemk/git/anugasgpp/Okushiri/dakota/data/evalPoints{dim}D_R{gridResolution}_{numPoints}_normal.dat'


with open(mcFile, 'rb') as fp:
    data = pickle.load(fp)

with open(dakotaInputFile, 'w') as fp:
    if dim == 3:
        fp.write('% eval_id interface            x1              x2            x3\n')
    elif dim == 5:
        fp.write(
            '% eval_id interface            x1              x2            x3           x4              x5\n')
    elif dim == 6:
        fp.write('% eval_id interface            x1              x2            x3           x4              x5            x6\n')
    elif dim == 8:
        fp.write('% eval_id interface            x1              x2            x3           x4              x5            x6              x7            x8\n')
    for i, key in enumerate(data):
        fp.write(f'{i}        NO_ID    ')
        for k in key:
            fp.write(f'{k}  ')
        fp.write('\n')
