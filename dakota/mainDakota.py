
import subprocess
import sys
from createDakotaInputFile import getText

# 3D sparse grid points, 1:7    2:31,   3:111,  4:303   5:804
# 5D sparse grid points, 1:11,  2:71
# 6D uniform,   1:13    2:97    3:545   4:2465
# 6D normal,    1: 13   2:97    3:539   4:2453
#               (42 s)  (4 m)   (26 m)  (120 m)                     reine Dakota-Zeit, da alle Werte schon vorberechnet
# 8D normal,    1: 17   2:161   3:1113

# NOTE Currently getText asigns distributions. DISTRIBUTION MUST BE SET MANUALLY IN createDakotaInputfile.py

dim = 6
# NOTE createDakotaInputFile.py needs to be manually adapted for dimension and distribution
sparse_grid_level = 4
numMCPoints = 10000
#objective = 'maxOkushiri'
objective = 'Okushiri'
gridResolution = 64
MC_distribution = 'uniform'  # 'normal'/'uniform'

text = getText(sparse_grid_level, objective, dim, numMCPoints, gridResolution, MC_distribution)
with open('/home/rehmemk/git/anugasgpp/Okushiri/dakota/dakota_pce_okushiri.in', 'w+') as fp:
    fp.write(text)

process = subprocess.run('dakota -i dakota_pce_okushiri.in',
                         shell=True,
                         cwd='/home/rehmemk/git/anugasgpp/Okushiri/dakota')
