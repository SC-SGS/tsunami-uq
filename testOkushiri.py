from sgppOkushiri import okushiriStorage
import matplotlib.pyplot as plt
import numpy as np
import time
# This is for verifying the parallely computed resutls
# from raijin are equal to my locally serially computed resutls

dim = 4
numTimeSteps = 451
gridResolution = 128
storage = okushiriStorage(dim, numTimeSteps, gridResolution)
qoi = 'all'
par = [0., 0., 2., 2.]
y, y_g5, y_g7, y_g9, y_bc = storage.eval(par, qoi, usePrecalc=True)
plt.plot(range(len(y)), y, '-C0', label='y precalc')
plt.plot(range(len(y_g5)), y_g5, '-C1', label='y_g5 precalc')
plt.plot(range(len(y_g7)), y_g7, '-C2', label='y_g7 precalc')
plt.plot(range(len(y_g9)), y_g9, '-C3', label='y_g9 precalc')
plt.plot(range(len(y_bc)), y_bc, '-C4', label='y_bc precalc')

start = time.time()
calc_y, calc_y_g5, calc_y_g7, calc_y_g9, calc_y_bc = storage.eval(
    par, qoi, usePrecalc=False)
print(f"calculation took {time.time()-start}s")

print(f"diff y: {np.linalg.norm(calc_y - y)}")
print(f"diff y_g5: {np.linalg.norm(calc_y_g5 - y_g5)}")
print(f"diff y_g7: {np.linalg.norm(calc_y_g7 - y_g7)}")
print(f"diff y_g9: {np.linalg.norm(calc_y_g9 - y_g9)}")
print(f"diff y_bc: {np.linalg.norm(calc_y_bc - y_bc)}")

plt.plot(range(len(calc_y)), calc_y, '+C0', label='y new')
plt.plot(range(len(calc_y_g5)), calc_y_g5, '+C1', label='y_g5 new')
plt.plot(range(len(calc_y_g7)), calc_y_g7, '+C2', label='y_g7 new')
plt.plot(range(len(calc_y_g9)), calc_y_g9, '+C3', label='y_g9 new')
plt.plot(range(len(calc_y_bc)), calc_y_bc, '+C4', label='y_bc new')


storage.cleanUp()
plt.legend()
plt.show()
