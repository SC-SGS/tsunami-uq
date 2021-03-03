import multiprocessing
import pickle
import logging
import sys

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
import okushiri
from sgppOkushiri import generateKey, getPrecalcFileName, \
                         loadPrecalcData, savePrecalcData  # nopep8


def worker(return_dict, par, numTimeSteps, gridResolution, normalization, residual, wave_type,
           minimum_allowed_height, index, numPoints):
    '''
    worker function for multiprocessing
    '''
    key = generateKey(par)
    return_dict[key] = okushiri.run(par, gridResolution, normalization, residual, wave_type, minimum_allowed_height)
    print(f'Calculated key={key}    [{index}/{numPoints}]')


def precalc_parallel(points, numTimeSteps, gridResolution, normalization, residual, wave_type, minimum_allowed_height):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for index, point in enumerate(points):
        proc = multiprocessing.Process(target=worker, args=(return_dict, point, numTimeSteps,
                                                            gridResolution, normalization, residual,
                                                            wave_type, minimum_allowed_height,
                                                            index, len(points)))
        jobs.append(proc)
        proc.start()
    print(f'set up list with {len(jobs)} jobs')

    for proc in jobs:
        proc.join()
    return return_dict


def calculate_missing_values(points, numTimeSteps, gridResolution, normalization, residual,
                             wave_type, minimum_allowed_height, maxNumPoints=0):
    precalcValuesFileName = getPrecalcFileName(numTimeSteps, gridResolution, normalization,
                                               residual, wave_type, minimum_allowed_height)
    precalculatedValues = loadPrecalcData(precalcValuesFileName)

    todoPoints = []
    for point in points:
        key = generateKey(point)
        if key not in precalculatedValues:
            todoPoints.append(point)

    if len(todoPoints) == 0:
        print('Nothing to do. All given points have already been precalculated')
    else:
        print(f'{len(todoPoints)} new points need to be evaluated')

    while len(todoPoints) > 0:
        totalNumPoints = len(todoPoints)
        if maxNumPoints > 0 and totalNumPoints > maxNumPoints:
            # limit the number of points which are processed at once to prevent out of memory and the like
            currentPoints = todoPoints[:maxNumPoints]
            todoPoints = todoPoints[maxNumPoints:]
        else:
            currentPoints = todoPoints
            todoPoints = []

        print(f"\ncalculating {len(currentPoints)} new evaluations")
        multiprocessing_dict = precalc_parallel(currentPoints, numTimeSteps, gridResolution, normalization,
                                                residual, wave_type, minimum_allowed_height)
        for key in multiprocessing_dict:
            precalculatedValues[key] = multiprocessing_dict[key]
        savePrecalcData(precalcValuesFileName, precalculatedValues)
        print(f"\ncalculated {len(currentPoints)} new Okushiri evaluations")
        print(f"now there are {len(precalculatedValues)} precalculated values for this setup")

        if maxNumPoints > 0 and totalNumPoints > maxNumPoints:
            print(
                f'Because maxNumPoints parameter was set, only {maxNumPoints} of {totalNumPoints} desired points were calcualted')
