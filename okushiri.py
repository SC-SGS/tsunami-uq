# ------------------------------------------------------------------------------
# Import necessary modules
# ------------------------------------------------------------------------------
import anuga
import numpy as np
import shutil
from anuga.utilities import plot_utils as util
from anuga.config import netcdf_float
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from netCDF4 import Dataset
import numpy as np
from scipy.optimize import fmin
import wave_functions


'''
The serial Okushiri simulation
'''

# par   input-wave parameters
# n     discretiazion size


def run(par, n, withNormalization, withResidual, wave_type='bumps', minimum_allowed_height=1e-5):
    # ------------------------------------------------------------------------------
    # Setup computational domain
    # ------------------------------------------------------------------------------
    xleft = 0
    xright = 5.448
    ybottom = 0
    ytop = 3.402

    # rectangular cross mesh
    points, vertices, boundary = anuga.rectangular_cross(int(n), int(n),
                                                         xright - xleft, ytop - ybottom,
                                                         (xleft, ybottom))

    newpoints = points.copy()

    # make refinement in x direction
    x = np.multiply([0., 0.1, 0.2, 0.335, 0.925, 1.], max(points[:, 0]))
    y = [0., 3., 4.25, 4.7, 5.3, max(points[:, 0])]
    f1 = interp1d(x, y, kind='linear')
    newpoints[:, 0] = f1(points[:, 0])

    # make refinement in y direction
    x = np.multiply([0., .125, .3, .7, .9, 1.], max(points[:, 1]))
    y = [0., 1.25, 1.75, 2.15, 2.65, max(points[:, 1])]
    f2 = interp1d(x, y, kind='linear')
    newpoints[:, 1] = f2(points[:, 1])

    c = abs(newpoints[:, 0] - 5.0) + .5 * abs(newpoints[:, 1] - 1.95)
    c = 0.125 * c

    points[:, 0] = c * points[:, 0] + (1 - c) * newpoints[:, 0]
    points[:, 1] = c * points[:, 1] + (1 - c) * newpoints[:, 1]

    # create domain
    domain = anuga.Domain(points, vertices, boundary)

    # don't store .sww file
    domain.set_quantities_to_be_stored(None)

    # ------------------------------------------------------------------------------
    # Initial Conditions
    # ------------------------------------------------------------------------------
    domain.set_quantity('friction', 0.01)  # 0.0
    domain.set_quantity('stage', 0.0)
    domain.set_quantity('elevation',
                        filename='/home/rehmemk/git/anugasgpp/Okushiri/data/bathymetry.pts',
                        alpha=0.02)

    # ------------------------------------------------------------------------------
    # Set simulation parameters
    # ------------------------------------------------------------------------------
    domain.set_name('output_okushiri')  # Output name
    # domain.set_minimum_storable_height(0.001)  # Don't store w < 0.001m
    domain.set_minimum_storable_height(1.0)  # Don't store w < 0.001m
    domain.set_flow_algorithm('DE0')

    # ------------------------------------------------------------------------------
    # Modify input wave
    # ------------------------------------------------------------------------------
    # rescale input parameter
    try:
        dummy = len(par)
    except:
        par = [par]
    par = np.dot(2, par)

    if wave_type == 'bumps':
        wave_function, _, _ = wave_functions.heights_wave(par, withResidual, withNormalization)
    elif wave_type == 'original':
        wave_function = wave_functions.original_wave_interpolant()
    elif wave_type == 'cubic':
        wave_function, _, _ = wave_functions.cubic_heights_wave(par, withResidual, withNormalization)
    else:
        print(f'Error. Wave type {wave_type} unknown')

    # ------------------------------------------------------------------------------
    # Setup boundary conditions
    # ------------------------------------------------------------------------------

    # Create boundary function from input wave [replaced by wave function]

    # Create and assign boundary objects
    Bts = anuga.Transmissive_momentum_set_stage_boundary(domain, wave_function)
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': Bts, 'right': Br, 'top': Br, 'bottom': Br})

    # ------------------------------------------------------------------------------
    # Evolve system through time
    # ------------------------------------------------------------------------------

    # this prevents problems w.r.t. divisions by zero
    # It might decrease the acheivable accuracy
    domain.set_minimum_allowed_height(minimum_allowed_height)  # default 1e-5

    # area for gulleys
    x1 = 4.85
    x2 = 5.25
    y1 = 2.05
    y2 = 1.85

    # index in gulley area
    x = domain.centroid_coordinates[:, 0]
    y = domain.centroid_coordinates[:, 1]
    v = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) + \
        np.sqrt((x - x2) ** 2 + (y - y2) ** 2) < 0.5

    # three gauges and a point somewhere on the boundary that could be used for verification
    # get id's of the corresponding triangles
    gauge = [[4.521, 1.196], [4.521, 1.696], [4.521, 2.196]]
    bdyloc = [0.00001, 2.5]
    g5_id = domain.get_triangle_containing_point(gauge[0])
    g7_id = domain.get_triangle_containing_point(gauge[1])
    g9_id = domain.get_triangle_containing_point(gauge[2])
    bc_id = domain.get_triangle_containing_point(bdyloc)

    k = 0
    # original number of timesteps is 451
    numTimeSteps = 451
    sumstage = np.nan * np.ones(numTimeSteps)
    stage_g5 = np.nan * np.ones(numTimeSteps)
    stage_g7 = np.nan * np.ones(numTimeSteps)
    stage_g9 = np.nan * np.ones(numTimeSteps)
    stage_bc = np.nan * np.ones(numTimeSteps)
    yieldstep = 0.05
    finaltime = (numTimeSteps - 1)*yieldstep
    for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
        # domain.write_time()

        # stage [=height of water]
        stage = domain.quantities['stage'].centroid_values[v]
        stage_g5[k] = domain.quantities['stage'].centroid_values[g5_id]
        stage_g7[k] = domain.quantities['stage'].centroid_values[g7_id]
        stage_g9[k] = domain.quantities['stage'].centroid_values[g9_id]
        stage_bc[k] = domain.quantities['stage'].centroid_values[bc_id]
        # averaging for smoothness
        sumstage[k] = np.sum(stage)
        # k is time
        k += 1

    # number of triangles which are active for the designated runup area
    numActiveTriangles = anuga.collect_value(np.count_nonzero(v))
    averageStage = sumstage / numActiveTriangles
    # normalizing to zero level
    # averageStage -= averageStage[0]

    return [averageStage, stage_g5, stage_g7, stage_g9, stage_bc]
