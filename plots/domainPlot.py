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
from netCDF4 import Dataset
import numpy as np
from scipy.optimize import fmin
import sys

sys.path.append('/home/rehmemk/git/anuga-clinic-2018/anuga_tools')  # nopep8
from animate import Domain_plotter
from animate import SWW_plotter

import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ['Times']})

# presentation
legendfontsize = 14
colorbarfontsize = 14
tickfontsize = 14

# par   input-wave parameters
# n     discretiazion size


def runOkushiri(par, n):
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
    f1 = interp1d(x, y, kind='quadratic')
    newpoints[:, 0] = f1(points[:, 0])

    # make refinement in y direction
    x = np.multiply([0., .125, .3, .7, .9, 1.], max(points[:, 1]))
    y = [0., 1.25, 1.75, 2.15, 2.65, max(points[:, 1])]
    f2 = interp1d(x, y, kind='quadratic')
    newpoints[:, 1] = f2(points[:, 1])

    c = abs(newpoints[:, 0] - 5.0) + .5 * abs(newpoints[:, 1] - 1.95)
    c = 0.125 * c

    points[:, 0] = c * points[:, 0] + (1 - c) * newpoints[:, 0]
    points[:, 1] = c * points[:, 1] + (1 - c) * newpoints[:, 1]

    # create domain
    domain = anuga.Domain(points, vertices, boundary)

    # don't store .sww file
    # domain.set_quantities_to_be_stored(None)

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
    domain.set_minimum_storable_height(0.001)  # Don't store w < 0.001m
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

    # load wave data
    # shutil.copyfile('boundary_wave_header.txt', 'boundary_wave_input.txt')
    data = np.loadtxt(
        '/home/rehmemk/git/anugasgpp/Okushiri/data/boundary_wave_original.txt', skiprows=1)
    t = data[:, 0]
    y = data[:, 1]
    energy = np.trapz(y ** 2, t)

    # define bumps [create input wave based on parameters]
    def bump(c):
        theta = c[0]
        position = c[1]
        weight = c[2]
        ybump = weight * np.exp(-.5 * (t - position) ** 2 * theta ** -2)
        return ybump

    nbump = len(par)
    residual = y.copy()
    c = np.zeros((nbump, 3))
    for k in range(nbump):
        maxid = np.argmax(np.abs(residual))
        c0 = np.array([1.5, t[maxid], residual[maxid]])

        def cost(c):
            ybump = bump(c)
            cost = np.sqrt(np.mean((ybump - residual) ** 2))
            return cost

        c[k, :] = fmin(cost, c0, disp=False)
        residual -= bump(c[k, :])

    # deform wave
    ynew = residual.copy()
    for k in range(nbump):
        ynew += par[k] * bump(c[k, :])
    energynew = np.trapz(ynew ** 2, t)
    ynew = np.sqrt(energy / energynew) * ynew

    # write data
    data[:, 1] = ynew.copy()
    import scipy
    wave_function = scipy.interpolate.interp1d(
        t, ynew, kind='zero', fill_value='extrapolate')

    # MR: uncomment to plot input wave
    # points = np.linspace(-10, 30, 10000)
    # evals = np.zeros(len(points))
    # for i in range(len(evals)):
    #     evals[i] = wave_function(points[i])
    # plt.figure()
    # # plt.plot(points, evals)
    # # plt.plot(t, residual, 'r')
    # for k in range(nbump):
    #     plt.plot(t, par[k]*bump(c[k, :]), label='bum {}'.format(k))
    # plt.title('Okushiri Input Wave')
    # plt.show()

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

    # area for gulleys
    x1 = 4.85
    x2 = 5.25
    y1 = 2.05
    y2 = 1.85

    # gauges
    gauges = [[4.521, 1.196], [4.521, 1.696], [4.521, 2.196]]

    # index in gulley area
    x = domain.centroid_coordinates[:, 0]
    y = domain.centroid_coordinates[:, 1]
    v = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) + \
        np.sqrt((x - x2) ** 2 + (y - y2) ** 2) < 0.5

    dplotter = Domain_plotter(domain, min_depth=0.001)

    k = 0
    # original number of timesteps is 451
    numTimeSteps = int(np.loadtxt(
        '/home/rehmemk/git/anugasgpp/Okushiri/data/numTimeSteps.txt'))
    meanstage = np.nan * np.ones((1, numTimeSteps))
    yieldstep = 0.05
    finaltime = (numTimeSteps - 1)*yieldstep
    meanlayer = 0

    # Do the actual calculation
    # for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
    #     # domain.write_time()

    #     # stage [=height of water]
    #     stage = domain.quantities['stage'].centroid_values[v]
    #     # averaging for smoothness
    #     meanstage[0, k] = np.mean(stage)
    #     # k is time
    #     k += 1

    # # PLOTTING

    # # Make movie of each timestep
    #     dplotter.save_depth_frame()
    # anim = dplotter.make_depth_animation()
    # anim.save('okushiri_%i.mp4' % n)
    # meanlayer = meanstage - meanstage[0, 0]

    # Plot the domain
    plt.figure()
    xya = np.loadtxt(
        '/home/rehmemk/git/anugasgpp/Okushiri/plots/Benchmark_2_Bathymetry.xya', skiprows=1, delimiter=',')
    X = xya[:, 0].reshape(393, 244)
    Y = xya[:, 1].reshape(393, 244)
    Z = xya[:, 2].reshape(393, 244)
    # Remove the white part of the seismic
    # Steves original code uses cmap('gist_earth')
    from matplotlib.colors import LinearSegmentedColormap
    interval = np.hstack([np.linspace(0.0, 0.3), np.linspace(0.5, 1.0)])
    colors = plt.cm.seismic(interval)
    my_cmap = LinearSegmentedColormap.from_list('name', colors)
    # Multiply heights by 400 so that we getreal scale, not model scale
    N1, N2 = np.shape(Z)
    for n1 in range(N1):
        for n2 in range(N2):
            Z[n1, n2] *= 400
    plt.contourf(X, Y, Z, 20, cmap=my_cmap)
    # plt.title('Bathymetry')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=colorbarfontsize)
    # cbar.set_label('elevation', rotation=270)
    import matplotlib.patches
    from matplotlib.patches import Ellipse
    # plt.plot(x1, y1, 'o')
    # plt.plot(x2, y2, 'o')
    ellipse = Ellipse(((x2+x1)/2., (y1+y2)/2.), width=0.5,
                      height=0.2, angle=-20, edgecolor='k', fill=False, label='area of interest', linewidth=4)
    plt.gca().add_patch(ellipse)
    # plt.plot(gauges[0][0], gauges[0][1], 'ok')
    # plt.plot(gauges[1][0], gauges[1][1], 'ok')
    # plt.plot(gauges[2][0], gauges[2][1], 'ok', markersize=8, label='gauge')

    # plt.axis('off')
    plt.legend(loc='upper left', fontsize=legendfontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=tickfontsize)
    plt.tight_layout()
    # ---------------- hack to get ellpse shaped ellipse in legend------------
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerPatch
    colors = ["k"]
    texts = ["area of interest"]

    class HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = mpatches.Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]
    c = [mpatches.Circle((0.5, 0.5), 1, facecolor='None', edgecolor='k', linewidth=3) for i in range(len(texts))]
    plt.legend(c, texts, bbox_to_anchor=(0., 1.), loc='upper left', ncol=1, fontsize=16, handler_map={
               mpatches.Circle: HandlerEllipse()})
    # ----------------------------
    plt.savefig('okushiri_domain.pdf')

    # Plot the triangle mesh
    plt.figure()
    mittelblau = (0./255, 81./255, 158./255)
    plt.triplot(dplotter.triang, linewidth=0.3, color=mittelblau)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('okushiri_mesh_%i.pdf' % n)
    # Plot the domain and the triangle mesh
    plt.figure()
    plt.tripcolor(dplotter.triang,
                  facecolors=dplotter.elev,
                  edgecolors='k',
                  cmap='gist_earth')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('okushiri_domainandmesh_%i.pdf' % n)

    # make video from sww file
    # swwplotter = SWW_plotter('output_okushiri.sww', min_depth=0.001)
    # lilo = len(swwplotter.time)
    # for k in range(lilo):
    #     if k % 10 == 0:
    #         print ' '
    #     swwplotter.save_stage_frame(frame=k, vmin=-0.02, vmax=0.1)
    #     print '(', swwplotter.time[k], k, ')',
    # print ' '
    # swwanim = swwplotter.make_stage_animation()
    # swwanim.save('okushiri_fromswwfile.mp4')

    return meanlayer

####################
####### MAIN #######
####################


n = 64  # 16
par = [0.6, 0.7]
y = runOkushiri(par, n)
# plt.show()
