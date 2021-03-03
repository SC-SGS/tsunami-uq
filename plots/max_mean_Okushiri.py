import ipdb
import time
import pysgpp
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
from matplotlib import gridspec
from scipy.stats import truncnorm
import os

sys.path.append('/home/rehmemk/git/anugasgpp/Okushiri')  # nopep8
from sgppOkushiri import okushiri  # nopep8
from sgppOkushiri import maxOkushiri1Out, maxOkushiri1OutUnitCube  # nopep8

import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ['Times']})


linewidth = 2
labelfontsize = 28
legendfontsize = 18
tickfontsize = 20


class objFuncSGpp(pysgpp.ScalarFunction):

    def __init__(self, objFunc):
        self.dim = objFunc.getDim()
        self.objFunc = objFunc
        super(objFuncSGpp, self).__init__(self.dim)

    def eval(self, v):
        res = self.objFunc.eval(v)
        return res

    def getName(self):
        return self.objFunc.getName()

    def getDim(self):
        return self.dim

    def getDistributions(self):
        return self.objFunc.getDistributions()

    def getMean(self):
        return self.objFunc.getMean()

    def getVar(self):
        return self.objFunc.getVar()

    def cleanUp(self):
        self.objFunc.cleanUp()


class objFuncSGppSign(pysgpp.ScalarFunction):
    # wraps the negative of the objective function for SGpp. Needed for optimization (Max -> Min)

    def __init__(self, objFunc):
        self.dim = objFunc.getDim()
        self.objFunc = objFunc
        super(objFuncSGppSign, self).__init__(self.dim)

    def eval(self, v):
        res = self.objFunc.eval(v)
        return -res

    def getName(self):
        return self.objFunc.getName()

    def getDim(self):
        return self.dim

    def getDistributions(self):
        return self.objFunc.getDistributions()

    def getMean(self):
        return self.objFunc.getMean()

    def getVar(self):
        return self.objFunc.getVar()

    def cleanUp(self):
        self.objFunc.cleanUp()


# unnormalizes the value x in [lN

class vectorObjFuncSGpp(pysgpp.VectorFunction):
    # wraps the objective function for SGpp
    # NOTE: If we want to optimize we have to introduce an
    #       objFuncSGppSigned

    # input dimension dim
    # output dimension out
    def __init__(self, objFunc):
        self.dim = objFunc.getDim()
        self.out = objFunc.getOut()
        self.objFunc = objFunc
        super().__init__(self.dim, self.out)

    def eval(self, x, value):
        result = self.objFunc.eval(x)
        for t in range(self.out):
            value.set(t, result[t])

    def evalJacobian(self, x):
        jacobian = self.objFunc.evalJacobian(x)
        return jacobian

    def getName(self):
        return self.objFunc.getName()

    def getDim(self):
        return self.dim

    def getOut(self):
        return self.out

    def getLowerBounds(self):
        lb, _ = self.objFunc.getDomain()
        return lb

    def getUpperBounds(self):
        _, ub = self.objFunc.getDomain()
        return ub

    def getDistributions(self):
        return self.objFunc.getDistributions()

    def getMean(self):
        return self.objFunc.getMean()

    def getVar(self):
        return self.objFunc.getVar()

    def cleanUp(self):
        try:
            self.objFunc.cleanUp()
        except:
            warnings.warn('could not clean up')

    def getPrecalcEvals(self):
        return self.objFunc.getPrecalcEvals()


def getPercentiles(reSurf, percentages, numSamples, distribution='normal'):
    # create a large set of samples
    dim = reSurf.getNumDim()
    numTimeSteps = reSurf.getNumRes()
    if distribution == 'normal':
        mean = 1.0
        sd = 0.125
        lower = 0.5
        upper = 1.5
        rng = truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
        sampleSet = rng.rvs((numSamples, dim))
    elif distribution == 'uniform':
        lb = 0.5
        ub = 1.5
        unitpoints = np.random.rand(numSamples, dim)
        sampleSet = np.zeros((numSamples, dim))
        for i, point in enumerate(unitpoints):
            for d in range(dim):
                sampleSet[i, d] = lb + (ub-lb)*point[d]

    point = pysgpp.DataVector(dim)
    results = np.zeros((numSamples, numTimeSteps))
    for i in range(numSamples):
        for d in range(dim):
            point[d] = sampleSet[i, d]
        res = reSurf.eval(point)
        for n in range(numTimeSteps):
            results[i, n] = res[n]
    print(f'max of all {numSamples} percentile samples: {np.max(results)*400:.4f}m')
    # calculate the percentiles from the set of samples
    # (Also Monte Carlo based mean to see if Stochastic Collocation
    # gives a significant increase in accuracy over simple MC.)
    percentiles = np.zeros((len(percentages), numTimeSteps))
    mcMeans = np.zeros(numTimeSteps)
    for n in range(numTimeSteps):
        mcMeans[n] = np.mean(results[:, n])
        for p in range(len(percentages)):
            percentiles[p, n] = np.percentile(results[:, n], percentages[p])
    return percentiles, mcMeans

########################################################################
################################ Main ##################################
########################################################################


################# Parameters #################
dim = 6
numTimeSteps = 451
gridResolution = 64  # 128
gridType = 'nakBsplineBoundary'  # nakPBspline / 'nakBsplineExtended' /'nakBsplineBoundary'
degree = 3
refineType = 'surplus'  # 'surplus'  / 'regular'
distribution = 'normal'  # 'uniform' / 'normal'   must be set in sgppOkushiri.py too!
maxLevel = 1  # 6
maxPoints = 2500  # 2500
initialLevel = 1
numRefine = 10
normalization = 1
residual = 0

use_nakbspllineboundary_2500_opt = 1
calcVar = 0
plotVar = 0

# percentiles:
percentages = [5, 95]  # [10, 90]
numSamples = 10000  # 100000

saveFig = 1
legendstyle = 'internal'    # internal / external
saveMean = 0

################# Initialization and loading #################
pyFunc = okushiri(dim, numTimeSteps, gridResolution, normalization, residual, 'bumps', distribution)
objFunc = vectorObjFuncSGpp(pyFunc)
lb = objFunc.getLowerBounds()
ub = objFunc.getUpperBounds()

# reference means
if distribution == 'uniform':
    reference_means = np.loadtxt(
        '/home/rehmemk/git/anugasgpp/Okushiri/data/referenceMean_quadOrder3Okushiri64_uniform_nakBsplineBoundary3_6D_uniform_level3_5_95.txt')
elif distribution == 'normal':
    reference_means = np.loadtxt(
        '/home/rehmemk/git/anugasgpp/Okushiri/data/referenceMean_quadOrder100Okushiri64_nakBsplineExtended3_6D_normal_level6_5_95.txt')

### Dakota ###
dakota_means = np.loadtxt(f'/home/rehmemk/git/anugasgpp/Okushiri/data/dakota_means_64_{distribution}.txt')
for l in [1, 2, 3, 4]:
    print(f'Dakota average mean error level {l}: {np.average(np.abs(reference_means-dakota_means[l-1,:]))}'
          f'    worst diff {np.max(np.abs(reference_means-dakota_means[l-1,:]))}')
# testwise optimized in Dakota, siehe Mail 4.8.20
dakota_opt = [1.2635259, 0.5000014275, 1.499987603, 1.350807189, 0.5, 0.5]
maxTimeline_dakota = pyFunc.eval(dakota_opt)
maxTimeline_dakota = [maxTimeline_dakota[t]*400 for t in range(numTimeSteps)]
print(f'Dakotas opt values result in {np.max(maxTimeline_dakota):.5f}m')
pyFunc.cleanUp()

################# Calculations #################

# create surrogate
reSurf = pysgpp.SplineResponseSurfaceVector(objFunc, lb, ub, pysgpp.Grid.stringToGridType(gridType),
                                            degree)

if refineType == 'regular':
    reSurf.regular(maxLevel)
elif refineType == 'surplus':
    verbose = True
    reSurf.surplusAdaptive(maxPoints, initialLevel, numRefine, verbose)
else:
    warnings.warn("refineType not supported")
print("created response surface with {} grid points\n".format(reSurf.getSize()))

# try:
# errorVec, _ = averageL2FromData(reSurf, objFunc)
# print("average data-based L2 {:.5E}   (min {:.5E} max {:.5E})".format(errorVec[0], errorVec[1], errorVec[2]))
# except:
#     warnings.warn('Could not calculate error of response surface')

# calculate percentiles
start = time.time()
percentiles, mcMeans = getPercentiles(reSurf, percentages, numSamples, distribution)
print(f"calculating percentiles from {numSamples} samples took {time.time()-start}s")


# Optimization based on Ritter Novak. Not significantly better than the surplus grid.
# It would be hard to argument in the paper, that we create another grid for optimization.
# So we simply use the same grid, that has already been created surplus adaptive.
# gamma = 0.95  # 0.85
# initialLevel = 1
# verbose = True
# ritterNovakPointsMax = 500  # 500
# optFunc = maxOkushiri1OutUnitCube(dim, lb.array(), ub.array(), numTimeSteps, gridResolution)
# maxFunc = objFuncSGppSign(optFunc)
# unitlb = pysgpp.DataVector(dim, 0.0)
# unitub = pysgpp.DataVector(dim, 1.0)
# reSurfRN = pysgpp.SplineResponseSurface(maxFunc, unitlb, unitub, pysgpp.Grid.stringToGridType(gridType), degree)
# reSurfRN.ritterNovak(ritterNovakPointsMax, gamma, initialLevel, verbose)
# objFunc.cleanUp()
# print('created Ritter Novak grid with {} grid points'.format(reSurfRN.getSize()))
# max_par_RN = reSurfRN.optimize()
# maxTimelineRN = -maxFunc.eval(max_par_RN)
# # remap opt from unit cube to actual parameter space
# max_par_RN = [lb[d] + (ub[d]-lb[d])*max_par_RN[d] for d in range(dim)]
# print(f'Ritter Novak: max parameters are {max_par_RN}, resulting in {np.max(maxTimelineRN)*400:.5f}m')
# maxFunc.cleanUp()
# sys.exit()

# Optimization based on same adaptive grid
optFunc = maxOkushiri1Out(dim, gridResolution, normalization, residual)
maxFunc = objFuncSGppSign(optFunc)
if use_nakbspllineboundary_2500_opt:
    max_par = [1.5, 0.5, 1.5, 0.671875, 0.5, 0.5]
else:
    grid = reSurf.getGrid()
    optReSurf = pysgpp.SplineResponseSurface(maxFunc, grid, lb, ub, degree)
    max_par = optReSurf.optimize().array()
maxTimeline = pyFunc.eval(max_par)
print(f'max parameters {max_par} result in {np.max(maxTimeline)*400:.5f}m')
pyFunc.cleanUp()


# Calculate means
pdfs = pyFunc.getDistributions()
start = time.time()
if distribution == 'uniform':
    quadOrder = degree
elif distribution == 'normal':
    # quadOrder 100 is accurate up to 12 decimals (in comparison to quadOrder 500)
    quadOrder = 100  # 100
print(f'\nNow calculating means with quadOrder {quadOrder}')
means = reSurf.getMeans(pdfs, quadOrder).array()
print(f"calculating the {numTimeSteps} means took {time.time()-start}")

# print(f'mean normalizing factor: {means[-1]:.16f}')  # this is for meanNormalizingFactor
# sys.exit()

# The distribution of Okushiri is Normal(1,0.125) truncted to [0.5,1.5].
# As with Dakota  by truncating the integral of
# the normal distribution is no longer 1 and the results are off by a factor.
# By calculating the mean of f(x)=1 following this distribution, I get this
#  factor  and can divide my results by this to get correct means.
if distribution == 'normal':
    print('Adapting mean result to fit Normal(1,0.125) truncated to [0.5,1.5]')
    if dim == 6:
        # calcualted with nakBsplineExtended deg 3 level 6 quadOrder 100
        meanNormalizingFactor = 0.9996200052747830
elif distribution == 'uniform':
    meanNormalizingFactor = 1.0
else:
    print('NO MEAN NORMALIZING FACTOR AVAILABLE. USING (WRONG!) DEFAULT 1')
    meanNormalizingFactor = 1.0
means = [m / meanNormalizingFactor for m in means]

### Multiply everything by 400 so that we get real scale, not model scale ###
for p in range(len(percentiles)):
    for t in range(numTimeSteps):
        percentiles[p, t] *= 400
maxTimeline = [maxTimeline[t]*400 for t in range(numTimeSteps)]
means = [means[t]*400 for t in range(numTimeSteps)]
mcMeans *= 400

print(
    f'average mean error: {np.average(np.abs(means-reference_means))}     worst diff {np.max(np.abs(means-reference_means))}')

# Calculate VARIANCES
if calcVar:
    dummyMeanSquares = pysgpp.DataVector(1)
    dummyMeans = pysgpp.DataVector(1)
    start = time.time()
    variances = reSurf.getVariances(pdfs, quadOrder, dummyMeans, dummyMeanSquares)
    variances_py = [variances[t] for t in range(numTimeSteps)]
    variances_py = [v*400 for v in variances_py]
    print(f'Calculating the variances took {time.time()-start}s')
    np.savetxt('/home/rehmemk/git/anugasgpp/Okushiri/plots/variances.txt', variances_py)
else:
    if plotVar:
        print('LOADING VARIANCES FROM FILE. NO GUARANTEES FOR THEM MATCHING THE CURRENT CALCULATIONS!')
        variances_py = np.loadtxt('/home/rehmemk/git/anugasgpp/Okushiri/plots/variances.txt')


### Plotting ###
time = [t*22.5/450 for t in range(451)]
# theoretically plot the same data on both axes
# I actually plot only one because everything is constant anyways on the left


def do_plot(plot_mean, plot_opt, axis_off=0):
    fig = plt.figure(figsize=(8, 6))
    originalFunc = okushiri(dim, numTimeSteps, gridResolution, normalization=1, residual=1,
                            wave_type='original')
    originalRunup = originalFunc.eval([1]*dim)*400
    originalFunc.cleanUp()

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    if plot_mean == 1:
        if plotVar:
            # SD Plots
            # TODO Ich muss hier abs nehmen, das dürfte eigentlich nicht passieren. Var>0!
            lower_sd = means - 0.5 * np.sqrt(np.abs(variances_py))
            upper_sd = means + 0.5 * np.sqrt(np.abs(variances_py))
            ax2.fill_between(time, lower_sd, upper_sd, label='sd', color='r', linewidth=linewidth, alpha=0.5)
            #ax2.plot(time, upper_sd, label='sd', color='r', linewidth=linewidth)
        ax2.plot(time, means, 'C1', label='mean', linewidth=linewidth)
        #ax2.plot(time, dakota_means[-1, :], 'C8', label='Dakota mean', linewidth=linewidth)

    ax2.plot(time, percentiles[0, :], 'C0', label=f"{percentages[0]}th-{percentages[1]}th percentile")
    ax2.plot(time, percentiles[1, :], 'C0')
    ax2.fill_between(time, percentiles[0, :], percentiles[1, :], color='C0', alpha=0.4)

    # ax2.plot(time, mcMeans, 'C4-', label='mc mean', linewidth=linewidth)
    if plot_opt == 1:
        ax2.plot(time, maxTimeline, 'C3', label='max runup', linewidth=linewidth)
        # ax2.plot(time, maxTimeline_dakota, 'C4-', label='Dakota max runup', linewidth=linewidth)

    ax.plot(time, originalRunup, '-', color='grey', linewidth=linewidth)
    ax2.plot(time, originalRunup, '-', color='grey', label='original run-up', linewidth=linewidth)

    ax.set_xlim(0, 1)
    ax2.set_xlim(14, 22.5)
    ax.set_ylim(24, 31.5)
    ax2.set_ylim(24, 31.5)

    # https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    ax.tick_params(axis='both', which='major', labelsize=tickfontsize)
    ax2.tick_params(axis='both', which='major', labelsize=tickfontsize)

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.
    if axis_off == 0:
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-d, 1+d), (-d, +d), **kwargs)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'

    ax.set_ylabel('Height in m', fontsize=labelfontsize)
    ax2.set_xlabel('time in s', fontsize=labelfontsize)
    if axis_off == 0 and legendstyle != 'external':
        ax2.legend(loc='upper right', fontsize=legendfontsize)
    if axis_off == 1:
        ax.axis('off')
        ax2.axis('off')
    plt.tight_layout()

    return fig


if refineType == 'regular':
    name = f'Okushiri{gridResolution}_{distribution}_{gridType}{degree}_{dim}D_{distribution}_level{maxLevel}_{percentages[0]}_{percentages[1]}'
else:
    name = f'Okushiri{gridResolution}_{distribution}_{gridType}{degree}_{dim}D_{distribution}_{reSurf.getSize()}pts_{percentages[0]}_{percentages[1]}'

if saveMean:
    np.savetxt(os.path.join('/home/rehmemk/git/anugasgpp/Okushiri/data/',
                            'referenceMean_'+f'quadOrder{quadOrder}'+name+'.txt'), means)

fig_perc_max_mean = do_plot(1, 1, 0)

figname = f'/home/rehmemk/git/anugasgpp/Okushiri/plots/perc_max_mean'+name
if saveFig == 1:
    # plt.savefig(figname)
    if legendstyle == 'external':
        plt.savefig(figname+'.pdf', dpi=1000, bbox_inches='tight', format='pdf')
        print('saved fig to {}'.format(figname+'.pdf'))
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()

        plt.figure()
        axe = plt.gca()
        ncol = 4
        axe.legend(handles, labels, loc='center', fontsize=legendfontsize, ncol=ncol)
        axe.xaxis.set_visible(False)
        axe.yaxis.set_visible(False)
        for v in axe.spines.values():
            v.set_visible(False)
        legendname = os.path.join(figname + '_legend'+'.pdf')
        # cut off whitespace
        plt.subplots_adjust(left=0.0, right=1.0, top=0.6, bottom=0.4)
        plt.savefig(legendname + '.pdf', dpi=1000,
                    bbox_inches='tight', pad_inches=0.0, format='pdf')
    else:
        plt.legend(fontsize=legendfontsize)
        plt.savefig(figname+'incl_legend.pdf', dpi=1000, bbox_inches='tight', format='pdf')


objFunc.cleanUp()
maxFunc.cleanUp()

if saveFig == 0:
    plt.show()
