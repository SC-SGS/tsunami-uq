import warnings
import numpy as np
import pysgpp
import warnings
import pickle
import os
import sys
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from okushiri import run as okushiri_run


def generateKey(par):
    # lists are not allowed as keys, but tuples are
    key = tuple(par)
    return key


def getPrecalcFileName(numTimeSteps, gridResolution, normalization, residual, wave_type, minimum_allowed_height):
    precalcValuesPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/precalcValues/'
    precalcValuesFileName = os.path.join(
        precalcValuesPath, f'sg_precalculations_{wave_type}_{numTimeSteps}T{gridResolution}R')
    if normalization == 0:
        precalcValuesFileName += '_notnormalized'
    if residual == 0:
        precalcValuesFileName += '_noresidual'
    if minimum_allowed_height == 1e-10:
        precalcValuesFileName += 'minheight1e-10'
    precalcValuesFileName += '.pkl'
    return precalcValuesFileName


def getMCFileName(numMCPoints, dim, numTimeSteps, gridResolution, wave_type='bumps',
                  normalization=1, residual=0, distribution='uniform', minimum_allowed_height=1e-5):
    filename = f'mc_{wave_type}_{numMCPoints}_{dim}D{numTimeSteps}T{gridResolution}R'
    if distribution == 'normal':
        filename += '_normal'
    if normalization == 0:
        filename += '_notnormalized'
    if residual == 0:
        filename += '_noresidual'
    if minimum_allowed_height == 1e-10:
        filename += '_minheight1e-10'
    filename += '.pkl'
    return filename


def loadPrecalcData(precalcValuesFileName):
    try:
        with open(precalcValuesFileName, 'rb') as f:
            precalculatedValues = pickle.load(f)  # , encoding='latin1')
        print(f'loaded precalculated evaluations from {precalcValuesFileName}')
    except (FileNotFoundError):
        print(f'could not find precalculated data at {precalcValuesFileName}\n Creating new data file.')
        precalculatedValues = {}
    return precalculatedValues


def savePrecalcData(precalcValuesFileName, precalculatedValues):
    with open(precalcValuesFileName, "wb") as f:
        pickle.dump(precalculatedValues, f)


class okushiriStorage():
    # hashes evaluations of the okushiri model, to speed up SG++ runtimes
    def __init__(self, dim, numTimeSteps, gridResolution, normalization=1, residual=1,
                 wave_type='bumps', minimum_allowed_height=1e-5):
        self.dim = dim
        self.numTimeSteps = numTimeSteps
        self.gridResolution = gridResolution
        self.normalization = normalization
        self.residual = residual
        self.wave_type = wave_type
        self.minimum_allowed_height = minimum_allowed_height
        self.precalcValuesFileName = getPrecalcFileName(numTimeSteps, gridResolution, normalization,
                                                        residual, wave_type, minimum_allowed_height)
        self.precalculatedValues = loadPrecalcData(self.precalcValuesFileName)
        self.numNew = 0

    def cleanUp(self):
        savePrecalcData(self.precalcValuesFileName, self.precalculatedValues)
        print(f"\ncalculated {self.numNew} new Okushiri evaluations")
        if self.numNew > 0:
            print(f"saved them to {self.precalcValuesFileName}")

    def eval(self, x, qoi='gulley', usePrecalc=True):
        try:
            # if sgpp DataVector
            parameters = x.array()
        except AttributeError:
            # if normal python list
            parameters = x

        # if self.wave_type == 'shape' and len(parameters) != 9:
        #     warnings.warn(f'wave type shape needs 9D input! Not {len(parameters)}')
        #     sys.exit()

        key = generateKey(parameters)
        if key in self.precalculatedValues and usePrecalc == True:
            # print("sgppOkushiri: found key {}".format(parameters))
            [y, y_g5, y_g7, y_g9, y_bc] = self.precalculatedValues[key]
        else:
            print(f"sgppOkushiri {self.wave_type}: processing {parameters}")
            # reset time, is this necessary?
            np.savetxt('/home/rehmemk/git/anugasgpp/Okushiri/data/t.txt', [-1])

            [y, y_g5, y_g7, y_g9, y_bc] = okushiri_run(parameters, self.gridResolution,
                                                       self.normalization, self.residual,
                                                       self.wave_type, self.minimum_allowed_height)

            self.precalculatedValues[key] = [y, y_g5, y_g7, y_g9, y_bc]
            self.numNew += 1
            print(f"sgppOkushiri: Done ({self.numNew} calculations)")
        if qoi == 'gulley':
            return y
        elif qoi == 'g5':
            return y_g5
        elif qoi == 'g7':
            return y_g7
        elif qoi == 'g9':
            return y_g9
        elif qoi == 'bc':
            return y_bc
        elif qoi == 'all':
            return y, y_g5, y_g7, y_g9, y_bc
        else:
            warnings.warn(f'qoi {qoi} does not exist')


class okushiri():
    # The Okushiri Benchmark
    def __init__(self, dim, numTimeSteps=451, gridResolution=16, normalization=1, residual=0,
                 wave_type='bumps', distribution='normal', minimum_allowed_height=1e-5):
        self.dim = dim
        self.out = numTimeSteps
        self.gridResolution = gridResolution  # 16/128
        self.normalization = normalization
        self.residual = residual
        self.wave_type = wave_type
        self.okushiriStorage = okushiriStorage(
            dim, numTimeSteps, self.gridResolution, self.normalization, self.residual, self.wave_type)
        self.pdfs = pysgpp.DistributionsVector()
        self.distribution = distribution
        self.minimum_allowed_height = minimum_allowed_height
        if wave_type == 'bumps':
            for d in range(self.dim):
                if self.distribution == 'uniform':
                    self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
                elif self.distribution == 'normal':
                    self.pdfs.push_back(pysgpp.DistributionTruncNormal(1.0, 0.125, 0.5, 1.5))
                    # self.pdfs.push_back(pysgpp.DistributionNormal(1.0, 0.125))
        # elif wave_type == 'shape':
        #     # TODO REPLACE THESE WITH NORMAL DISTRIBUTIONS WHEN CALCULATING STOCHASTIC MOMENTS
        #     self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
        #     self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
        #     self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
        #     self.pdfs.push_back(pysgpp.DistributionUniform(1.0, 2.0))
        #     self.pdfs.push_back(pysgpp.DistributionUniform(1.0, 2.0))
        #     self.pdfs.push_back(pysgpp.DistributionUniform(1.0, 2.0))
        elif wave_type == 'original':
            for _ in range(self.dim):
                self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        name = f"okushiri_{self.wave_type}_{self.dim}D{self.out}T{self.gridResolution}R_{self.distribution}"
        if self.normalization == 0:
            name += '_notnormalized'
        if self.minimum_allowed_height == 1e-10:
            name += '_minheight1e-10'
        return name

    def getDim(self):
        return self.dim

    def getOut(self):
        return self.out

    def eval(self, x):
        y = self.okushiriStorage.eval(x)
        return y
        # return np.ones(self.out)

    def cleanUp(self):
        self.okushiriStorage.cleanUp()

    def getPrecalcData(self):
        # !! THIS IS DEPRECATED !!
        #  The new error calcualtion routine uses 'getPrecalcEvals', which is basically the same as
        #  this function, but the dict with reference values is differently set up.
        #   TODO delete this routine when everything has been updatet
        #
        # path to precalculated data for mc error calculation
        #numMCPoints = 100
        #numMCPoints = 1000
        numMCPoints = 10000
        mcPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues'
        filename = getMCFileName(numMCPoints, self.dim, self.out, self.gridResolution,
                                 self.wave_type, self.normalization, self.residual, self.distribution)
        precalcPath = os.path.join(mcPath, filename)
        print(f'loading precalculated mc data from {precalcPath}')

        with open(precalcPath, 'rb') as fp:
            precalcData = pickle.load(fp)  # , encoding='latin1')
        precalcData_y = {}
        # TODO: This reads the stage from the (stage,g5,g7,g9,bc) dataset everytime
        for key in precalcData:
            precalcData_y[key] = precalcData[key][0]
        return precalcData_y

    def getPrecalcEvals(self):
        # rewrite the dict wirh reference values to new style
        precalcData_y = self.getPrecalcData()
        numPrecalc = len(precalcData_y)
        points = np.zeros((numPrecalc, self.getDim()))
        evaluations = np.zeros((numPrecalc, self.getOut()))
        for i, key in enumerate(precalcData_y):
            points[i, :] = key
            evaluations[i, :] = precalcData_y[key]
        precalcData_newFormat = {'points': points, 'evaluations': evaluations}
        return precalcData_newFormat

    def getDistributions(self):
        return self.pdfs


class okushiri_input_wave():
    # The Okushiri Benchmark
    def __init__(self, dim, normalization=1, residual=1):
        self.dim = dim
        self.normalization = normalization
        self.residual = residual
        self.out = 451
        self.pdfs = pysgpp.DistributionsVector()
        for d in range(dim):
            self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
        data = np.loadtxt(
            '/home/rehmemk/git/anugasgpp/Okushiri/data/boundary_wave_original.txt', skiprows=1)
        self.t = data[:, 0]
        self.y = data[:, 1]
        self.energy = np.trapz(self.y ** 2, self.t)

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        return "okushiri_input_wave{}D".format(self.getDim())

    def getDim(self):
        return self.dim

    def getOut(self):
        return self.out

    def bump(self, c):
        # define bumps [create input wave based on parameters]
        theta = c[0]
        position = c[1]
        weight = c[2]
        ybump = weight * np.exp(-.5 * (self.t - position) ** 2 * theta ** -2)
        return ybump

    def eval(self, par):
        nbump = len(par)
        residual = self.y.copy()
        c = np.zeros((nbump, 3))
        for k in range(nbump):
            maxid = np.argmax(np.abs(residual))
            c0 = np.array([1.5, self.t[maxid], residual[maxid]])

            def cost(c):
                ybump = self.bump(c)
                cost = np.sqrt(np.mean((ybump - residual) ** 2))
                return cost

            c[k, :] = fmin(cost, c0, disp=False)
            residual -= self.bump(c[k, :])

        # deform wave
        if self.residual == 1:
            ynew = residual.copy()
        else:
            ynew = np.zeros(len(residual))

        for k in range(nbump):
            ynew += par[k] * self.bump(c[k, :])

        if self.normalization == 1:
            energynew = np.trapz(ynew ** 2, self.t)
            ynew = np.sqrt(self.energy / energynew) * ynew
        else:
            pass

        # write data
        wave_function = interp1d(
            self.t, ynew, kind='zero', fill_value='extrapolate')

        wave = [wave_function(t).item() for t in self.t]
        return wave

    def getDistributions(self):
        return self.pdfs


class maxOkushiri1Out():
    # The Okushiri Benchmark reduced to its maximum runup
    # Important are the height of the runup and the time of its occurence
    def __init__(self, dim, gridResolution=16, normalization=1, residual=0):
        self.dim = dim
        self.gridResolution = gridResolution
        self.normalization = normalization
        self.residual = residual
        self.numTimeSteps = 451
        self.okushiriStorage = okushiriStorage(
            dim, self.numTimeSteps, self.gridResolution, self.normalization, self.residual)
        self.pdfs = pysgpp.DistributionsVector()
        for d in range(self.dim):
            # self.pdfs.push_back(pysgpp.DistributionUniform(0.0, 2.0))
            # self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
            self.pdfs.push_back(pysgpp.DistributionTruncNormal(1.0, 0.5, 0.5, 1.5))

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        name = "maxOkushiri1Out{}D{}R".format(self.getDim(), self.gridResolution)
        if self.normalization == 0:
            name += '_notnormalized'
        if self.residual == 0:
            name += '_noresidual'
        return name

    def getDim(self):
        return self.dim

    def getOut(self):
        return 1

    def eval(self, x):
        y = self.okushiriStorage.eval(x)
        maxRunUp = np.max(y)
        time = float(np.argmax(y))
        return maxRunUp

    def cleanUp(self):
        self.okushiriStorage.cleanUp()

    def getDistributions(self):
        return self.pdfs

    def getMean(self):
        warnings.warn("referrence mean not available")
        return 77

    def getVar(self):
        warnings.warn("reference var not available")
        return -1

    def getPrecalcData(self):
        # path to precalculated data for mc error calculation
        # numMCPoints = 100
        numMCPoints = 1000
        mcPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues'
        filename = getMCFileName(numMCPoints, self.dim, self.numTimeSteps, self.gridResolution)
        precalcPath = os.path.join(mcPath, filename)
        # if self.normalization == 0:
        #     filename += '_notnormalized'
        # if self.residual == 0:
        #     filename += '_noresidual'
        # precalcPath = filename + '.pkl'
        print(f'loading precalculated mc data from {precalcPath}')

        with open(precalcPath, 'rb') as fp:
            precalcData = pickle.load(fp)  # , encoding='latin1')
        precalcData_y = {}
        # TODO: This reads the stage from the (stage,g5,g7,g9,bc) dataset everytime
        for key in precalcData:
            precalcData_y[key] = np.max(precalcData[key][0])
        return precalcData_y


class maxOkushiriShape1Out():
    # The Okushiri Benchmark reduced to its maximum runup based on the 9D shape wave_type
    # Important are the height of the runup and the time of its occurence
    def __init__(self, gridResolution=16, normalization=1):
        self.dim = 6
        self.gridResolution = gridResolution
        self.normalization = normalization
        self.numTimeSteps = 451
        residual = 0
        self.wave_type = 'shape'
        self.okushiriStorage = okushiriStorage(self.dim, self.numTimeSteps, self.gridResolution,
                                               self.normalization, residual, self.wave_type)

        self.pdfs = pysgpp.DistributionsVector()
        # TODO REPLACE THESE WITH NORMAL DISTRIBUTIONS WHEN CALCULATING STOCHASTIC MOMENTS
        self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
        self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
        self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))
        self.pdfs.push_back(pysgpp.DistributionUniform(1.0, 2.0))
        self.pdfs.push_back(pysgpp.DistributionUniform(1.0, 2.0))
        self.pdfs.push_back(pysgpp.DistributionUniform(1.0, 2.0))
        # self.pdfs.push_back(pysgpp.DistributionUniform(-2.0, 2.0))
        # self.pdfs.push_back(pysgpp.DistributionUniform(-2.0, 2.0))
        # self.pdfs.push_back(pysgpp.DistributionUniform(-2.0, 2.0))

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        name = "maxOkushiriShape1Out{}D{}R".format(self.getDim(), self.gridResolution)
        if self.normalization == 0:
            name += '_notnormalized'
        return name

    def getDim(self):
        return self.dim

    def getOut(self):
        return 1

    def eval(self, x):
        y = self.okushiriStorage.eval(x)
        maxRunUp = np.max(y)
        time = float(np.argmax(y))
        return maxRunUp

    def cleanUp(self):
        self.okushiriStorage.cleanUp()

    def getDistributions(self):
        return self.pdfs

    def getMean(self):
        warnings.warn("referrence mean not available")
        return 77

    def getVar(self):
        warnings.warn("reference var not available")
        return -1

    def getPrecalcData(self):
        # path to precalculated data for mc error calculation
        numMCPoints = 100
        # numMCPoints = 1000
        mcPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues'
        filename = getMCFileName(numMCPoints, self.dim, self.numTimeSteps, self.gridResolution,
                                 self.wave_type)
        precalcPath = os.path.join(mcPath, filename)
        print(f'loading precalculated mc data from {precalcPath}')

        with open(precalcPath, 'rb') as fp:
            precalcData = pickle.load(fp)  # , encoding='latin1')
        precalcData_y = {}
        # TODO: This reads the stage from the (stage,g5,g7,g9,bc) dataset everytime
        for key in precalcData:
            precalcData_y[key] = np.max(precalcData[key][0])
        return precalcData_y


class maxOkushiri1OutUnitCube():
    # To allow the Ritter Novak Grid generation which needs a function
    # to be defined on [0,1]^D, this is maxOkushiri1Out with inputs in
    # [0,1]^D which are mapped to the real domain reallb, realub
    # The Okushiri Benchmark reduced to its maximum runup
    # Only the height of the runup is returned
    def __init__(self, dim, reallb, realub, numTimeSteps=451, gridResolution=16, normalization=1, residual=0):
        self.dim = dim
        self.reallb = reallb
        self.realub = realub
        self.gridResolution = gridResolution
        self.numTimeSteps = numTimeSteps
        self.normalization = normalization
        self.residual = residual
        self.okushiriStorage = okushiriStorage(
            dim, self.numTimeSteps, self.gridResolution, self.normalization, self.residual)
        self.pdfs = pysgpp.DistributionsVector()
        for _ in range(dim):
            self.pdfs.push_back(pysgpp.DistributionUniform(0.0, 1.0))

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        return "maxOkushiri1OutUnitCube{}D{}R".format(self.getDim(), self.gridResolution)

    def getDim(self):
        return self.dim

    def getOut(self):
        return 1

    def eval(self, x):
        # map x from [0,1]^D to real parameter space reallb, realub
        transX = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            transX[d] = self.reallb[d] + (self.realub[d]-self.reallb[d])*x[d]
        # print(f"{x.toString()}")
        # print(f"{transX.toString()}\n")

        y = self.okushiriStorage.eval(transX)
        maxRunUp = np.max(y)
        time = float(np.argmax(y))
        return maxRunUp

    def cleanUp(self):
        self.okushiriStorage.cleanUp()

    def getDistributions(self):
        return self.pdfs


class okushiri_g5():
    # The Okushiri Benchmark
    def __init__(self, dim, numTimeSteps=451, gridResolution=16):
        self.dim = dim
        self.out = numTimeSteps
        self.gridResolution = gridResolution
        self.okushiriStorage = okushiriStorage(
            dim, numTimeSteps, self.gridResolution)
        self.pdfs = pysgpp.DistributionsVector()
        for d in range(self.dim):
            self.pdfs.push_back(pysgpp.DistributionUniform(0.0, 2.0))
            # self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        return "okushiri_g5{}D{}T{}R".format(self.getDim(), self.getOut(), self.gridResolution)

    def getDim(self):
        return self.dim

    def getOut(self):
        return self.out

    def eval(self, x):
        y = self.okushiriStorage.eval(x, qoi='g5')
        return y

    def cleanUp(self):
        self.okushiriStorage.cleanUp()

    def getPrecalcData(self):
        # path to precalculated data for mc error calculation
        precalcPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_precalculations{}D{}T{}R.pkl'.format(
            self.getDim(), self.getOut(), self.gridResolution)
        with open(precalcPath, 'rb') as fp:
            precalcData = pickle.load(fp, encoding='latin1')
        precalcData_y = {}
        # TODO: This reads the stage from the (stage,g5,g7,g9,bc) dataset everytime
        for key in precalcData:
            precalcData_y[key] = precalcData[key][1]
        return precalcData_y

    def getDistributions(self):
        return self.pdfs


class okushiri_g7():
    # The Okushiri Benchmark
    def __init__(self, dim, numTimeSteps=451, gridResolution=16):
        self.dim = dim
        self.out = numTimeSteps
        self.gridResolution = gridResolution
        self.okushiriStorage = okushiriStorage(
            dim, numTimeSteps, self.gridResolution)
        self.pdfs = pysgpp.DistributionsVector()
        for d in range(self.dim):
            self.pdfs.push_back(pysgpp.DistributionUniform(0.0, 2.0))
            # self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        return "okushiri_g7{}D{}T{}R".format(self.getDim(), self.getOut(), self.gridResolution)

    def getDim(self):
        return self.dim

    def getOut(self):
        return self.out

    def eval(self, x):
        y = self.okushiriStorage.eval(x, qoi='g7')
        return y

    def cleanUp(self):
        self.okushiriStorage.cleanUp()

    def getPrecalcData(self):
        # path to precalculated data for mc error calculation
        precalcPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_precalculations{}D{}T{}R.pkl'.format(
            self.getDim(), self.getOut(), self.gridResolution)
        with open(precalcPath, 'rb') as fp:
            precalcData = pickle.load(fp, encoding='latin1')
        precalcData_y = {}
        # TODO: This reads the stage from the (stage,g5,g7,g9,bc) dataset everytime
        for key in precalcData:
            precalcData_y[key] = precalcData[key][2]
        return precalcData_y

    def getDistributions(self):
        return self.pdfs


class okushiri_g9():
    # The Okushiri Benchmark
    def __init__(self, dim, numTimeSteps=451, gridResolution=16, normalization=1):
        self.dim = dim
        self.out = numTimeSteps
        self.gridResolution = gridResolution
        self.normalization = normalization
        self.okushiriStorage = okushiriStorage(
            dim, numTimeSteps, self.gridResolution, self.normalization)
        self.pdfs = pysgpp.DistributionsVector()
        for d in range(self.dim):
            self.pdfs.push_back(pysgpp.DistributionUniform(0.0, 2.0))
            # self.pdfs.push_back(pysgpp.DistributionUniform(0.5, 1.5))

    def getDomain(self):
        lb = pysgpp.DataVector(self.dim)
        ub = pysgpp.DataVector(self.dim)
        for d in range(self.dim):
            bounds = self.pdfs.get(d).getBounds()
            lb[d] = bounds[0]
            ub[d] = bounds[1]
        return lb, ub

    def getName(self):
        if self.normalization == 1:
            return "okushiri_g9{}D{}T{}R".format(self.getDim(), self.getOut(), self.gridResolution)
        elif self.normalization == 0:
            return "okushiri_g9{}D{}T{}R_notnormalized".format(self.getDim(), self.getOut(), self.gridResolution)

    def getDim(self):
        return self.dim

    def getOut(self):
        return self.out

    def eval(self, x):
        y = self.okushiriStorage.eval(x, qoi='g9')
        return y

    def cleanUp(self):
        self.okushiriStorage.cleanUp()

    def getPrecalcData(self):
        # path to precalculated data for mc error calculation
        precalcPath = '/home/rehmemk/git/anugasgpp/Okushiri/precalc/mcValues/mc_precalculations{}D{}T{}R.pkl'.format(
            self.getDim(), self.getOut(), self.gridResolution)
        with open(precalcPath, 'rb') as fp:
            precalcData = pickle.load(fp, encoding='latin1')
        precalcData_y = {}
        # TODO: This reads the stage from the (stage,g5,g7,g9,bc) dataset everytime
        for key in precalcData:
            precalcData_y[key] = precalcData[key][3]
        return precalcData_y

    def getDistributions(self):
        return self.pdfs
