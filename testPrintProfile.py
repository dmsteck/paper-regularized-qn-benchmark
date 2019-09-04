"""
...
"""
import numpy as np
from utility import parameters
from utility import perfprof


resultsFile = np.load('results.npz')
fx, normDfx, iter, nf = [resultsFile[var]
                         for var in ['fx', 'normDfx', 'iter', 'nf']]

nfValid = nf.astype(float)
nfValid[normDfx > parameters.tolGrad] = np.inf

linespecs = ['r-', 'b:', 'c--', 'g-.', 'y-']
legend = ['armijoLBFGS', 'wolfeLBFGS', 'regLBFGS', 'regLSR1', 'regLPSB']

# Monotone algorithms
perfprof.perfprof(nfValid[0].T, linespecs=linespecs, legendnames=legend, legendpos=4, thmax=5.)

# Nonmonotone algorithms
perfprof.perfprof(nfValid[1].T, linespecs=linespecs, legendnames=legend, legendpos=4, thmax=5.)
