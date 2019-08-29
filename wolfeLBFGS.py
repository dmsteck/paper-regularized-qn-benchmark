"""
Monotone and nonmonotone L-BFGS algorithms with Wolfe line search.
"""
import numpy as np
import scipy.optimize.linesearch
from utility import linesearch, parameters, limitedMemory


def solve(f, Df, x):
    """Monotone L-BFGS algorithm with Wolfe line search."""
    lmData = limitedMemory.LmData(x.shape[0], parameters.memory)
    return linesearch.genericMonotone(lmData, updateLmData, inverseBFGS, wolfe, f, Df, x)


def solveNonmonotone(f, Df, x):
    """Nonmonotone L-BFGS algorithm with Wolfe line search."""
    lmData = limitedMemory.LmData(x.shape[0], parameters.memory)
    return linesearch.genericNonmonotone(lmData, updateLmData, inverseBFGS, wolfe, f, Df, x)


def updateLmData(lmData, sn, yn):
    """Update limited memory data for Wolfe-type L-BFGS algorithms."""
    # Compute new gamma
    gamma = np.dot(yn, yn) / np.dot(sn, yn)
    # Perform update
    lmData.update(sn, yn, gamma)


def inverseBFGS(data, g):
    """Compute L-BFGS search direction."""
    return limitedMemory.twoLoopRecursion(data.S, data.Y, data.sty, data.mUpd, data.gamma, -g)


def wolfe(x, f, Df, d, fx, gx):
    """Perform Wolfe line-search by calling DCSRCH from MINPACK."""
    t, it, _, fn, _, gn = scipy.optimize.linesearch.line_search_wolfe1(
        f, Df, x, d, gx, fx, c1=1e-4, c2=0.5, amin=parameters.minStep)
    if t is None:
        # Unsucessful
        return x, False, fx, gx, it, np.zeros_like(d), np.zeros_like(gx)
    else:
        # Successful
        return x + t * d, True, fn, gn, it, t * d, gn - gx
