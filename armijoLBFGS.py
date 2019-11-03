"""
L-BFGS algorithms using Armijo line search.
"""
import numpy as np
from utility import linesearch, parameters, limitedMemory


def solve(f, Df, x):
    """Monotone L-BFGS algorithm with Armijo line search."""
    lmData = limitedMemory.LmData(x.shape[0], parameters.memory)
    return linesearch.genericMonotone(lmData, updateLmData, inverseBFGS, armijo, f, Df, x)


def solveNonmonotone(f, Df, x):
    """Nonmonotone L-BFGS algorithm with Armijo line search."""
    lmData = limitedMemory.LmData(x.shape[0], parameters.memory)
    return linesearch.genericNonmonotone(lmData, updateLmData, inverseBFGS, armijo, f, Df, x)


def updateLmData(lmData, sn, yn):
    """Update L-BFGS data using cautious updating scheme."""
    gamma = lmData.gamma
    sty = np.dot(sn, yn)
    if (sty >= parameters.crvThreshold * np.dot(sn, sn)):
        # Compute new gamma
        gamma = np.dot(yn, yn) / sty
        # Perform update
        lmData.update(sn, yn, gamma)


def inverseBFGS(data, g):
    """Compute L-BFGS step."""
    return limitedMemory.twoLoopRecursion(data.S, data.Y, data.sty, data.mUpd, data.gamma, -g)


def armijo(x, f, Df, d, fx, gx):
    """Armijo line search."""
    it, t, dtgx, xn = 1, 1.0, np.dot(d,gx), x + d
    fn = f(xn)
    while fn > fx + 1e-4 * t *dtgx and t >= parameters.minStep:
        t *= 0.5
        it += 1
        xn = x + t * d
        fn = f(xn)
    
    if t < parameters.minStep:
        # Unsuccessful
        return x, False, fx, gx, it, np.zeros_like(d), np.zeros_like(gx)
    else:
        # Successful
        gn = Df(xn)
        return xn, True, fn, gn, it, t * d, gn - gx
