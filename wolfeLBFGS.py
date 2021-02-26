"""
Monotone and nonmonotone L-BFGS algorithms with Wolfe line search.
"""
import numpy as np
from utility import linesearch, morethuente, parameters, limitedMemory


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
    fun = lambda x: (f(x), Df(x))

    xn,fn,gn,t,exls,it = \
        morethuente.cvsrch(fun,len(x),x,fx,gx,d,1,1e-4,0.9,1e-16,1e-20,1e20,20)
    
    if exls != 1:
        # Unsucessful
        return x, False, fx, gx, it, np.zeros_like(d), np.zeros_like(gx)
    else:
        # Successful
        return xn, True, fn, gn, it, t * d, gn - gx
