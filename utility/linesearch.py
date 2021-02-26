"""
Generic line-search algorithms.
"""
import numpy as np
from . import parameters


def genericMonotone(lmData, updateCalculator, directionCalculator, lineSearch, f, Df, x):
    """Generic monotone line-search algorithm."""
    iter = np.array([0, 1])
    fx, gx = f(x), Df(x)

    while not stoppingTest(iter, gx):
        d = directionCalculator(lmData, gx)
        x, ok, fx, gx, it, sn, yn = lineSearch(x, f, Df, d, fx, gx)
        iter += [1, it]
        if not ok:
            break
        updateCalculator(lmData, sn, yn)

    return [x, iter]


def genericNonmonotone(lmData, updateCalculator, directionCalculator, lineSearch, f, Df, x):
    """Generic nonmonotone line-search algorithm."""
    iter = np.array([0, 1])
    fx, gx = f(x), Df(x)
    fxList = np.append(np.zeros(parameters.nonmon-1), fx)

    while not stoppingTest(iter, gx):
        d = directionCalculator(lmData, gx)
        fxRef = fx if iter[0]+1 < parameters.nonmon else max(fxList)
        x, ok, fx, gx, it, sn, yn = lineSearch(x, f, Df, d, fxRef, gx)
        fxList = np.append(fxList[1:], fx)
        iter += [1, it]
        if not ok:
            break
        updateCalculator(lmData, sn, yn)

    return [x, iter]


def stoppingTest(iter, gx):
    """Stopping criterion used for all line-search algorithms."""
    return iter[0] >= parameters.maxIter \
        or iter[1] >= parameters.maxEval \
        or np.linalg.norm(gx, np.inf) < parameters.tolGrad
