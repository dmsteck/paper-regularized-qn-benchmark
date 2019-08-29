"""
This file contains generic algorithm prototypes for monotone and
nonmonotone limited-memory regularization methods. The behaviour of
the functions can be controlled through the following parameters:
  * lmData: a structure containing limited memory data. This
    object does not need to follow any particular interface.
  * updateCalculator: a function updating the lmData object in
    case of a successful step.
  * directionCalculator: a function calculating the search
    direction based on the contents of lmData
"""
import numpy as np
from . import parameters


def genericMonotone(lmData, updateCalculator, directionCalculator, f, Df, x):
    """Generic monotone regularization method."""
    iter = np.array([0, 0])
    fx, gx, lam = f(x), Df(x), 1.0

    while not stoppingTest(iter, lam, gx):
        d = directionCalculator(lmData, lam, gx)
        pred = 0.5*lam*np.dot(d, d)-0.5*np.dot(gx, d)

        # Check whether predicted reduction is sufficient
        if not pred >= 1e-4*np.linalg.norm(gx)*np.linalg.norm(d):
            lam *= 4
            continue

        # Compute trial point and actual reduction
        xtry, ftry, ared = computeTrialPoint(x, f, fx, d)

        # Check whether iteration was successful
        if (ared <= 1e-4*pred):
            lam *= 4
            iter += [0, 1]
        else:
            x, fx, gx, yn = acceptTrialPoint(xtry, ftry, Df, gx)
            updateCalculator(lmData, d, yn)
            if (ared >= 0.9*pred):
                lam = max(1e-4, 0.5*lam)
            iter += [1, 1]

    return [x, iter]


def genericNonmonotone(lmData, updateCalculator, directionCalculator, f, Df, x):
    """Generic nonmonotone regularization method."""
    iter = np.array([0, 0])
    fx, gx, lam = f(x), Df(x), 1.0
    fx_list = np.append(np.zeros(parameters.nonmon-1), fx)

    while not stoppingTest(iter, lam, gx):
        d = directionCalculator(lmData, lam, gx)
        pred = 0.5*lam*np.dot(d, d)-0.5*np.dot(gx, d)

        # Check whether predicted reduction is sufficient
        if not pred >= 1e-4*np.linalg.norm(gx)*np.linalg.norm(d):
            lam *= 4
            continue

        # Compute trial point and actual reduction
        fx_ref = fx if iter[0]+1 < parameters.nonmon else max(fx_list)
        xtry, ftry, ared = computeTrialPoint(x, f, fx_ref, d)

        # Check whether iteration was successful
        if (ared <= 1e-4*pred):
            lam *= 4
            iter += [0, 1]
        else:
            x, fx, gx, yn = acceptTrialPoint(xtry, ftry, Df, gx)
            fx_list = np.append(fx_list[1:], fx)
            updateCalculator(lmData, d, yn)
            if (ared >= 0.9*pred):
                lam = max(1e-4, 0.5*lam)
            iter += [1, 1]

    return [x, iter]


def stoppingTest(iter, lam, gx):
    """Generic stopping test used for all regularization algorithms."""
    return iter[0] >= parameters.maxIter \
        or iter[1] >= parameters.maxEval \
        or lam > parameters.maxReg \
        or np.linalg.norm(gx, np.inf) < parameters.tolGrad


def computeTrialPoint(x, f, fx, d):
    """Compute trial point, function value, and reduction."""
    xtry = x + d
    ftry = f(xtry)
    return xtry, ftry, fx - ftry


def acceptTrialPoint(xtry, ftry, Df, gx):
    """Accept trial point and assign new values."""
    gx_new = Df(xtry)
    return xtry, ftry, gx_new, gx_new - gx
