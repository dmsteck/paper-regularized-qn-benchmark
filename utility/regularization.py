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
from . import parameters, morethuente


def genericMonotone(lmData, updateCalculator, directionCalculator, f, Df, x):
    """Generic monotone regularization method."""
    iter = np.array([0, 1])
    fx, gx, mu = f(x), Df(x), 1

    if stoppingTest(iter, mu, gx):
        return [x, iter]
    
    fun = lambda x: (f(x), Df(x))
    d = -gx / np.linalg.norm(gx)
    xn,fn,gn,t,exls,it = \
        morethuente.cvsrch(fun,len(x),x,fx,gx,d,1,1e-4,0.9,1e-16,1e-20,1e20,20)
    
    iter += [1, it]
    if exls != 1:  # line search failed
        return [x, iter]
    
    updateCalculator(lmData, t * d, gn - gx)
    x, fx, gx = xn, fn, gn

    while not stoppingTest(iter, mu, gx):
        d = directionCalculator(lmData, mu, gx)
        pred = 0.5*mu*np.dot(d, d)-0.5*np.dot(gx, d)

        # Check whether predicted reduction is sufficient
        if not pred >= 1e-4*np.linalg.norm(gx)*np.linalg.norm(d):
            mu *= 4
            continue

        # Compute trial point and actual reduction
        xtry, ftry, ared = computeTrialPoint(x, f, fx, d)

        # Check whether iteration was successful
        if (ared <= 1e-4*pred):
            mu *= 4
            iter += [0, 1]
        else:
            x, fx, gx, yn = acceptTrialPoint(xtry, ftry, Df, gx)
            updateCalculator(lmData, d, yn)
            if (ared >= 0.9*pred):
                mu = max(1e-4, 0.5*mu)
            iter += [1, 1]

    return [x, iter]


def genericNonmonotone(lmData, updateCalculator, directionCalculator, f, Df, x):
    """Generic nonmonotone regularization method."""
    iter = np.array([0, 1])
    fx, gx, mu = f(x), Df(x), 1
    fxList = np.append(np.zeros(parameters.nonmon-1), fx)

    if stoppingTest(iter, mu, gx):
        return [x, iter]
    
    fun = lambda x: (f(x), Df(x))
    d = -gx / np.linalg.norm(gx)
    xn,fn,gn,t,exls,it = \
        morethuente.cvsrch(fun,len(x),x,fx,gx,d,1,1e-4,0.9,1e-16,1e-20,1e20,20)
    
    iter += [1, it]
    if exls != 1:  # line search failed
        return [x, iter]
    
    updateCalculator(lmData, t * d, gn - gx)
    x, fx, gx = xn, fn, gn
    fxList = np.append(fxList[1:], fx)

    while not stoppingTest(iter, mu, gx):
        d = directionCalculator(lmData, mu, gx)
        pred = 0.5*mu*np.dot(d, d)-0.5*np.dot(gx, d)

        # Check whether predicted reduction is sufficient
        if not pred >= 1e-4*np.linalg.norm(gx)*np.linalg.norm(d):
            mu *= 4
            continue

        # Compute trial point and actual reduction
        fxRef = fx if iter[0]+1 < parameters.nonmon else max(fxList)
        xtry, ftry, ared = computeTrialPoint(x, f, fxRef, d)

        # Check whether iteration was successful
        if (ared <= 1e-4*pred):
            mu *= 4
            iter += [0, 1]
        else:
            x, fx, gx, yn = acceptTrialPoint(xtry, ftry, Df, gx)
            fxList = np.append(fxList[1:], fx)
            updateCalculator(lmData, d, yn)
            if (ared >= 0.9*pred):
                mu = max(1e-4, 0.5*mu)
            iter += [1, 1]

    return [x, iter]


def stoppingTest(iter, mu, gx):
    """Generic stopping test used for all regularization algorithms."""
    return iter[0] >= parameters.maxIter \
        or iter[1] >= parameters.maxEval \
        or mu > parameters.maxReg \
        or np.linalg.norm(gx, np.inf) <= parameters.tolGrad
        #or np.linalg.norm(gx) <= parameters.tolGrad * max(1, np.linalg.norm(x))


def computeTrialPoint(x, f, fx, d):
    """Compute trial point, function value, and reduction."""
    xtry = x + d
    ftry = f(xtry)
    return xtry, ftry, fx - ftry


def acceptTrialPoint(xtry, ftry, Df, gx):
    """Accept trial point and assign new values."""
    gxNew = Df(xtry)
    return xtry, ftry, gxNew, gxNew - gx
