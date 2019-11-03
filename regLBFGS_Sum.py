"""

"""
import numpy as np
from utility import regularization, parameters, limitedMemory


def solve(f, Df, x):
    lmData = limitedMemory.LmData(x.shape[0], parameters.nonmon)
    return regularization.genericMonotone(lmData, updateLmData, calculateStep, f, Df, x)


def solveNonmonotone(f, Df, x):
    lmData = limitedMemory.LmData(x.shape[0], parameters.nonmon)
    return regularization.genericNonmonotone(lmData, updateLmData, calculateStep, f, Df, x)

# .............

def updateLmData(data, sn, yn):
    gamma = data.gamma
    sty = np.dot(sn, yn)
    if (sty >= parameters.crvThreshold * np.dot(sn, sn)):
        # Compute new gamma
        gamma = np.dot(yn, yn) / sty
        # Perform update
        data.update(sn, yn, gamma)

# .............

# data is LM data, lam regularization
def calculateStep(data, lam, g):
    return limitedMemory.twoLoopRecursion(data.S, \
        data.Y + lam * data.S,
        data.sty + lam * data.sts,
        data.mUpd,
        data.gamma + lam,
        -g)
