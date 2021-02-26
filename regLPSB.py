"""
Monotone and nonmonotone regularized L-PSB methods.
"""
import numpy as np
from utility import regularization, parameters, limitedMemory


def solve(f, Df, x):
    """Monotone regularized L-PSB method."""
    lmData = limitedMemory.ExtendedLmData(x.shape[0], parameters.memory)
    return regularization.genericMonotone(lmData, updateLmData, inverseLPSB, f, Df, x)


def solveNonmonotone(f, Df, x):
    """Nonmonotone regularized L-PSB method."""
    lmData = limitedMemory.ExtendedLmData(x.shape[0], parameters.memory)
    return regularization.genericNonmonotone(lmData, updateLmData, inverseLPSB, f, Df, x)


def updateLmData(data, sn, yn):
    """Update L-PSB data."""
    gamma = data.gamma
    sty = np.dot(sn, yn)
    if (sty >= parameters.crvThreshold * np.dot(sn, sn)):
        # Compute new gamma
        gamma = np.dot(yn, yn) / sty
    # Perform update
    data.update(sn, yn, gamma)


def inverseLPSB(data, mu, g):
    """Compute regularized L-PSB step."""
    mUpd = data.mUpd
    gamma = data.gamma
    gammah = data.gamma + mu

    Q22 = np.tril(data.STY[:mUpd, :mUpd], -1) + np.tril(data.STY[:mUpd, :mUpd], -1).T + \
        np.diag(np.diag(data.STY[:mUpd, :mUpd])) + \
        gamma * np.diag(np.diag(data.STS[:mUpd, :mUpd]))

    Q = np.block([
        [np.zeros((mUpd, mUpd)), np.triu(data.STS[:mUpd, :mUpd])],
        [np.triu(data.STS[:mUpd, :mUpd]).T, Q22]
    ])

    Q += 1/gammah * np.block([
        [data.STS[:mUpd, :mUpd], data.STY[:mUpd, :mUpd]],
        [data.STY[:mUpd, :mUpd].T, data.YTY[:mUpd, :mUpd]]
    ])

    ATg = np.block([data.S[:, :mUpd].T @ g, data.Y[:, :mUpd].T @ g])
    p = np.linalg.solve(Q, ATg)
    #p = scipy.linalg.solve(Q, ATg, assume_a='sym')
    Ap = data.S[:, :mUpd] @ p[:mUpd] + data.Y[:, :mUpd] @ p[mUpd:]
    d = 1/gammah**2 * Ap - 1/gammah * g
    return d
