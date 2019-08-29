"""
Monotone and nonmonotone regularized L-BFGS methods.
"""
import numpy as np
from utility import regularization, parameters, limitedMemory


def solve(f, Df, x):
    "Monotone regularized L-BFGS method."
    lmData = limitedMemory.ExtendedLmData(x.shape[0], parameters.memory)
    return regularization.genericMonotone(lmData, updateLmData, inverseLBFGS, f, Df, x)

def solveNonmonotone(f, Df, x):
    "Nonmonotone regularized L-BFGS method."
    lmData = limitedMemory.ExtendedLmData(x.shape[0], parameters.memory)
    return regularization.genericNonmonotone(lmData, updateLmData, inverseLBFGS, f, Df, x)


def updateLmData(data, sn, yn):
    """Update L-BFGS data."""
    gamma = data.gamma
    sty = np.dot(sn, yn)
    if (sty >= 1e-8 * np.dot(sn, sn)):
        # Compute new gamma
        gamma = np.dot(yn, yn) / sty
        # Perform update
        data.update(sn, yn, gamma)


def inverseLBFGS(data, lam, g):
    """Compute regularized L-BFGS step."""
    mUpd = data.mUpd
    gamma = data.gamma
    gammah = data.gamma+lam

    Q = np.block([
        [-1/gamma*data.STS[:mUpd, :mUpd], -1/gamma *
            np.tril(data.STY[:mUpd, :mUpd], -1)],
        [-1/gamma*np.tril(data.STY[:mUpd, :mUpd], -1).T,
         np.diag(np.diag(data.STY[:mUpd, :mUpd]))]
    ])

    Q += 1/gammah * np.block([
        [data.STS[:mUpd, :mUpd], data.STY[:mUpd, :mUpd]],
        [data.STY[:mUpd, :mUpd].T, data.YTY[:mUpd, :mUpd]]
    ])

    ATg = np.block([data.S[:, :mUpd].T @ g, data.Y[:, :mUpd].T @ g])
    p = np.linalg.solve(Q, ATg)
    Ap = data.S[:, :mUpd] @ p[:mUpd] + data.Y[:, :mUpd] @ p[mUpd:]
    d = 1/gammah**2 * Ap - 1/gammah * g
    return d
