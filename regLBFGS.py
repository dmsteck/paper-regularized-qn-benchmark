"""
Monotone and nonmonotone regularized L-BFGS methods.
"""
import numpy as np
import scipy.linalg
from utility import regularization, parameters, limitedMemory


def solve(f, Df, x):
    "Monotone regularized L-BFGS method."
    lmData = limitedMemory.NormalizedLmData(x.shape[0], parameters.memory)
    return regularization.genericMonotone(lmData, updateLmData, inverseLBFGS, f, Df, x)

def solveNonmonotone(f, Df, x):
    "Nonmonotone regularized L-BFGS method."
    lmData = limitedMemory.NormalizedLmData(x.shape[0], parameters.memory)
    return regularization.genericNonmonotone(lmData, updateLmData, inverseLBFGS, f, Df, x)


def updateLmData(data, sn, yn):
    """Update L-BFGS data."""
    sty = np.dot(sn, yn)
    if (sty >= parameters.crvThreshold * np.dot(sn, sn)):
        # Compute new gamma
        gamma = np.dot(yn, yn) / sty
        # Perform update
        data.update(sn, yn, gamma)


def inverseLBFGS(data, mu, g):
    """Compute regularized L-BFGS step."""
    mUpd = data.mUpd
    gamma = data.gamma
    gammah = data.gamma + mu

    Q11 = gammah * np.diag(data.sty[:mUpd] / data.yty[:mUpd]) + data.YnTYn[:mUpd, :mUpd]
    Q21 = np.triu(data.SnTYn[:mUpd, :mUpd]) - mu/gamma * np.tril(data.SnTYn[:mUpd, :mUpd], -1)
    Q22 = -mu/gamma * data.SnTSn[:mUpd, :mUpd]

    M = np.linalg.cholesky(Q11)
    MinvQ21T = scipy.linalg.solve_triangular(M, Q21.T, lower=True)
    QoverQ11 = Q22 - Q21 @ scipy.linalg.solve_triangular(M.T, MinvQ21T)
    J = np.linalg.cholesky(-QoverQ11)

    # Triangular factorization Q = Qfactor1 @ Qfactor2
    Qfactor1 = np.block([[M, np.zeros((mUpd, mUpd))], [MinvQ21T.T, -J]])
    Qfactor2 = np.block([[M.T, MinvQ21T], [np.zeros((mUpd, mUpd)), J.T]])

    ATg = np.block([data.Yn[:, :mUpd].T @ g, data.Sn[:, :mUpd].T @ g])
    p = scipy.linalg.solve_triangular(Qfactor2, scipy.linalg.solve_triangular(Qfactor1, ATg, lower=True))
    Ap = data.Yn[:, :mUpd] @ p[:mUpd] + data.Sn[:, :mUpd] @ p[mUpd:]
    d = 1/gammah * Ap - 1/gammah * g

    return d
