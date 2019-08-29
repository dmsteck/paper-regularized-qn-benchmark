"""
Monotone and nonmonotone regularized L-SR1 methods.
"""
import numpy as np
from utility import regularization, parameters, limitedMemory


def solve(f, Df, x):
    """Monotone regularized L-SR1 method."""
    lmData = limitedMemory.ExtendedLmData(x.shape[0], parameters.memory)
    return regularization.genericMonotone(lmData, updateLmData, inverseLSR1, f, Df, x)

def solveNonmonotone(f, Df, x):
    """Nonmonotone regularized L-SR1 method."""
    lmData = limitedMemory.ExtendedLmData(x.shape[0], parameters.memory)
    return regularization.genericNonmonotone(lmData, updateLmData, inverseLSR1, f, Df, x)


def updateLmData(data, sn, yn):
    """Update L-SR1 data."""
    gamma = data.gamma
    sty = np.dot(sn, yn)
    if (sty >= 1e-8 * np.dot(sn, sn)):
        # Compute new gamma
        gamma = np.dot(yn, yn) / sty
    # Perform update
    data.update(sn, yn, gamma)


def inverseLSR1(data, lam, g):
    """Compute regularized L-SR1 step."""
    mUpd = data.mUpd
    gamma = data.gamma
    gammah = data.gamma+lam

    A = data.Y[:, :mUpd] - gamma * data.S[:, :mUpd]
    Q = np.diag(np.diag(data.STY[:mUpd, :mUpd])) - gamma * data.STS[:mUpd, :mUpd] + \
        np.tril(data.STY[:mUpd, :mUpd], -1) + np.tril(data.STY[:mUpd, :mUpd], -1).T
    Q += 1/gammah * (data.YTY[:mUpd, :mUpd] + gamma**2*data.STS[:mUpd, :mUpd] -
        gamma * data.STY[:mUpd, :mUpd] - gamma * data.STY[:mUpd, :mUpd].T)
    
    [L, U, piv] = adaptiveLU(Q)
    ATg = A[:, piv].T @ g
    p = np.linalg.solve(U, np.linalg.solve(L, ATg))
    Ap = A[:, piv] @ p
    d = 1 / gammah**2 * Ap - 1 / gammah * g
    return d


def adaptiveLU(A):
    """Adaptive LU decomposition where rows/columns corresponding to zero pivots are discarded."""
    m = A.shape[0]
    piv = np.full(m, False)
    for i in range(m):
        if (abs(A[i, i]) >= 1e-6):
            piv[i] = True
            A[i+1:, i] /= A[i, i]
            A[i+1:, i+1:] -= np.outer(A[i+1:, i], A[i, i+1:])
    
    L = np.tril(A[np.ix_(piv, piv)], -1) + np.eye(np.sum(piv))
    return L, np.triu(A[np.ix_(piv, piv)]), piv
