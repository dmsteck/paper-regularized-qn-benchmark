"""
Utility functions and classes to reduce code duplication.
"""
import numpy as np


def updateData1(v, mUpd, vn):
    """Cyclically update limited memory vectors (e.g., sty)"""
    m = v.shape[0]
    if (mUpd >= m):
        v[:m-1] = v[1:]
        v[m-1] = vn
    else:
        v[mUpd] = vn


def updateData2(S, mUpd, sn):
    """Cyclically update limited memory matrices (e.g., S)"""
    m = S.shape[1]
    if (mUpd >= m):
        S[:, :m-1] = S[:, 1:]
        S[:, m-1] = sn
    else:
        S[:, mUpd] = sn


def updateData3(STY, mUpd, STyn, YTsn=None):
    """Cyclically update limited memory product matrices (e.g., STY)"""
    m = STY.shape[0]
    if (mUpd >= m):
        STY[:m-1, :m-1] = STY[1:, 1:]
        STY[:, m-1] = STyn
        STY[m-1, :] = STyn if YTsn is None else YTsn
    else:
        STY[:, mUpd] = STyn
        STY[mUpd, :] = STyn if YTsn is None else YTsn


def twoLoopRecursion(S, Y, rho, mUpd, gamma, rhs):
    """Compute trial step using the standard two-loop recursion"""
    alpha = np.zeros(S.shape[1])
    x = rhs.copy()

    # Two-loop recursion
    for i in reversed(range(mUpd)):
        alpha[i] = np.dot(S[:, i], x) / rho[i]
        x -= alpha[i] * Y[:, i]
    x /= gamma
    for i in range(mUpd):
        beta = np.dot(Y[:, i], x) / rho[i]
        x += (alpha[i] - beta) * S[:, i]
    
    return x


class LmData:
    """Basic limited-memory data structure"""
    def __init__(self, n, m):
        self.gamma = 1
        self.S = np.zeros((n, m))
        self.Y = np.zeros((n, m))
        self.sts = np.zeros(m)
        self.sty = np.zeros(m)
        self.yty = np.zeros(m)
        self.n = n
        self.m = m
        self.mUpd = 0

    def update(self, sn, yn, gamma):
        self.gamma = gamma
        updateData2(self.S, self.mUpd, sn)
        updateData2(self.Y, self.mUpd, yn)
        updateData1(self.sts, self.mUpd, np.dot(sn, sn))
        updateData1(self.sty, self.mUpd, np.dot(sn, yn))
        updateData1(self.yty, self.mUpd, np.dot(yn, yn))
        self.mUpd = min(self.mUpd+1, self.m)


class ExtendedLmData:
    """Extended limited-memory data structure"""
    def __init__(self, n, m):
        self.gamma = 1
        self.S = np.zeros((n, m))
        self.Y = np.zeros((n, m))
        self.STS = np.zeros((m, m))
        self.STY = np.zeros((m, m))
        self.YTY = np.zeros((m, m))
        self.n = n
        self.m = m
        self.mUpd = 0
    
    def update(self, sn, yn, gamma):
        self.gamma = gamma
        updateData2(self.S, self.mUpd, sn)
        updateData2(self.Y, self.mUpd, yn)
        updateData3(self.STS, self.mUpd, self.S.T @ sn)
        updateData3(self.STY, self.mUpd, self.S.T @ yn, self.Y.T @ sn)
        updateData3(self.YTY, self.mUpd, self.Y.T @ yn)
        self.mUpd = min(self.mUpd+1, self.m)


class NormalizedLmData:
    """Like ExtendedLmData but the matrices are kept normalized."""
    def __init__(self, n, m):
        self.gamma = 1

        self.Sn = np.zeros((n, m))
        self.Yn = np.zeros((n, m))
        self.SnTSn = np.zeros((m, m))
        self.SnTYn = np.zeros((m, m))
        self.YnTYn = np.zeros((m, m))
        self.sts = np.zeros(m)
        self.sty = np.zeros(m)
        self.yty = np.zeros(m)

        self.n = n
        self.m = m
        self.mUpd = 0
    
    def update(self, sn, yn, gamma):
        self.gamma = gamma
        norm_sn = np.linalg.norm(sn)
        norm_yn = np.linalg.norm(yn)

        updateData2(self.Sn, self.mUpd, sn / norm_sn)
        updateData2(self.Yn, self.mUpd, yn / norm_yn)
        updateData3(self.SnTSn, self.mUpd, self.Sn.T @ (sn / norm_sn))
        updateData3(self.SnTYn, self.mUpd, self.Sn.T @ (yn / norm_yn), self.Yn.T @ (sn / norm_sn))
        updateData3(self.YnTYn, self.mUpd, self.Yn.T @ (yn / norm_yn))
        updateData1(self.sts, self.mUpd, np.dot(sn, sn))
        updateData1(self.sty, self.mUpd, np.dot(sn, yn))
        updateData1(self.yty, self.mUpd, np.dot(yn, yn))

        self.mUpd = min(self.mUpd+1, self.m)
