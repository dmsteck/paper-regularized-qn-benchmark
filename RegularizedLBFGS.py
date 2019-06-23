import numpy as np
import collections


def solve(f, Df, x, m=5, tol=1e-6, lam=1.0):
    data = InitLMdata(x.shape[0], m)
    iter = np.array([0, 0])
    fx = f(x)
    gx = Df(x)

    while not StoppingTest(iter, lam, gx, tol):
        [data, d] = InverseLBFGS(data, lam, gx)
        pred = 0.5*lam*np.dot(d, d)-0.5*np.dot(gx, d)

        # Check whether predicted reduction is sufficient
        if not pred >= 1e-4*np.linalg.norm(gx)*np.linalg.norm(d):
            lam *= 4
            continue

        # Compute trial point and actual reduction
        xtry = x + d
        ftry = f(xtry)
        ared = fx-ftry

        # Check whether iteration was successful
        if (ared <= 1e-4*pred):
            lam *= 4
            iter += [0, 1]
        else:
            x = xtry
            fx = ftry
            gx_new = Df(x)
            data = UpdateLMdata(data, d, gx_new - gx)
            gx = gx_new
            if (ared >= 0.9*pred):
                lam = max(1e-4, 0.5*lam)
            iter += [1, 1]

    return [x, iter]


# .............

def InitLMdata(n, m):
    data = collections.namedtuple("data", "gamma S Y STS STY YTY n m mUpd")
    data.gamma = 1
    data.S = np.zeros((n, m))
    data.Y = np.zeros((n, m))
    data.STS = np.zeros((m, m))
    data.STY = np.zeros((m, m))
    data.YTY = np.zeros((m, m))
    data.n = n
    data.m = m
    data.mUpd = 0
    return data


# .............

def StoppingTest(iter, lam, gx, tol):
    return iter[0] >= 10000 or lam > 1e15 or np.linalg.norm(gx, np.inf) < tol


# .............

# modify in place possible?
def UpdateLMdata(data, sn, yn):
    m = data.m
    mUpd = data.mUpd
    # Check well-definedness
    if np.dot(yn, sn) <= 1e-8*np.dot(sn, sn):
        return data

    # Perform update
    if (mUpd >= m):
        # Update S and Y
        data.S[:, :m-1] = data.S[:, 1:]
        data.S[:, m-1] = sn
        data.Y[:, :m-1] = data.Y[:, 1:]
        data.Y[:, m-1] = yn
        # Update STS
        data.STS[:m-1, :m-1] = data.STS[1:, 1:]
        data.STS[:, m-1] = data.S.T @ sn
        data.STS[m-1, :] = data.STS[:, m-1]
        # Update STY
        data.STY[:m-1, :m-1] = data.STY[1:, 1:]
        data.STY[:, m-1] = data.S.T @ yn
        data.STY[m-1, :] = data.Y.T @ sn
        # Update YTY
        data.YTY[:m-1, :m-1] = data.YTY[1:, 1:]
        data.YTY[:, m-1] = data.Y.T @ yn
        data.YTY[m-1, :] = data.YTY[:, m-1]
        # Update gamma (note that well-definedness is already checked)
        data.gamma = np.dot(yn, yn) / np.dot(yn, sn)
    else:
        # Update S and Y
        data.S[:, mUpd] = sn
        data.Y[:, mUpd] = yn
        # Update STS
        data.STS[:mUpd+1, mUpd] = data.S[:, :mUpd+1].T @ sn
        data.STS[mUpd, :mUpd+1] = data.STS[:mUpd+1, mUpd].T
        # Update STY
        data.STY[:mUpd+1, mUpd] = data.S[:, :mUpd+1].T @ yn
        data.STY[mUpd, :mUpd+1] = data.Y[:, :mUpd+1].T @ sn
        # Update YTY
        data.YTY[:mUpd+1, mUpd] = data.Y[:, :mUpd+1].T @ yn
        data.YTY[mUpd, :mUpd+1] = data.YTY[:mUpd+1, mUpd].T
        # Update gamma (note that well-definedness is already checked)
        data.gamma = np.dot(yn, yn) / np.dot(yn, sn)
        data.mUpd = mUpd+1

    return data


# .............

# data is BFGS data, lambda regularization
def InverseLBFGS(data, lam, g):
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
    return [data, d]
