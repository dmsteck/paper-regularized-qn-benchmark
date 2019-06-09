import numpy as np
import collections

def RegularizedLSR1( f, Df, x, m=5, tol=1e-6, lam=1.0 ):
    data=InitLMdata(x.shape[0],m)
    iter=np.array([0,0])
    fx=f(x); gx=Df(x)

    while (not StoppingTest(iter,lam,gx,tol)):
        d=InverseLSR1(data,iter[1],lam,-gx)
        xtry=x+d; ftry=f(xtry)

        if (ftry<=fx-1e-8*np.dot(d,d)):
            x=xtry; fx=ftry; gx_new=Df(x)
            data=UpdateLMdata(data,iter[1],d,gx_new-gx)
            gx=gx_new; lam=0.5*lam
            iter=iter+[1,1]
        else:
            lam=2*lam
            iter=iter+[1,0]

    return [x, iter]

# .............

def InitLMdata( n, m ):
    data = collections.namedtuple('data', 'gamma, S, Y, n, m')
    data.gamma = 1
    data.S = np.zeros((n, m))
    data.Y = np.zeros((n, m))
    data.n = n
    data.m = m
    return data

# .............

def StoppingTest( iter, lam, gx, tol ):
    return (iter[0] >= 10000
            or lam > 1e15
            or np.linalg.norm(gx, np.inf) < tol
            )

# .............

# modify in place possible?
def UpdateLMdata( data, iter, sn, yn ):
    if (iter>=data.m):
        data.S[:, 0:data.m-2] = data.S[:,1:data.m-1]
        data.Y[:, 0:data.m-2] = data.S[:,1:data.m-1]
        data.S[:, data.m-1] = sn
        data.Y[:, data.m-1] = yn
    else:
        data.S[:, iter] = sn
        data.Y[:, iter] = yn
    return data

# .............

# data is SR1 data, Dv initial diagonal, lam regularization
def InverseLSR1( data, iter, mu, rhs ):
    yls = data.Y+mu*data.S
    Hy = yls/(data.gamma+mu)
    x=rhs/(data.gamma+mu)
    m=data.m
    
    for i in range(1, min(m, iter)):
        v = data.S[:, i]-Hy[:, i]
        q = np.dot(v, yls[:, i])
        if (abs(q)>1e-8*np.linalg.norm(v)*np.linalg.norm(yls[:, i])):
            Hy[:, i+1:m] = Hy[:, i+1:m]+(1/q)*np.outer(v, yls[:, i+1:m].T @ v)
            x = x+(1/q)*v*np.dot(v, rhs)
    return x
