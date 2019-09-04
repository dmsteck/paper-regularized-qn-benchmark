"""

"""
import pycutest
import numpy as np
import multiprocessing
import SharedArray
import regLSR1
import regLBFGS
import regLPSB
import armijoLBFGS
import wolfeLBFGS


def problemsToRun():
    """..."""
    # General list of eligible problems
    problems = 'ARWHEAD', 'BA-L16LS', 'BA-L21LS', 'BA-L49LS', 'BA-L52LS', \
        'BA-L73LS', 'BDQRTIC', 'BOX', 'BOXPOWER', 'BROYDN3DLS', 'BROYDN7D', \
        'BROYDNBDLS', 'BRYBND', 'CHAINWOO', 'COSINE', 'CRAGGLVY', 'CURLY10', \
        'CURLY20', 'CURLY30', 'DIXMAANA', 'DIXMAANB', 'DIXMAANC', 'DIXMAAND', \
        'DIXMAANE', 'DIXMAANF', 'DIXMAANG', 'DIXMAANH', 'DIXMAANI', 'DIXMAANJ', \
        'DIXMAANK', 'DIXMAANL', 'DIXMAANM', 'DIXMAANN', 'DIXMAANO', 'DIXMAANP', \
        'DIXON3DQ', 'DQDRTIC', 'DQRTIC', 'EDENSCH', 'EG2', 'EIGENALS', 'EIGENBLS', \
        'EIGENCLS', 'ENGVAL1', 'EXTROSNB', 'FLETBV3M', 'FLETCBV2', 'FLETCBV3', \
        'FLETCHBV', 'FLETCHCR', 'FMINSRF2', 'FMINSURF', 'FREUROTH', 'GENHUMPS', \
        'INDEF', 'INDEFM', 'JIMACK', 'LIARWHD', 'MODBEALE', 'MOREBV', 'MSQRTALS', \
        'MSQRTBLS', 'NCB20', 'NCB20B', 'NONCVXU2', 'NONCVXUN', 'NONDIA', 'NONDQUAR', \
        'NONMSQRT', 'OSCIGRAD', 'PENALTY1', 'POWELLSG', 'POWER', 'QUARTC', 'SBRYBND', \
        'SCHMVETT', 'SCOSINE', 'SCURLY10', 'SCURLY20', 'SCURLY30', 'SINQUAD', \
        'SPARSINE', 'SPARSQUR', 'SPMSRTLS', 'SROSENBR', 'SSBRYBND', 'SSCOSINE', \
        'TESTQUAD', 'TOINTGSS', 'TQUARTIC', 'TRIDIA', 'WOODS', 'YATP1LS', 'YATP2LS'

    # Fast problems
    #problems = 'ARWHEAD', 'BDQRTIC', 'BOXPOWER', 'BROYDN3DLS', 'BROYDN7D', \
    #    'BROYDNBDLS', 'BRYBND', 'COSINE', 'CRAGGLVY', 'DIXMAANA', 'DIXMAANB', \
    #    'DIXMAANC','DIXMAAND','DIXMAANE','DIXMAANF','DIXMAANG','DIXMAANH', \
    #    'DIXMAANI','DIXMAANJ','DIXMAANK','DIXMAANL','DIXMAANM','DIXMAANN', \
    #    'DIXMAANO','DIXMAANP','DIXON3DQ','DQDRTIC','DQRTIC','EDENSCH', \
    #    'EG2','ENGVAL1','FLETBV3M','FLETCBV2','FMINSRF2','FMINSURF', \
    #    'FREUROTH','LIARWHD','MOREBV','NCB20B','NONDIA','NONDQUAR', \
    #    'PENALTY1','POWELLSG','QUARTC','SCHMVETT','SINQUAD', \
    #    'SPARSQUR','SROSENBR','TOINTGSS','TQUARTIC','WOODS','YATP1LS', \
    #    'YATP2LS'

    # List of problems that can be excluded because all algorithms are known to fail
    exclude = 'BA-L16LS', 'BA-L21LS', 'BA-L49LS', 'BA-L52LS', 'BA-L73LS', \
        'CURLY30', 'FLETCBV3', 'FLETCHBV', 'INDEF', 'NONMSQRT', 'SBRYBND', \
        'SCOSINE', 'SCURLY10', 'SCURLY20', 'SCURLY30', 'SPARSINE', 'SSCOSINE'

    return set(problems) - set(exclude)


def prepareSharedArrays():
    """Prepares output data arrays."""    
    nModes, nAlgs, nProbs = len(modes), len(algorithms), len(problems)
    fx = SharedArray.create('fx', (nModes, nAlgs, nProbs))
    normDfx = SharedArray.create('normDfx', (nModes, nAlgs, nProbs))
    iter = SharedArray.create('iter', (nModes, nAlgs, nProbs), dtype=int)
    nf = SharedArray.create('nf', (nModes, nAlgs, nProbs), dtype=int)
    return fx, normDfx, iter, nf


def solveProblem(pId, problem):
    """
    Solve a named problem with all algorithms/modes and write the
    results into fx, normDfx, iter, and nf.
    """
    pycutestProb = pycutest.import_problem(problem)
    def f(x): return pycutestProb.obj(x)
    def Df(x): return pycutestProb.lagjac(x)[0]
    for a, algorithm in enumerate(algorithms):
        for m, mode in enumerate(modes):
            # Solve problem with current algorithm and mode
            x, it = getattr(algorithm, mode)(f, Df, pycutestProb.x0)
            fx[m, a, pId] = f(x)
            normDfx[m, a, pId] = np.linalg.norm(Df(x), np.inf)
            iter[m, a, pId], nf[m, a, pId] = it


# Specify algorithms to run
algorithms = armijoLBFGS, wolfeLBFGS, regLBFGS, regLSR1, regLPSB

# Specify modes to run
modes = 'solve', 'solveNonmonotone'

# List of problems to solve
problems = problemsToRun()

# Initialize output data
fx, normDfx, iter, nf = prepareSharedArrays()

# Solve all problems
pool = multiprocessing.Pool()
pool.starmap(solveProblem, enumerate(problems))
#for p, problem in enumerate(problems):
#    solveProblem(p, problem)

# Save results to file
np.savez('results', fx=fx, normDfx=normDfx, iter=iter, nf=nf)

# Clear shared arrays
SharedArray.delete(fx.base.name)
SharedArray.delete(normDfx.base.name)
SharedArray.delete(iter.base.name)
SharedArray.delete(nf.base.name)
