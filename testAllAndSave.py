"""

"""
import pycutest
import numpy as np
import multiprocessing
import time

# Algorithms
import regLSR1
import regLBFGS
import regLBFGSsec
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
    #problems = 'ARWHEAD', 'BDQRTIC', 'BOXPOWER', 'BROYDN3DLS', 'BROYDNBDLS', \
    #    'BRYBND', 'COSINE', 'CRAGGLVY', 'DIXMAANA', 'DIXMAANB', 'DIXMAANC', \
    #    'DIXMAAND','DIXMAANE','DIXMAANF','DIXMAANG','DIXMAANH', 'DIXMAANI', \
    #    'DIXMAANJ','DIXMAANK','DIXMAANL','DIXMAANM','DIXMAANN', 'DIXMAANO', \
    #    'DIXMAANP','DIXON3DQ','DQDRTIC','DQRTIC','EDENSCH', 'EG2', 'ENGVAL1', \
    #    'FLETBV3M','FLETCBV2','FMINSRF2','FMINSURF', 'FREUROTH','LIARWHD', \
    #    'MOREBV','NONDIA','NONDQUAR', 'PENALTY1','POWELLSG','QUARTC','SCHMVETT', \
    #    'SINQUAD', 'SPARSQUR','SROSENBR','TOINTGSS','TQUARTIC', 'WOODS'

    # List of problems that can be excluded because either:
    #  * the initial point is already stationary (FLETCBV2)
    #  * all algorithms are known to fail (the rest)
    exclude = 'BA-L16LS', 'BA-L21LS', 'BA-L49LS', 'BA-L52LS', 'BA-L73LS', \
        'CURLY30', 'FLETCBV2', 'FLETCBV3', 'FLETCHBV', 'INDEF', 'NONMSQRT', \
        'SBRYBND', 'SCOSINE', 'SCURLY10', 'SCURLY20', 'SCURLY30', 'SSCOSINE'

    # Use list comprehension to maintain order
    return [p for p in problems if p not in exclude]


def sharedArray(dtype, dims):
    """Create a shared numpy array."""
    mpArray = multiprocessing.Array(dtype, int(np.prod(dims)), lock=False)
    return np.frombuffer(mpArray, dtype=dtype).reshape(dims)


# It would be nice if we could parallelise on a more granular level
# (e.g., for every combination of problem, algorithm and mode) but
# sadly pycutest modules are not pickle-able, and importing a problem
# seems to be quite costly (might require a linking step).
def solveProblem(pId, problem):
    """
    Solve a named problem with all algorithms/modes and write the
    results into fx, opt, iter, and nf.
    """
    print(f"Starting problem: {problem}")
    pycutestProb = pycutest.import_problem(problem)
    def f(x): return pycutestProb.obj(x)
    def Df(x): return pycutestProb.lagjac(x)[0]
    for a, algorithm in enumerate(algorithms):
        for m, mode in enumerate(modes):
            # Solve problem with current algorithm and mode
            x, it = getattr(algorithm, mode)(f, Df, pycutestProb.x0)
            fx[m, a, pId] = f(x)
            #opt[m, a, pId] = np.linalg.norm(Df(x)) / max(1, np.linalg.norm(x))
            opt[m, a, pId] = np.linalg.norm(Df(x), np.inf)
            iter[m, a, pId], nf[m, a, pId] = it
    
    # To show where we are
    print(f"Completed problem: {problem}")


# Specify algorithms to run
algorithms = regLBFGS, armijoLBFGS, wolfeLBFGS, regLBFGSsec, regLSR1, regLPSB

# Specify modes to run
modes = 'solve', 'solveNonmonotone'

# List of problems to solve
problems = problemsToRun()

# Initialize output data
nModes, nAlgs, nProbs = len(modes), len(algorithms), len(problems)
fx = sharedArray('d', (nModes, nAlgs, nProbs))
opt = sharedArray('d', (nModes, nAlgs, nProbs))
iter = sharedArray('I', (nModes, nAlgs, nProbs))
nf = sharedArray('I', (nModes, nAlgs, nProbs))

# Solve all problems
tic = time.perf_counter()
pool = multiprocessing.Pool()
pool.starmap(solveProblem, enumerate(problems))
#for p, problem in enumerate(problems):
#    solveProblem(p, problem)
toc = time.perf_counter()
print(f"Total time: {toc - tic:0.4f} seconds")

# Save results to files
for i, m in enumerate(modes):
    for j, a in enumerate(algorithms):
        data = np.vstack([nf[i, j, :], iter[i, j, :], fx[i, j, :], opt[i, j, :]]).T
        np.savetxt(f"results/{a.__name__}_{m}.csv", data, header="nf, iter, fx, opt", fmt="%i, %i, %.5e, %.5e")
