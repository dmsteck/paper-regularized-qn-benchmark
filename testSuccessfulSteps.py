import numpy as np


algorithms = ['regLBFGS', 'regLBFGSsec', 'regLSR1', 'regLPSB', 'armijoLBFGS', 'wolfeLBFGS']


# Read data for the algorithms above
nfM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 0] for a in algorithms])
iterM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 1] for a in algorithms])
nfN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 0] for a in algorithms])
iterN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 1] for a in algorithms])


# Load LMTR results
lmtrResults = np.loadtxt('results/lmtr.csv', delimiter=',')
iterM = np.vstack([iterM, lmtrResults[:, 0]])
nfM = np.vstack([nfM, lmtrResults[:, 1]])


#successM = np.sum(iterM / nfM, axis=1) / iterM.shape[1]
#successN = np.sum(iterN / nfN, axis=1) / iterN.shape[1]
successM = np.sum(iterM, axis=1) / np.sum(nfM, axis=1)
successN = np.sum(iterN, axis=1) / np.sum(nfN, axis=1)


print(f"Average step success rate for monotone algorithms {algorithms + ['eigLBFGS']}")
print(successM)
print(f"Average step success rate for nonmonotone algorithms {algorithms}")
print(successN)
