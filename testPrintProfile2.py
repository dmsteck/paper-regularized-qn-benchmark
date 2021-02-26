"""
...
"""
import numpy as np
import matplotlib.pyplot as plt
from utility import parameters
from utility import perfprof


algorithms = ['regLBFGS', 'armijoLBFGS', 'wolfeLBFGS']


# Read all the data in the ugliest fashion possible
nfM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 0] for a in algorithms])
iterM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 1] for a in algorithms])
fxM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 2] for a in algorithms])
optM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 3] for a in algorithms])
nfN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 0] for a in algorithms])
iterN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 1] for a in algorithms])
fxN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 2] for a in algorithms])
optN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 3] for a in algorithms])


# Load LMTR results
lmtrResults = np.loadtxt('results/lmtr.csv', delimiter=',')
fxM = np.vstack([fxM, lmtrResults[:, 2]])
optM = np.vstack([optM, lmtrResults[:, 3]])
iterM = np.vstack([iterM, lmtrResults[:, 0]])
nfM = np.vstack([nfM, lmtrResults[:, 1]])


# Discard problems that weren't solved
nfM[optM > parameters.tolGrad] = np.inf
nfN[optN > parameters.tolGrad] = np.inf


# Print!
palette = ['o-r', 'o:b', 'o--c', 'o-.g', 'o:k', 'o-y', 'o:m', 'o--b']
perfprof.perfprof(nfM.T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend(['regLBFGS', 'armijoLBFGS', 'wolfeLBFGS', 'eigLBFGS'], loc=4, fontsize=16)
plt.savefig("figures/Monotone2.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
perfprof.perfprof(nfN.T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend([r'regLBFGS$_n$', r'armijoLBFGS$_n$', r'wolfeLBFGS$_n$'], loc=4, fontsize=16)
plt.savefig("figures/Nonmonotone2.pdf", bbox_inches='tight', pad_inches=0)
plt.show()

# Print monotone vs nonmonotone comparison
perfprof.perfprof(np.vstack([nfM[1:3, :], nfN[1:3, :]]).T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend(['armijoLFBGS', 'wolfeLBFGS', r'armijoLFBGS$_n$', r'wolfeLBFGS$_n$'], loc=4, fontsize=16)
plt.savefig("figures/All21.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
perfprof.perfprof(np.vstack([nfM[0, :], nfM[3, :], nfN[0, :]]).T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend(['regLBFGS', 'eigLBFGS', r'regLBFGS$_n$'], loc=4, fontsize=16)
plt.savefig("figures/All22.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
