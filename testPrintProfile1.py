"""
...
"""
import numpy as np
import matplotlib.pyplot as plt
from utility import parameters
from utility import perfprof


algorithms = ['regLBFGS', 'regLBFGSsec', 'regLSR1', 'regLPSB']


# Read all the data in the ugliest fashion possible
nfM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 0] for a in algorithms])
iterM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 1] for a in algorithms])
fxM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 2] for a in algorithms])
optM = np.array([np.loadtxt(f"results/{a}_solve.csv", delimiter=',')[:, 3] for a in algorithms])
nfN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 0] for a in algorithms])
iterN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 1] for a in algorithms])
fxN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 2] for a in algorithms])
optN = np.array([np.loadtxt(f"results/{a}_solveNonmonotone.csv", delimiter=',')[:, 3] for a in algorithms])


# Discard problems that weren't solved
nfM[optM > parameters.tolGrad] = np.inf
nfN[optN > parameters.tolGrad] = np.inf


# Print!
palette = ['o-r', 'o:b', 'o--c', 'o-.g', 'o:k', 'o-y', 'o:m', 'o--b']
perfprof.perfprof(nfM.T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend(['regLBFGS', 'regLBFGSsec', 'regLSR1', 'regLPSB'], loc=4, fontsize=16)
plt.savefig("figures/Monotone1.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
perfprof.perfprof(nfN.T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend([r'regLBFGS$_n$', r'regLBFGSsec$_n$', r'regLSR1$_n$', r'regLPSB$_n$'], loc=4, fontsize=16)
plt.savefig("figures/Nonmonotone1.pdf", bbox_inches='tight', pad_inches=0)
plt.show()

# Print monotone vs nonmonotone comparison
perfprof.perfprof(np.vstack([nfM[:2, :], nfN[:2, :]]).T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend(['regLBFGS', 'regLBFGSsec', r'regLBFGS$_n$', r'regLBFGSsec$_n$'], loc=4, fontsize=16)
plt.savefig("figures/All11.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
perfprof.perfprof(np.vstack([nfM[2:, :], nfN[2:, :]]).T, linestyle=palette, thmax=5., markersize=4, markevery=[0])
plt.legend(['regLSR1', 'regLPSB', r'regLSR1$_n$', r'regLPSB$_n$'], loc=4, fontsize=16)
plt.savefig("figures/All12.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
