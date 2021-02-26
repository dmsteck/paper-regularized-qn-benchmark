"""
This file contains algorithm parameters shared by all implemented methods.

The purpose of this file is to reduce duplication and enforce comparability.
"""


# Stopping criteria
maxIter = 100_000
maxEval = 100_000
tolGrad = 1e-4
maxReg = 1e15
minStep = 1e-15

# Cautious update threshold
crvThreshold = 1e-8

# Limited memory bound
memory = 5

# Nonmonotonicity bound
nonmon = 8
