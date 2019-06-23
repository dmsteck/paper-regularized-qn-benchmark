import pycutest
#import RegularizedLSR1
import RegularizedLBFGS


p = pycutest.import_problem('DIXMAANA')

x = p.x0
f = lambda x: p.obj(x)
g = lambda x: p.lagjac(x)[0]

#[x, iter] = RegularizedLSR1.solve(f, g, x)
[x, iter] = RegularizedLBFGS.solve(f, g, x)
print(iter)