import pycutest
import RegularizedLSR1


p = pycutest.import_problem('DIXMAANA')

x = p.x0
f = lambda x: p.obj(x)
g = lambda x: p.lagjac(x)[0]

[x, iter] = RegularizedLSR1.RegularizedLSR1(f, g, x)
print(iter)