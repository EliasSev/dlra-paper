"""
Functions for source generation. `V` denotes a FunctionSpace (fenics).
"""
from fenics import Expression, Function, interpolate


def get_square_f(V, x0=0.5, y0=0.5, w=0.15, h=0.15):
    x1 = x0 + w
    y1 = y0 + h
    code = f'x[0] >= {x0} && x[0] <= {x1} && x[1] >= {y0} && x[1] <= {y1} ? 1.0 : 0.0'
    f_expr = Expression(code, degree=1)
    f = Function(V)
    f.interpolate(f_expr)
    return f


def get_Gaussian_f(V, x=0.5, y=0.5, sigma=0.05, A=1.0):
    f_expr = Expression(
        "A*exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / (2*sigma*sigma))",
        degree=4, A=A, x0=x, y0=y, sigma=sigma
    )
    return interpolate(f_expr, V)


def get_disk_f(V, x, y, r=0.05):
    f_expr = Expression(
        "((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0) <= r*r) ? 1.0 : 0.0",
        degree=1, x0=x, y0=y, r=r
    )
    return interpolate(f_expr, V)
