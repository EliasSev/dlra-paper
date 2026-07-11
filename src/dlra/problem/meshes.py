"""
Collection of mesh generation functions.
"""
from fenics import Point, UnitSquareMesh
from mshr import Circle, Polygon, generate_mesh


def get_donut_mesh(n: int, r: float = 0.3, R: float = 1.0):
    outer = Circle(Point(0, 0), R)
    inner = Circle(Point(0, 0), r)

    domain = outer - inner
    return generate_mesh(domain, n)


def get_square_mesh(n: int):
    return UnitSquareMesh(n, n)
    

def get_L_mesh(n: int):
    domain = Polygon([
        Point(0, 0),
        Point(2, 0),
        Point(2, 1),
        Point(1, 1),
        Point(1, 2),
        Point(0, 2)
    ])

    return generate_mesh(domain, n)


def get_ellipse_mesh(n: int):
    domain = Circle(Point(0,0), 1.0)
    mesh = generate_mesh(domain, n)

    # Stretch mesh into an ellipse
    X = mesh.coordinates()
    X[:,0] *= 2.0   # scale x
    X[:,1] *= 1.0   # scale y
    return mesh
