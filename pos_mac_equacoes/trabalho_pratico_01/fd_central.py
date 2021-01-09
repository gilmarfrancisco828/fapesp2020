import autograd.numpy as np

def fd_central(r, q, p, x, y_a, y_b, method="central"):

    if type( x ) != np.ndarray:
        if type( x ) == list:
            x = np.array( x )
        else:
            x = np.array( [ float( x ) ] )

    n = len( x )

    # Make sure that u, v, and w are either scalars or n-element vectors.
    # If they are scalars then we create vectors with the scalar value in
    # each position.

    if type( r ) == int or type( r ) == float:
        r = np.array( [ float( r ) ] * n )

    if type( q ) == int or type( q ) == float:
        q = np.array( [ float( q ) ] * n )

    if type( p ) == int or type( p ) == float:
        p = np.array( [ float( p ) ] * n )

    # Compute the stepsize.  It is assumed that all elements in t are
    # equally spaced.

    h = x[1] - x[0]
    print(h)
    if method == "central":
        ai = (1 + (1/2) * h**2 * q)
        bi = (-1/2) * ((h/2) * p + 1)
        ci = (-1/2) * (1 - (h/2) * p)
        ri = (-1/2) * h**2 * r
    elif method == "backward":
        ai = -2/h**2 -p/h - q
        bi = 1/h**2 + p/h
        ci = np.array([1/h**2] * n)
        ri = r
    elif method == "foward":
        ai = (-2/h**2) + p/h - q
        bi = np.array([1/h**2] * n)
        ci = (1/h**2) - p/h
        ri = r
        print(bi)
    else:
        pass

    ri[1]  = ri[1] - (bi[1]*y_a)
    ri[-2] = ri[-2] - (ci[-2]*y_b)

    A = np.diag(ai[1:-1]) + np.diag(bi[2:-1], -1) + np.diag(ci[1:-2], 1)

    y = np.linalg.solve(A, ri[1:-1])
    y = np.concatenate(([y_a], y, [y_b]))
    return y