from enum import Enum 

class basisType(Enum):
    hermite = "hermite"
    laguerre = "laguerre"
    legendre = "legendre"
    chebyshev_1 = "chebyshev1"
    chebyshev_2 = "chebyshev2"
    fourier = "fourier"

class orthoType(Enum):
    ortho = "ortho"
    n_ortho = "n_ortho"
    proj = "proj"

