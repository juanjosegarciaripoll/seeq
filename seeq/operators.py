import numpy as np
import scipy.sparse as sp

σx = sp.csr_matrix([[0.0,1.0],[1.0,0.0]])
σz = sp.csr_matrix([[1.0,0.0],[0.0,-1.0]])
σy = sp.csr_matrix([[0.0,-1.j],[1.j,0.0]])

def boson_creation(nmax, format='csr'):
    ad = np.sqrt(np.arange(1., nmax))
    if format == 'dense':
        return np.diag(ad, -1)
    else:
        return sp.diags(ad, -1, format=format)

def boson_anihilation(nmax, format='csr'):
    a = np.sqrt(np.arange(1., nmax))
    if format == 'dense':
        return np.diag(a, +1)
    else:
        return sp.diags(a, +1, format=format)

def boson_number(nmax, format='csr'):
    n = np.arange(0., nmax)
    if format == 'dense':
        return np.diag(0, +1)
    else:
        return sp.diags(n, 0, format=format)
