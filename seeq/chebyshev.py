
import numpy as np
import scipy.sparse.linalg
from scipy.special import jn
import scipy.linalg as la
import scipy.sparse.linalg as sla
import warnings

class AccuracyWarning(Warning):
    pass

class ChebyshevExpm:
    """
    This object is created to compute $exp(-i*H*dt)*v$ using
    the Chebyshev series to approximate the exponential. The
    object does some preconditioning or scaling of the operator
    based on the range of eigenvalues and 'factor', transforming
    it to $exp(-i*A*R)*v$ where B has eigenvalues in [-1,1] and
    R is a real number.
    
    Parameters
    ----------
    H         -- A matrix or a LinearOperator() with shape (d,d)
    d         -- matrix size, only needs to be provided when H is a function
    bandwidth -- An upper bound on the spectral bandwidth of A
    """
    def __init__(self, H, d=None, bandwidth=None):
        #
        # H can be a function, a matrix, a sparse matrix, etc. We construct
        # a linear operator in all cases, which is a general abstraction that
        # numpy can work with and allows multiplication times a vector '@'
        #
        # The method demands that the eigenvalues be in the range [-1,1]
        # We need analyze the operator to estimate a range of eigenvalues
        # and rescale the operator "A*factor" to a smaller range.
        #
        if callable(H) and not isinstance(H, sla.LinearOperator):
            H = scipy.sparse.linalg.LinearOperator((d,d),matvec=H)
        self.H = H
        #
        # Estimate the spectral range of H, computing the smallest and
        # largest eigenvalues. ARPACK in standard Python is not good at
        # computing small values. We thus use a trick of computing the
        # largest magnitude X and assume the spectrum is in [-X,X]
        #
        if bandwidth is None:
            λmax = sla.eigs(H, k=1, which='LM', return_eigenvectors=0)[0]
            Hnorm = abs(λmax)
            bandwidth = 2*Hnorm
        if np.isscalar(bandwidth):
            self.height = bandwidth/2.0
            self.center = 0.0
        else:
            Emin, Emax = bandwidth
            self.height = 0.5 * (Emax - Emin)
            self.center = 0.5 * (Emax + Emin)

    @staticmethod
    def weights(order, rm):
        ndx = np.arange(0,order)
        return jn(ndx, rm) * ((-1j)**ndx)
                    
    def apply(self, v, dt=1.0, order=None, maxorder=None, tol=1e-14):
        """Apply the Chebyshev approximation of the exponential exp(-1i*dt*A)
        onto the vector or matrix `v`.        
        Parameters
        ----------
        v     -- A vector or a matrix
        order -- the order of the Chebyshev approximation
        dt    -- time interval in the exponential above
        tol   -- relative tolerance for deciding when to stop the
                 Chebyshev expansion
        """
        rp = dt * self.center
        rm = dt * self.height
        if order is None:
            order = max(100, 2*int(rm))
        
        # Apply a version of A that is shifted and rescaled 
        def Btimes(phi):
            #
            # There's something stupid about LinearOperators that always return
            # matrices when applied to arrays. This screws our use of vdot()
            # and other operations below
            return np.asarray((self.H @ phi) * (dt/rm) - phi * (rp/rm))
    
        # Bessel coefficients ak, plus imaginary parts from chebyshev polynomials    
        ak = self.weights(order, rm)
    
        # Zeroth and first order
        phi0 = v
        phi1 = Btimes(phi0)
        cheb = ak[0] * phi0 + 2 * ak[1] * phi1
    
        # We can define an absolute tolerance if we assume unitary evolution
        # and always a nonzero relative tolerance 'tol'
        # Note the 'vector' 'v' is actually a matrix of vectors with columns
        # corresponding to different states. We want to compute the total
        # norm of the vectors in a speedy way.
        atol2 = np.abs(tol**2 * np.vdot(v, v))
        
        # Higher orders
        for jj in range(2,order):
            phi2 = 2 * Btimes(phi1) - phi0
            tmp = 2 * ak[jj] * phi2
            cheb += tmp
            if abs(np.vdot(tmp,tmp)) < atol2:
                break
            phi0 = phi1
            phi1 = phi2
        else:
            warnings.warn(f'Desired precision {tol} not reached with {order} steps. '
                          f'Error is {np.linalg.norm(tmp)/np.linalg.norm(cheb)}',
                          AccuracyWarning)

        return cheb * np.exp(-1j*rp)
    
def expm(A, v, d=None, bandwidth=None, **kwargs):
    """Apply the Chebyshev approximation of the exponential exp(1i*dt*A)
    onto the vector or matrix `v`.
    
    Parameters
    ----------
    A         -- A matrix or a LinearOperator() with shape (d,d), or
                 a linear function that acts on vectors.
    v         -- A vector or a matrix with shapes (d,) or (d,M)
    d         -- Dimension of matrix A. Only required when A is
                 a function or callable object
    bandwidth -- An upper bound on the spectral bandwidth of A or
                 a pair (Emin, Emax) of extreme eigenvalues
    order     -- the order of the Chebyshev approximation
    dt        -- time interval in the exponential above
    tol       -- relative tolerance for deciding when to stop the
                 Chebyshev expansion

    Returns
    -------
    newv      -- A vector or a matrix approximating expm(-1j*dt*A) @ v
    """
    aux = ChebyshevExpm(A, d=d, bandwidth=bandwidth)
    return aux.apply(v, **kwargs)
