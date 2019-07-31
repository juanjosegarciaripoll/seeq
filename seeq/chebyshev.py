
import numpy as np
import scipy.sparse.linalg
from scipy.special import jn
import scipy.linalg as la
import scipy.sparse.linalg as sla

class ChebyshevExpm:
    """
    This object is created to compute $exp(-i*H*dt)*v$ using
    the Chebyshev series to approximate the exponential. The
    object does some preconditioning or scaling of the operator
    based on the range of eigenvalues and 'factor', transforming
    it to $exp(-i*A*R)*v$ where B has eigenvalues in [-1,1] and
    R is a real number.
    
    H     -- a matrix, or a linear function f(v) on vectors
    d     -- matrix size, only needs to be provided when H is a function
    order -- the order of the Chebyshev approximation
    dt    -- time interval in the exponential above
    rtol  -- relative tolerance for deciding when to stop the
             Chebyshev expansion
    """
    def __init__(self, H, d=None, largestEig=0):
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
        if largestEig:
            self.largestEig = largestEig
            self.smallestEig = -largestEig
        else:
            self.Hnorm = abs(sla.eigsh(self.H, k=1, which='LM',
                                       return_eigenvectors=0)[0])
            self.largestEig = self.Hnorm
            self.smallestEig = - self.Hnorm
    
    @staticmethod
    def weights(order, rm):
        ndx = np.arange(0,order)
        return jn(ndx, rm) * ((-1j)**ndx)
                    
    def apply(self, v, order=100, dt=1.0, tol=1e-14):
        
        rp = dt * (self.largestEig + self.smallestEig) / 2
        rm = dt * (self.largestEig - self.smallestEig) / 2
        order = max(order, 2 * int(rm))
        
        # Apply a version of A that is shifted and rescaled 
        def Btimes(phi):
            return ((self.H @ phi) * (dt/rm) - phi * (rp/rm))
    
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
            print("Desired precision of ", tol, "not reached.")
            print("Error = ", np.linalg.norm(tmp)/np.linalg.norm(cheb))

        return cheb * np.exp(-1j*rp)
    
def expm(A, v, **kwargs):
    aux = ChebyshevExpm(A)
    return aux.apply(v, **kwargs)
