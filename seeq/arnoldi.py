
import numpy
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
import warnings

class AccuracyWarning(Warning):
    pass

class ArnoldiExpm:
    def __init__(self, H, d=0):
        #
        # H can be a function, a matrix, a sparse matrix, etc. We construct
        # a linear operator in all cases, which is a general abstraction that
        # numpy can work with and allows multiplication times a vector '@'
        #
        if callable(H) and not isinstance(H, sla.LinearOperator):
            H = sla.LinearOperator((d,d),matvec=H)
        self.H = H            
    
    def estimateOrder(self, dt=1.0):
        #
        # Estimate the order of the Lanczos expansion
        #
        self.Hnorm = abs(sla.eigs(self.H, k=1, which='LM', return_eigenvectors=0)[0])
        return max(int(3*self.Hnorm*dt+1),4)

    def apply(self, v, dt=1.0, order=None, tol=1e-13):
        # This function has two ways to operate: if we provide an order,
        # it applies the Lanczos expansion up to that basis size; otherwise
        # it estimates the number of vectors based on the norm of the matrix
        # that we will exponentiate (which was estimated before in __init__)
        nmax = 12 if order is None else order
        #
        # Construct a projected version of the matrix 'H' on the
        # Krylov subspace generated around vector 'v'
        #
        v = numpy.asarray(v)
        β = numpy.linalg.norm(v)
        vn = v / β
        V = []
        H = scipy.sparse.dok_matrix((v.size,v.size), dtype=numpy.complex128)
        lasterr = 0
        start = 0
        while True:
            for j in range(start,nmax):
                V.append(vn)
                w = (-1j*dt)*(self.H @ vn)
                for (i,Vi) in enumerate(V):
                    H[i,j] = hij = numpy.vdot(Vi, w)
                    w -= hij * Vi
                H[j+1,j] = hlast = numpy.linalg.norm(w)
                if hlast < 1e-16:
                    # Stop if our vectors become too small
                    break
                w /= hlast
                vn = w
            #
            # We diagonalize the banded matrix formed by α and β and
            # use that to compute the exponential of the effective
            # truncated matrix. This also allows us to estimate the error
            # due to the Lanczos method.
            #
            Hj = H.copy()
            if True:
                Hj.resize((j+2,j+2))
                Hj[j+1,j+1] = 1.0
                e1 = np.zeros(j+2)
                e1[0] = 1.0
                y = scipy.sparse.linalg.expm_multiply(Hj.tocsr(), e1)
                #y = scipy.linalg.expm(Hj.todense())[:,0]
                err = abs(hlast * y[-1])
                y = y[:-1]
            else:
                Hj.resize((j+1,j+1))
                e1 = np.zeros(j+1)
                e1[0] = 1.0
                y = scipy.sparse.linalg.expm_multiply(Hj.tocsr(), e1)
                #y = scipy.linalg.expm(Hj.todense())[:,0]
                err = abs(hlast * y[-1])
            if err < tol:
                break
            if (nmax == v.size) or (order is not None):
                warnings.warn(f'Arnoldi failed to converge at {j+1} iterations with error {err}',
                              AccuracyWarning)
                lasterr = err
                break
            start = nmax
            nmax = min(int(1.5*nmax+1), v.size)
        #
        # Given the approximation of the exponential, recompute the
        # Lanczos basis 
        #
        return np.array(V).T @ (β * y)
    
def expm(A, v, **kwargs):
    """Apply the Arnoldi approximation of the exponential exp(1i*dt*A)
    onto the vector or matrix `v`.
    
    Parameters
    ----------
    A         -- A matrix or a LinearOperator() with shape (d,d), or
                 a linear function that acts on vectors.
    v         -- A vector or a matrix with shapes (d,) or (d,M)
    d         -- Dimension of matrix A. Only required when A is
                 a function or callable object
    order     -- maximum order of the Arnoldi approximation
    dt        -- time interval in the exponential above
    tol       -- relative tolerance for deciding when to stop

    Returns
    -------
    newv      -- A vector or a matrix approximating expm(1j*dt*A) @ v
    """
    aux = ArnoldiExpm(A, d=v.size)
    return aux.apply(v, **kwargs)

