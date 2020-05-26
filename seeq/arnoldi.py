
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

    def apply(self, v, dt=1.0, order=None, adaptive=False, dtmin=None, tol=1e-12, warning=True):
        """Apply the Arnodi approximation of the exponential exp(-1i*dt*A)
        onto the vector or array `v`.        
        Parameters
        ----------
        v        -- A vector or a matrix
        order    -- Maximum number of Arnoldi vectors
        dt       -- time interval in the exponential above (can be complex)
        tol      -- relative tolerance for deciding when to stop the expansion
        adaptive -- Reduce time step if errors exceed tolerance
        dtmin    -- Minimum time interval allowed when adaptive = True
        warning  -- Emit warning when errors exceed tolerance
        """
        nmax = 12 if order is None else order
        if dtmin is None:
            dtmin = dt / 128
        #
        # While 'v' could be any tensor, we have to rewrite it in
        # matrix form because if self.H is a LinearOperator or a
        # sparse matrix, it only works with objects that have <= 2
        # indices
        v = numpy.asarray(v)
        shape = v.shape
        if v.ndim > 1:
            shape = (shape[0],v.size // shape[0])
        start = 0
        t = dt
        δt = dt
        while t > δt/2:
            if start == 0:
                β = numpy.linalg.norm(v)
                vn = v.flatten() / β
                V = []
                H = numpy.zeros((nmax+1,nmax+1),dtype=numpy.complex128)
            else:
                aux = np.zeros((nmax+1,nmax+1), dtype=H.dtype)
                aux[numpy.ix_(range(H.shape[0]),range(H.shape[1]))] = H
                H = aux
            for j in range(start,nmax):
                V.append(vn)
                w = (self.H @ vn.reshape(shape)).flatten()
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
            # Corrected Arnoldi method
            H[j+1,j+1] = 1.0
            e1 = numpy.zeros(j+2)
            e1[0] = 1.0
            y = scipy.sparse.linalg.expm_multiply((-1j*δt)*H, e1)
            err, y = abs(hlast * y[-1]), y[:-1]
            #
            if err <= tol:
                # Error is small, we keep approximation, advance time
                pass
            elif (order is not None) and adaptive:
                # Error is big, try reducing time step
                while err > tol and δt > dtmin*dt:
                    δt /= 2
                    y = scipy.sparse.linalg.expm_multiply((-1j*δt)*H, e1)
                    err, y = abs(hlast * y[-1]), y[:-1]
                adaptive = False
            elif (order is None) and (nmax < v.size):
                # Try mitigating error by increasing number of Arnoldi
                # vectors, if feasible.
                start = nmax
                nmax = min(int(1.5*nmax+1), v.size)
                continue
            else:
                # We cannot reduce time, and cannot enlarge the number
                # of vectors, emit a warning if we did not so before
                if warning:
                    warnings.warn(f'Arnoldi failed to converge at {j+1} '
                                  f'iterations with error {err}',
                                  AccuracyWarning)
                warning = False
            #
            # Compute the new vector and (possibly) start again at an
            # increased time
            v = sum(Vi * (β * yi) for Vi, yi in zip(V, y)).reshape(v.shape)
            start = 0
            t -= δt
        return v
    
def expm(A, v, **kwargs):
    """Apply the Arnoldi approximation of the exponential exp(-1i*dt*A)
    onto the vector or matrix `v`.
    
    Parameters
    ----------
    A         -- A matrix or a LinearOperator() with shape (d,d), or
                 a linear function that acts on vectors.
    v         -- A vector or a matrix with shapes (d,) or (d,M)
    d         -- Dimension of matrix A. Only required when A is
                 a function or callable object
    order     -- maximum order of the Arnoldi approximation
    dt        -- time interval in the exponential above (can be complex)
    tol       -- relative tolerance for deciding when to stop

    Returns
    -------
    newv      -- A vector or a matrix approximating expm(1j*dt*A) @ v
    """
    aux = ArnoldiExpm(A, d=v.size)
    return aux.apply(v, **kwargs)

