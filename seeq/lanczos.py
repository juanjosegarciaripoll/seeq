
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

    def apply(self, v, dt=1.0, order=None, tol=1e-14):
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
        H = scipy.sparse.dok_matrix((v.size,v.size))
        lasterr = 1.0
        start = 0
        while True:
            for j in range(start,nmax):
                V.append(vn)
                w = numpy.asarray(self.H @ vn)
                for i in range(j):
                    H[i,j] = hij = numpy.vdot(w, V[i])
                    w -= hij * V[i]
                H[j+1,j] = hlast = numpy.linalg.norm(w)
                if hlast < 1e-20:
                    break
                vn = w / hlast
            #
            # We diagonalize the banded matrix formed by α and β and
            # use that to compute the exponential of the effective
            # truncated matrix. This also allows us to estimate the error
            # due to the Lanczos method.
            #
            Hj = H.copy()
            Hj.resize((j+1,j+1))
            e1 = np.zeros(j+1)
            e1[0] = 1.0
            y = scipy.sparse.linalg.expm_multiply((-1j*dt)*Hj.tocsr(), e1)
            err = abs(hlast * y[-1])
            if err < tol:
                break
            if lasterr < err or nmax == v.size or (order is not None):
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
    aux = LanczosExpm(A, d=v.size)
    return aux.apply(v, **kwargs)


import numpy
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
import warnings

class AccuracyWarning(Warning):
    pass

class LanczosExpm:
    def __init__(self, H, d=0):
        #
        # H can be a function, a matrix, a sparse matrix, etc. We construct
        # a linear operator in all cases, which is a general abstraction that
        # numpy can work with and allows multiplication times a vector '@'
        #
        if callable(H) and not isinstance(H, sla.LinearOperator):
            H = sla.LinearOperator((d,d),matvec=H)
        self.H = H

    def apply(self, v, dt=1.0, order=None, tol=1e-14):
        # This function has two ways to operate: if we provide an order,
        # it applies the Lanczos expansion up to that basis size; otherwise
        # it estimates the number of vectors based on the norm of the matrix
        # that we will exponentiate (which was estimated before in __init__)
        nmax = 10 if order is None else order
        if nmax > v.size:
            nmax = v.size
        #
        # Construct a projected version of the matrix 'H' on the
        # Krylov subspace generated around vector 'v'
        #
        v = numpy.array(v)
        vnrm = la.norm(v)
        vn = v / vnrm
        vnm1 = numpy.zeros(v.shape)
        α = []
        β = [0.0]
        start = 0
        lasterr = vnrm * 1e10
        while True:
            #
            # Iteratively extend the Krylov basis using the lanczos
            # recursion without restart or reorthogonalization.
            #
            for n in range(start, nmax):
                w = numpy.asarray(self.H @ vn)
                α.append(numpy.vdot(vn, w))
                w = w - α[n] * vn - β[n] * vnm1
                vnm1 = vn
                aux = la.norm(w)
                β.append(aux)
                if aux < 1e-20:
                    break
                vn = w / aux
            #
            # We diagonalize the banded matrix formed by α and β and
            # use that to compute the exponential of the effective
            # truncated matrix. This also allows us to estimate the error
            # due to the Lanczos method.
            #
            w, u = scipy.linalg.eig_banded(numpy.array([β[:-1],α]),
                                           overwrite_a_band=True)
            fHt = u @ (numpy.exp(-1j*dt*w) * u[0,:].conj())
            err = abs(fHt[n]*β[n+1])
            if err < tol:
                break
            if lasterr < err or nmax == v.size or (order is not None):
                warnings.warn(f'Lanczos failed to converge at {len(α)} iterations with error {err}',
                              AccuracyWarning)
                lasterr = err
                break
            start = nmax
            nmax = min(int(1.5*nmax+1), v.size)
        #
        # Given the approximation of the exponential, recompute the
        # Lanczos basis 
        #
        vnm1 = vn = v
        output = fHt[0] * vn
        for n in range(1, nmax):
            w = numpy.asarray(self.H @ vn) - α[n-1] * vn - β[n-1] * vnm1
            vnm1 = vn
            vn = w / β[n]
            output = output + fHt[n] * vn
        return output
    
def expm(A, v, **kwargs):
    aux = LanczosExpm(A, d=v.size)
    return aux.apply(v, **kwargs)

