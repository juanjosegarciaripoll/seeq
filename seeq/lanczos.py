
import numpy
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla

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
        #
        # Estimate the spectral range of H, computing the smallest and
        # largest eigenvalues. ARPACK in standard Python is not good at
        # computing small values. We thus use a trick of computing the
        # largest magnitude X and assume the spectrum is in [-X,X]
        #
        self.Hnorm = abs(sla.eigs(H, k=1, which='LM', return_eigenvectors=0)[0])
    
    def estimateOrder(self, dt=1.0):
        return max(int(3*self.Hnorm*dt+1),4)

    def apply(self, v, dt=1.0, order=0, tol=1e-14):
        # This function has two ways to operate: if we provide an order,
        # it applies the Lanczos expansion up to that basis size; otherwise
        # it estimates the number of vectors based on the norm of the matrix
        # that we will exponentiate (which was estimated before in __init__)
        if order == 0:
            nmax = self.estimateOrder(dt)
        else:
            nmax = order
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
        lasterr = 1.0
        while True:
            #
            # Iteratively extend the Krylov basis using the lanczos
            # recursion without restart or reorthogonalization.
            #
            for n in range(start, nmax):
                w = (self.H @ vn)
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
            if lasterr < err or nmax == v.size:
                print('Lanczos failed to converge at ', len(α), ' iterations')
                print('err='+str(err))
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
            w = (self.H @ vn) - α[n-1] * vn - β[n-1] * vnm1
            vnm1 = vn
            vn = w / β[n]
            output = output + fHt[n] * vn
        return output
    
def expm(A, v, **kwargs):
    aux = LanczosExpm(A, d=v.size)
    return aux.apply(v, **kwargs)

