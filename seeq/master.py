import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import scipy.integrate

class Lindblad(LinearOperator):
    
    def __init__(self, H, dissipators=[], time=0.0, dimension=None):
        if dimension is None:
            if callable(H):
                raise Exception('In Lindblad, you must provide a dimension of the Hilbert space')
            dimension = H.shape[0]
        super(Lindblad, self).__init__(np.complex128, (dimension**2,dimension**2))
        self.Hamiltonian = H
        self.dissipators = dissipators
        self.dimension = dimension
        self.ρshape = (dimension,dimension)
        self.time = 0.0

    def apply(self, t, ρ):
        ρ = np.asarray(ρ)
        flat = ρ.ndim == 1
        if flat:
            ρ = ρ.reshape(self.ρshape)
        H = self.Hamiltonian
        if callable(H):
            Lρ = -1j * (H(t, ρ - H(t, ρ.conj().T).conj()))
        else:
            Lρ = -1j * (H @ ρ - ρ @ H)
        for (γi, Ai, Bi) in self.dissipators:
            Lρ += γi * (Ai @ ρ @ Bi - 0.5 * ((ρ @ Bi) @ Ai + Bi @ (Ai @ ρ)))
        return Lρ.flatten() if flat else Lρ

    def _matvec(self, ρ):
        return self.apply(self.time, ρ)

def stationary_state(L, guess=None, method='eigs', tol=10**(-8), maxiter=1000):
    #
    # Compute the stationary state of a master equation using a variety
    # of methods:
    #
    #  - L : a Lindblad operator class that implements a method
    #    L.Lindblad() that returns the Linbdlad superoperator in
    #    matrix representation.
    #
    #  - guess : a guess for the density matrix. It may be either
    #    a vector, representing a pure state, or a density matrix.
    #
    #  - method : which method use to solve the equation
    #      SOLVE_EIGS = compute the zero-decay eigenstate of the
    #                   Lindblad operator using Arpack
    #
    #  - observables: return a list of expected values over the
    #    computed density matrix
    #
    d = L.dimension
    
    if guess is not None:
        if guess.size == d:
            # This is a pure state, make a density matrix
            guess = np.reshape(guess, (d,1))
            guess = guess @ guess.T.conjugate()
            guess /= np.trace(guess)
        guess = np.reshape(guess, (d*d,))

    def replace(vρ):
        #
        # This internal function creates a linear system of
        # equations that consists of 'd*d-1' rows corresponding
        # to the lindblad operator plus one row corresponding
        # to the trace of the density matrix. We have to solve
        #        A * vρ = rhs
        # where 'rhs' is zeros except for the row corresponding
        # to the trace, which is '1'.
        ρ = vρ.reshape(d,d)
        Lvρ = (L @ ρ).flatten()
        Lvρ[-1] = np.trace(ρ)
        return Lvρ

    if method == 'eigs':
        #
        # Compute one (k=1) eigenstate of the Lindblad operator which is
        # closest to the eigenvalue sigma=1 Since all eigenvalues of the
        # Lindblad operator have zero or negative real part, the closest
        # one is exactly the zero eigenstate.
        #
        value, vρ = sp.linalg.eigs(L, k=1, maxiter=maxiter, tol=tol,
                                   sigma=1, v0=guess)
        vρ = vρ.flatten()
    elif method == 'replace':
        vρ, info = sp.linalg.bicgstab(LinearOperator(L.ρshape, matvec=replace), rhs)
        if info > 0:
            raise Exception('Problem did not converge')
    else:
        raise Exception(f'Unknown method "{method}" in master.stationary_state()')
    #
    # Normalize the density matrix. Since the system of equations is linear,
    # any multiple of the solution is also a valid solution.
    ρ = np.reshape(vρ, (d,d))
    ρ /= np.trace(ρ)
    #
    # Even if we normalized, there may be some residual error and the
    # matrix might not be completely Hermitian. We check both
    Lvρ = L @ vρ
    λ = np.vdot(vρ, Lvρ) / np.vdot(vρ, vρ)
    ε = np.sum(np.abs(ρ - ρ.T.conjugate()))
    return ρ, [λ, ε]
