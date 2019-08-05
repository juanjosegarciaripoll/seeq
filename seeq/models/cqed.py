import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from seeq.tools import lowest_eigenstates, lowest_eigenvalues
import copy

class Transmons(LinearOperator):
    
    """Transmons() implements one or more coupled transmons. This class
    acts as a LinearOperator that implements the Hamiltonian. It can
    also produce copies of itself with adjusted parameters. If a parameter
    is a scalar, the same value is used for all qubits.
    
    Parameters
    ----------
    nqubits -- number of transmons
    Ec      -- capacitive energy (defaults to 1/95.)
    EJ      -- Josephson energy (defaults to 1.).
    g       -- couplings (scalar or matrix)
    ng      -- offset in number (defaults to 0)
    nmax    -- cutoff in charge space (defaults to 8)
    format  -- format of matrices (defaults to 'csr')
    """

    def __init__(self, nqubits, Ec=1/95., EJ=1., g=0, ng=0, nmax=8, format='csr'):
        self.nqubits = nqubits
        self.Ec = Ec = Ec * np.ones(nqubits)
        self.ng = ng = ng * np.ones(nqubits)
        self.EJ = EJ = EJ * np.ones(nqubits)
        assert len(Ec) == len(ng) == len(EJ) == nqubits
        # Dimensions of one-qubit problem
        dim = 2*nmax+1
        # Dimension of operators and states for the full problem
        fulldim = dim**nqubits
        #
        # This class inherits from LinearOperator because that implements
        # useful multiplication operators.
        super(Transmons,self).__init__(np.float64, (fulldim,fulldim))       
        #
        # Operators for one qubit
        self.nmax = nmax
        N = sp.diags(np.arange(-nmax, nmax + 1, 1), 0,
                     shape=(dim, dim), format=format)
        Sup = sp.diags([1.0], [1], shape=(dim,dim), format=format)
        Sdo = Sup.T
        #
        # Extend an operator to act on the whole Hilbert space
        def qubit_operator(op, j, N):
            d = op.shape[0]
            il = sp.eye(d**j, format=format)
            ir = sp.eye(d**(N-j-1), format=format)
            return sp.kron(il, sp.kron(op, ir))
        #
        # Local operators on all qubits:
        #
        self.N = [qubit_operator(N, j, nqubits) for j in range(nqubits)]
        self.nmax = nmax
        #
        # Capacitive energy
        Id = sp.eye(fulldim)
        self.Hcap = sum((4.0*Ec) * (N-ng*Id)**2
                        for ng, Ec, N in zip(ng, self.Ec, self.N))
        #
        # Inductive energy
        self.HJJ = [qubit_operator((Sup+Sdo)/2., j, nqubits)
                    for j, EJ in enumerate(self.EJ)]
        #
        # The interaction must be symmetric
        g = g * np.ones((nqubits, nqubits))
        self.g = (g + g.T)/2.0

    def hamiltonian(self):
        """Return the Hamiltonian of this set of transmons."""
        return self.Hcap + \
            sum((-EJ) * hi for EJ, hi in zip(self.EJ,self.HJJ)) + \
            sum((2*self.g[i,j]) * (self.N[i] * self.N[j])
                     for i in range(self.nqubits)
                     for j in range(i)
                     if self.g[i,j])
            
    def apply(self, ψ):
        """Act with the Hamiltonian of this set of transmons, onto
        the state ψ."""
        g = self.g
        N = self.N
        return self.Hcap @ ψ \
            - sum(EJi * (hi @ ψ) for EJi, hi in zip(self.EJ,self.HJJ)) \
            + sum((2*g[i,j]) * (N[i] @ (N[j] @ ψ))
                       for i in range(self.nqubits)
                       for j in range(i)
                       if g[i,j])

    def _matvec(self, A):
        return self.apply(A)

    def _matmat(self, A):
        return self.apply(A)

    def tune(self, EJ=None, g=None):
        """Return a new Transmon with tuned parameters."""
        out = copy.copy(self)
        if EJ is not None:
            out.EJ = EJ * np.ones(self.nqubits)
        if g is not None:
            g = g * np.ones((self.nqubits,self.nqubits))
            out.g = 0.5 * (g + g.T)
        return out

    def qubit_basis(self, which=None):
        """Return the computational basis for the transmons in the limit
        of no coupling.
        
        Arguments:
        ----------
        which -- If None, return all 2**nqubits eigenstates. If it is
                 an index, return the eigenstates for the n-th qubit.
        
        Returns:
        --------
        ψ     -- Matrix with columns for the computational basis states.
        """
        nqubits = self.nqubits
        if which is None:
            basis = 1
            for i in range(nqubits):
                basis = np.kron(basis, self.qubit_basis(i))
        else:
            ti = Transmons(nqubits=1, Ec=self.Ec[which],
                           EJ=self.EJ[which], nmax=self.nmax)
            _, basis = lowest_eigenstates(ti, 2)
        return basis
    
    def frequencies(self, n=1):
        """Return gaps between states 1, 2, ... n and the ground state"""
        λ = lowest_eigenvalues(self, neig=n+1)
        return tuple(λ[1:]-λ[0]) if n > 1 else λ[1]-λ[0]
