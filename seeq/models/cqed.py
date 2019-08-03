import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from seeq.tools import lowest_eigenstates, lowest_eigenvalues

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
        self.Ec = Ec = Ec * np.ones(nqubits)
        self.ng = ng = ng * np.ones(nqubits)
        Id = sp.eye(fulldim)
        self.Hcap = sum((4.0*Ec) * (N-ng*Id)**2
                        for ng, Ec, N in zip(ng, self.Ec, self.N))
        #
        # Inductive energy
        self.EJ = EJ = EJ * np.ones(nqubits)
        self.HJJ = [EJ * qubit_operator((Sup+Sdo)/2., j, nqubits)
                     for j, EJ in enumerate(self.EJ)]
        #
        # The interaction must be symmetric
        g = g * np.ones((nqubits, nqubits))
        self.g = g = (g + g.T)/2.0
        self.Hint = sum((2.0 * g[i,j]) * (self.N[i] * self.N[j])
                         for i in range(self.nqubits)
                         for j in range(i)
                         if g[i,j])

    def _normalize_EJ(self, EJ):
        return self.EJ if EJ is None else EJ * np.ones(self.nqubits)

    def hamiltonian(self, EJ=None, gfactor=1.):
        """Return the Hamiltonian of this set of transmons, possibly
        changing the Josephson energies or rescaling the couplings.
        
        Arguments:
        ----------
        EJ      -- A scalar or a vector of Josephson energies, or
                   None if we use the default values.
        gfactor -- Multiplicative factor on the interaction term.
        """
        EJ = self._normalize_EJ(EJ)
        return sum((-EJ) * hi for EJ, hi in zip(EJ,self.HJJ)) + \
                self.Hcap + gfactor * self.Hint

    def apply(self, ψ, EJ=None, gfactor=1.):
        """Act with the Hamiltonian of this set of transmons, onto
        the state ψ. Arguments are the same as for hamiltonian().
        """
        EJ = self._normalize_EJ(EJ)
        out = self.Hcap @ ψ - sum(EJi * (hi @ ψ) for EJi, hi in zip(EJ,self.HJJ))
        if gfactor and self.Hint is not 0:
            out += gfactor * (self.Hint @ ψ)
        return out

    def _matvec(self, A):
        return self.apply(A)

    def qubit_basis(self, EJ=None, which=None):
        """Return the computational basis for the transmons in the limit
        of no coupling.
        
        Arguments:
        ----------
        which -- If None, return all 2**nqubits eigenstates. If it is
                 an index, return the eigenstates for the n-th qubit.
        EJ    -- Josephson energy (or None, for the default values)
        
        Returns:
        --------
        ψ     -- Matrix with columns for the computational basis states.
        """
        nqubits = self.nqubits
        EJ = self._normalize_EJ(EJ)
        if which is None:
            basis = 1
            for i in range(nqubits):
                basis = np.kron(basis, self.qubit_basis(ϵ, i))
        else:
            ti = Transmons(nqubits=1, Ec=self.Ec[i], nmax=self.nmax)
            _, basis = ti.eigenstates(2, EJ=self.EJ[i])
        return basis
