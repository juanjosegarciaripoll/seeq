import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from seeq.models.lattice import Lattice, Regular3DLattice

def SilbeyHarris(H, g, Δ=1.0):
    """Compute the effective couplings and renormalized frequency
    of the Silbey-Harris transformation, using the self-consistent
    equations.
    
    Parameters
    ----------
    H  -- Hamiltonian of the bosons (matrix or LinearOperator)
    g  -- Vector of couplings between the emitter and the bosons
    Δ  -- Gap of the quantum emitter (defaults to 1)
    
    Returns
    -------
    f  -- Vector of displacements
    Δr -- Renormalized gap for the emitter
    """
    Δr = Δ
    f = g
    if np.sum(np.abs(g)) > 1e-15:
        #
        # The Silbey-Harris procedure only works when the Hamiltonian
        # of the bosons has a nonzero gap.
        λ = sp.linalg.eigsh(H, k=1, which='SA', return_eigenvectors=False)
        if np.min(λ) <= 0:
            raise Exception(f'Eigenvalue negative or zero {λ}, singular lattice.')
        #
        # Solve the coupled equations iteratively.
        Id = sp.eye(env.size)
        for i in range(20):
            newf = sp.linalg.spsolve(env.H + Δr * Id, g)
            newΔr = Δ * np.exp(-2*np.vdot(newf,newf).real)
            if f.size > 1:
                err = 1.0 - np.vdot(newf,f)/(np.linalg.norm(f)*np.linalg.norm(newf))
            else:
                err = np.linalg.norm(newf-f)/np.linalg.norm(f)
            f = newf
            Δr = newΔr
            if abs(err) < 1e-12:
                break
    return f, Δr
from seeq.evolution import evolve

class SinglePhotonHamiltonian(LinearOperator):
    """Class for the single-photon limit of the Polaron or RWA
    Hamiltonians of the spin-boson model.
    
    Parameters
    ----------
    lattice  -- an object of Lattice class representing the bosonic environment
    r        -- position of the quantum emitter in `lattice`
    g        -- vector of couplings between the emitter and the environment or
                a scalar to multiply the output of `lattice.coupling_at(r)`.
    Δ        -- energy gap of the bare emitter (defaults to 1.)
    polaron  -- True if you wish to use the polaron approximation.
    Lanczos  -- True if you wish to simplify the lattice using the Lanczos method
    nmodes   -- number of modes to keep in the Lanczos method
    """

    def __init__(self, lattice, g=1., r=None, Δ=1.0, polaron=False,
                 Lanczos=False, nmodes=None):
        #
        # `N` is the number of photon modes. The Hilbert space of
        # photons include 1 photon state per mode plus one state for
        # the excited qubit, hence dimension is N+1
        N = lattice.size
        L = N+1
        super(SinglePhotonHamiltonian, self).__init__(np.complex128, (L,L))
        self.N = N
        self.size = L

        if np.isscalar(g) and r is None:
            raise Exception('You need to provide the position of the emitter.')
        self.g = g = g * lattice.coupling_at(r)
        self.Δ = Δ
        #
        # When using the Lanczos optimization, the vector of couplings
        # changes, because it is expressed in a new basis.
        self.Lanczos = Lanczos
        if Lanczos:
            self.old_lattice = lattice
            lattice = lattice = LanczosEnv(lattice, g, nmodes=nmodes)
            g = lattice.g
        self.lattice = lattice
        #
        # When using the polaron approximation, we compute the displacements
        # the renormalized frequency and the effective couplings
        self.polaron = polaron
        if polaron:
            f, Δr = SilbeyHarris(lattice, Δ=Δ)
            self.f = f
        #
        # The qubit operator is excited only in the state '0'
        sz = sp.diags([[1.0]+[-1.0]*N], offsets=[0], format='coo')
        #
        # Boson Hamiltonian
        Hw = sp.bmat([[0.0, sp.coo_matrix((1,N))],
                      [sp.coo_matrix((N,1)), lattice.hamiltonian()]])
        #
        # Number of excitations in photon space
        nphotons = sp.diags([[0.0]+[1.0]*N], offsets=[0], format='coo')
        #
        # Given a vector `g` construct `\sum_k g_k \sigma^+ a_k`
        spB = sp.bmat([[0.0, g.conj()],
                       [sp.coo_matrix((N,1)), sp.coo_matrix((N,N))]])
        smBd = spB.T.conj()
        #
        # These are the differences between the pure polaron and
        # the RWA model. The polaron model involves a dense matrix
        # B^\dagger * B, which is why we code it as a LinearOperator
        # that does not need to build the matrix
        if polaron:
            Hint = 2.0 * DeltaR * (spB + smBd)
            polSz = lambda ψ: (Δr/Delta)* (sz @ (ψ - 4.0 * (smBd @ (spB @ ψ))))
            applyH = lambda ψ: Hw @ ψ + Hint @ ψ + 0.5 * Δ * polSz(ψ)
            self._applyH = _applyH
            self._H = LinearOperator(matvec=applyH, shape=(L,L))
            self.lab_sz = sp.linalg.LinearOperator(matvec=polSz, shape=(L,L))
        else:
            H = Hw + (spB + smBd) + 0.5 * Δ * sz
            self._applyH = lambda ψ: H @ ψ
            self.lab_sz = sz
        self.sz = sz
        self.nphotons = nphotons

    def Hamiltonian(self):
        return self._H
    
    def _matvec(self, ψ):
        return self._applyH(ψ)
    
    def _matmat(self, A):
        return self._applyH(A)

    def one_photon(self, ϕ):
        """Return a state with a photon wavepacket.
        
        Parameters
        ----------
        ϕ  -- Photon wavepacket. Complex vector of size N, with
              the amplitudes of the state on each lattice site."""
        return np.concatenate([[0.], ϕ / np.linalg.norm(ϕ)])

    def excited(self):
        """Return a state with an excited qubit."""
        return np.concatenate([[1.], np.zeros(self.N,)])

    def photon_part(self, ψ):
        if self.Lanczos:
            return self.lattice.basis @ ψ[1:]
        else:
            return ψ[1:]

    def emission(self, T, steps=20, collect_states=False, **kwdargs):
        """Simulate the spontaneous emission of the excited impurity on
        a background of no photons.
        
        Parameters
        ----------
        T, steps       -- Similar semantics to seeq.evolution.evolve()
        collect_states -- True if we collect all states
        **kwdargs      -- All other arguments accepted by evolve()
        
        Returns
        -------
        t              -- Vector of times
        P1             -- Excited population of the qubit
        Pphoton        -- Density of photons
        ψ              -- Matrix of all states (if collect_states) or
                          final state otherwise"""
        if np.isscalar(T):
            T = np.linspace(0, T, steps)
        times = []
        out_ψ = []
        P1 = []
        Pphoton = []
        for (t, ψt) in evolve(self.excited(), self, T, constant=True, **kwdargs):
            if collect_states:
                out_ψ.append(ψt)
            else:
                out_ψ = ψt
            times.append(t)
            P1.append(np.abs(ψt[0])**2)
            Pphoton.append(np.abs(ψt[1:])**2)
        return np.array(times), np.array(P1), np.array(Pphoton), np.array(out_ψ)
