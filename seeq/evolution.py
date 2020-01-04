
import numpy as np
import scipy.sparse as sp
import scipy.integrate
import seeq.chebyshev
import seeq.lanczos

def evolve(ψ, Hamiltonian, times, method='chebyshev',
           observables=[], constant=False, **kwdargs):
    """
    Time evolution of a quantum state `ψ` under `Hamiltonian`.
    
    Arguments:
    ----------
    ψ           -- Either a complex vector for the initial state, or
                   a matrix with columns formed by different initial
                   states.
    Hamiltonian -- Hamiltonian for the Schrödinger equation. It may be
                   a `dxd` dense or sparse matrix or callable object
                   H(t, ψ) that returns the product of the Hamiltonian
                   times the vectors `ψ`.
    observables -- A list of `dxd` Hermitian matrices representing
                   observables.
    constant    -- True if Hamiltonian is a callable object but always
                   applies the same operator.
    method      -- One of 'eig', 'expm', 'expm_multiply', 'chebyshev',
                   'lanczos', 'zvode'

    Output:
    -------
    If `observables` is empty (default) this function  generates pairs
    of times and states (t, ψt). Otherwise it returns (t, Ot) where Ot
    is a list of expected values, commencing by the norm of the state.
    """
    times = np.array(times)
    if isinstance(Hamiltonian, (np.ndarray, np.matrix)) or sp.issparse(Hamiltonian):
        constant = True
    #
    # Time integrator.
    #
    d = len(ψ)
    lastt = times[0]
    gen = _evolve_dict.get((method, constant), None)
    if gen is None:
        raise ValueError(f'Unknown method "{method}" in evolve():')
    else:
        gen = gen(Hamiltonian, ψ, lastt, **kwdargs)
    for t in times:
        δt = t - lastt
        lastt = t
        if δt:
            ψ = gen(t, δt, ψ)
        if observables:
            yield t, np.array([np.sum(ψ.conj() * (op @ ψ), 0).real for op in observables])
        else:
            yield t, ψ

def to_matrix(H, t, d):
    if isinstance(H, (np.ndarray, np.matrix)):
        return H
    if sp.issparse(H):
        return H.todense()
    return H(t, np.eye(d))

def ct_eig_gen(H, ψ0, t0):
    # Constant Hamiltonian, exact diagonalization
    d = ψ0.shape[0]
    ω, V = scipy.linalg.eigh(to_matrix(H, 0.0, ψ0.shape[0]))
    Vt = V.conj().T
    if ψ0.ndim == 1:
        return lambda t, δt, ψ: V @ (np.exp((-1j*δt) * ω) * (Vt @ ψ))
    else:
        return lambda t, δt, ψ: V @ (np.exp((-1j*δt) * ω).reshape(d,1) * (Vt @ ψ))

def eig_gen(H, ψ0, t0):
    # Time-dependent Hamiltonian, exact diagonalization
    d = ψ0.shape[0]
    def step(t, δt, ψ):
        ω, V = scipy.linalg.eigh(to_matrix(H, t, d))
        if ψ.ndim == 2:
            ω = ω.reshape(d,1)
        return V @ (np.exp((-1j*δt)*ω) * (V.conj().T @ ψ))
    return step

def expm_gen(H, ψ0, t0):
    # Any Hamiltonian, exact diagonalization
    d = ψ0.shape[0]
    return lambda t, δt, ψ: scipy.linalg.expm((-1j*δt)*to_matrix(H, t, d)) @ ψ

def ct_chebyshev_gen(H, ψ0, t0, bandwidth=None, tol=1e-10, order=100):
    # Constant Hamiltonian, Chebyshev method
    d = ψ0.shape[0]
    U = seeq.chebyshev.ChebyshevExpm(H, d=d, bandwidth=bandwidth)
    return lambda t, δt, ψ: U.apply(ψ, dt=δt, tol=tol, order=order)

def chebyshev_gen(H, ψ0, t0, bandwidth=None, tol=1e-10, order=100):
    # Time-dependent Hamiltonian, Chebyshev method
    d = ψ0.shape[0]
    def step(t, δt, ψ):
        U = seeq.chebyshev.ChebyshevExpm(scipy.sparse.linalg.LinearOperator((d,d), matvec=lambda ψ: H(t, ψ)),
                                         bandwidth=bandwidth)
        return U.apply(ψ, dt=δt, tol=tol, order=order)
    return step

def ct_lanczos_gen(H, ψ0, t0, tol=1e-10, order=100):
    # Constant Hamiltonian, Chebyshev method
    U = seeq.lanczos.LanczosExpm(H, d=ψ0.shape[0])
    return lambda t, δt, ψ: U.apply(ψ, dt=δt, tol=tol, order=order)

def lanczos_gen(H, ψ0, t0, largestEigenvalue=0.0, tol=1e-10, order=100):
    # Time-dependent Hamiltonian, Chebyshev method
    d = ψ0.shape[0]
    def step(t, δt, ψ):
        U = seeq.lanczos.LanczosExpm(scipy.sparse.linalg.LinearOperator((d,d), matvec=lambda ψ: H(t, ψ)))
        return U.apply(ψ, dt=δt, tol=tol, order=order)
    return step

def expm_multiply_gen(H, ψ0, t0):
    # Time-dependent Hamiltonian, Scipy's method
    d = ψ0.shape[0]
    def step(t, δt, ψ):
        if callable(H):
            aux = lambda ψ: -1j * δt * H(t, ψ)
        else:
            aux = lambda ψ: -1j * δt * (H @ ψ)
        aux = scipy.sparse.linalg.LinearOperator((d,d), matvec=aux)
        return scipy.sparse.linalg.expm_multiply(aux, ψ)
    return step

def ct_expm_multiply_gen(H, ψ0, t0):
    # Constant Hamiltonian, Scipy's method
    if callable(H):
        aux = lambda ψ: -1j * δt * H(0, ψ)
    else:
        aux = lambda ψ: -1j * δt * (H @ ψ)
    # Time-dependent Hamiltonian, Chebyshev method
    return lambda t, δt, ψ: scipy.sparse.linalg.expm_multiply(aux, ψ)

def zvode_gen(H, ψ0, t0, rtol=1e-10, atol=1e-10):
    # Time-dependent Hamiltonian, Chebyshev method
    if callable(H):
        dydt = lambda t, ψ: -1j * H(t, ψ)
    else:
        dydt = lambda t, ψ: -1j * (H @ ψ)
    integrator = scipy.integrate.ode(dydt)
    integrator.set_integrator('zvode', method='adams', rtol=rtol, atol=atol)
    integrator.set_initial_value(ψ0, t0)
    return lambda t, δt, ψ: integrator.integrate(t)


_evolve_dict = {
    ('eig', False): eig_gen,
    ('expm', False): expm_gen,
    ('expm_multiply', False): expm_multiply_gen,
    ('zvode', False): zvode_gen,
    ('chebyshev', False): chebyshev_gen,
    ('lanczos', False): lanczos_gen,

    ('eig', True): ct_eig_gen,
    ('expm', True): expm_gen,
    ('expm_multiply', True): ct_expm_multiply_gen,
    ('zvode', True): zvode_gen,
    ('chebyshev', True): ct_chebyshev_gen,
    ('lanczos', True): ct_lanczos_gen,
}
