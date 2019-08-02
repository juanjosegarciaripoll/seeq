import numpy as np
from seeq.evolution import evolve
from numbers import Number
import scipy.optimize

def parametric_control(x0, H, ψ0, Ug, T, dH=None, check_gradient=False,
                       steps=100, tol=1e-10, method='chebyshev', target='fidelity',
                       debug=False, optimizer='BFGS', **kwdargs):
    """Solve the quantum control problem for a Hamiltonian H acting on
    a basis of states ψ.
    
    Arguments:
    ----------
    H      - Callable object H(t,x,ψ) that applies H(t,x) on ψ
    ψ0     - N x d object with N wavefunctions
    Ug     - Desired quantum operation
    T      - Either a time or a vector of times
    steps  - # time steps in the algorithm (if T is a number)
    tol    - Optimization tolerance
    target - 'fidelity' or 'overlap', depending on the cost function
    method - Solution method for the time evolution
    debug  - Return cost functions
    optimizer - Method for scipy.optimize.minimize
    
    Output:
    -------
    x      - Optimal control
    F      - Fidelity"""

    if isinstance(T, Number):
        times = np.linspace(0, T, steps)
    else:
        times = np.array(T)
        steps = len(T)
    if ψ0.ndim == 1:
        ψ0 = ψ0.reshape(ψ0.size, 1)
    N = ψ0.shape[1]
    ξT = Ug @ ψ0
    
    def compute_ψT(x):
        for t, ψ in evolve(ψ0, lambda t,ψ: H(t,x,ψ), times, method=method):
            ψT = ψ
        return ψT
    
    def compute_ψT_and_f(x):
        Hx = lambda t, ψ: H(t, x, ψ)
        ξ = list(ξt for t, ξt in evolve(ξT, Hx, times[-1::-1], method=method))
        ξ.reverse()
        f = np.zeros((len(x), N, len(times)))
        for (i, ((t, ψt), ξt)) in enumerate(zip(evolve(ψ0, Hx, times, method=method), ξ)):
            ψT = ψt
            f[:,:,i] = np.array([np.sum(ξt.conj()*dHi, 0) for dHi in dH(t, x, ψt)]).imag
        return ψT, f
    
    def fidelity(x):
        return -np.vdot(ξT, compute_ψT(x)).real/N

    def fidelity_and_gradient(x):
        ψT, f = compute_ψT_and_f(x)
        F = -np.vdot(ξT, ψT).real/N
        dFdx = -scipy.integrate.simps(np.sum(f, 1), x=times, axis=-1)/N
        return F, dFdx
    
    def overlap(x):
        Fn = np.sum(ξT.conj() * compute_ψT(x), 0)
        return -np.vdot(Fn, Fn).real/N

    def overlap_and_gradient(x, verbose=False):
        ψT, f = compute_ψT_and_f(x)
        dFndx = scipy.integrate.simps(f, x=times, axis=-1)
        Fn = np.sum(ξT.conj()*ψT, 0)
        O = -np.vdot(Fn, Fn).real/N
        dOdx = -2*np.sum(Fn.conj()*dFndx, 1).real/N
        return O, dOdx
    
    if target == 'fidelity':
        fn = fidelity
        dfn = fidelity_and_gradient
    else:
        fn = overlap
        dfn = overlap_and_gradient
    if dH is None:
        r = scipy.optimize.minimize(fn, x0, tol=tol, method=optimizer)
    else:
        if check_gradient:
            _, grad = dfn(x0)
            grad2 = scipy.optimize.approx_fprime(x0, fn, 1e-6)
            err = np.max(np.abs(grad2 - grad))
            print(f'max gradient error:   {err}')
            print(f'Finite diff gradient: {grad2}')
            print(f'Our estimate:         {grad}')
        r = scipy.optimize.minimize(dfn, x0, tol=tol, jac=True, method=optimizer)
    if debug:
        return r, fn
    else:
        return r
