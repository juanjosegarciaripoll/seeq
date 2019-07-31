import numpy as np
from seeq.evolution import evolve
from numbers import Number
import scipy.optimize

def parametric_control(x0, H, ψ0, Ug, T, dH=None, check_gradient=False,
                       steps=100, tol=1e-10, method='chebyshev',
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
    ξT = Ug @ ψ0
    
    def bare_cost(x, verbose=False):
        for t, ψ in evolve(ψ0, lambda t,ψ: H(t,x,ψ), times, method=method):
            ψT = ψ
        return -np.vdot(ξT, ψT).real

    def cost_and_gradient(x, verbose=False):
        Hx = lambda t, ψ: H(t, x, ψ)
        ξ = list(ξt for t, ξt in evolve(ξT, Hx, times[-1::-1], method=method))
        ξ.reverse()
        f = np.zeros((len(x), len(times)))
        for (i, ((t, ψt), ξt)) in enumerate(zip(evolve(ψ0, Hx, times, method=method), ξ)):
            ψT = ψt
            f[:,i] = np.array([np.vdot(ξt, dHi) for dHi in dH(t, x, ψt)]).imag
        dFdx = np.array([-scipy.integrate.simps(fx, x=times) for fx in f])
        return -np.vdot(ξT, ψT).real, dFdx
    
    if dH is None:
        r = scipy.optimize.minimize(bare_cost, x0, tol=tol, method=optimizer)
        fn = bare_cost
    else:
        if check_gradient:
            _, dFdx = cost_and_gradient(x0)
            dFdx2 = scipy.optimize.approx_fprime(x0, bare_cost, 1e-6)
            err = np.max(np.abs(dFdx2 - dFdx))
            print(f'max gradient error:   {err}')
            print(f'Finite diff gradient: {dFdx2}')
            print(f'Our estimate:         {dFdx}')
        r = scipy.optimize.minimize(cost_and_gradient, x0, tol=tol, jac=True, method=optimizer)
        fn = cost_and_gradient
    if debug:
        return r, fn
    else:
        return r
