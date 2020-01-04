import numpy as np
from seeq.evolution import evolve
from numbers import Number
import scipy.optimize

def parametric_control(x0, H, ψ0, T, Ug=None, ξT=None, dH=None, check_gradient=False,
                       steps=100, tol=1e-10, method='chebyshev', target='fidelity',
                       debug=False, optimizer='BFGS', dispiter=0, **kwdargs):
    """Solve the quantum control problem for a Hamiltonian H acting on
    a basis of states ψ.
    
    Arguments:
    ----------
    H      - Callable object H(t,x,ψ) that applies H(t,x) on ψ
    ψ0     - N x d object with N wavefunctions
    Ug, ξT - Desired quantum operation or desired quantum states
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
    if ξT is None:
        if Ug is None:
            raise Exception('Must provide either Ug or ξT')
        ξT = Ug @ ψ0
    
    def compute_ψT(x):
        for _, ψT in evolve(ψ0, lambda t,ψ: H(t,x,ψ), times, method=method):
            pass
        return renormalize(ψT)
    
    def renormalize(ψ):
        # Remove errors in norm due to numerical dissipation
        return ψ / np.sqrt(np.sum(ψ.conj() * ψ, 0).real)

    def compute_Fn_dFndx(x):
        Hx = lambda t, ψ: H(t, x, ψ)
        ξ = evolve(ξT, Hx, times[-1::-1], method=method)
        ψ = evolve(ψ0, Hx, times, method=method)
        f = np.zeros((len(times), len(x), N), dtype=np.complex128)
        for (i, ((t, ψt), (t, ξt))) in enumerate(zip(ψ, reversed(list(ξ)))):
            ξt = renormalize(ξt)
            ψt = renormalize(ψt)
            f[i,:,:] = np.sum(ξt.conj() * dH(t, x, ψt), 1)
        Fn = np.sum(ξT.conj() * ψt, 0)
        dFndx = -1j * scipy.integrate.simps(f, x=times, axis=0)
        return Fn, dFndx
    
    global iterations
    iterations = (0,0,0)

    def maybe_display(F,x,gradF=None):
        global iterations
        if dispiter and iterations[0] % dispiter == 0:
            dF = F - iterations[1] if iterations[0] else 0
            dx = np.linalg.norm(x - iterations[2]) if iterations[0] else 0
            gradF = np.linalg(gradF) if gradF is not None else 0
            print(f'evaluation # {iterations[0]:6} - error={F:+10.4e} - variation={dF:+10.4e} - |dx|={dx:10.4e} - |grad|={gradF:10.4e}')
        iterations = (iterations[0]+1, F, x)
        return F

    def fidelity(x):
        F = np.vdot(ξT, compute_ψT(x)).real/N
        return maybe_display(1.0-F,x)

    def fidelity_and_gradient(x):
        Fn, dFndx = compute_Fn_dFndx(x)
        F = np.sum(Fn).real/N
        dFdx = np.sum(dFndx, axis=-1).real/N
        return maybe_display(1.0-F,x), -dFdx
    
    def overlap(x):
        Fn = np.sum(ξT.conj() * compute_ψT(x), 0)
        return maybe_display(1.0-np.vdot(Fn, Fn).real/N,x)

    def overlap_and_gradient(x, verbose=False):
        Fn, dFndx = compute_Fn_dFndx(x)
        O = np.vdot(Fn, Fn).real/N
        dOdx = 2*np.sum(Fn.conj() * dFndx, axis=-1).real/N
        return maybe_display(1.0-O,x), -dOdx
    
    if target == 'fidelity':
        fn = fidelity
        dfn = fidelity_and_gradient
    else:
        fn = overlap
        dfn = overlap_and_gradient
    if dH is None:
        r = scipy.optimize.minimize(fn, x0, tol=tol, method=optimizer, **kwdargs)
    else:
        if check_gradient:
            _, grad = dfn(x0)
            grad2 = scipy.optimize.approx_fprime(x0, fn, 1e-6)
            err = np.max(np.abs(grad2 - grad))
            print(f'max gradient error:   {err}')
            print(f'Finite diff gradient: {grad2}')
            print(f'Our estimate:         {grad}')
        r = scipy.optimize.minimize(dfn, x0, tol=tol, jac=True, method=optimizer, **kwdargs)
    if debug:
        return r, fn
    else:
        return r
