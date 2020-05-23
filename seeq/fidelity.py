import numpy as np
import scipy.linalg

def state_fidelity(σ1, σ2, normalize=False):
    """Compute the fidelity between states σ1 and σ1, which may be either
    vectors (pure states) or density matrices. Neither the states nor the
    density matrices need be normalized."""
    if σ1.ndim == 2:
        if normalize:
            σ1 /= np.trace(σ1)
        if σ2.ndim == 1:
            #
            # σ1 is a matrix, σ2 is a pure state
            if normalize:
                σ2 = σ2 / np.linalg.norm(σ2)
            return abs(np.vdot(σ2, σ1 @ σ2))
        elif σ2.ndim == 2:
            if normalize:
                σ2 /= np.trace(σ2)
            #
            # Both σ1 and σ2 are density matrices
            #
            λ1, U1 = scipy.linalg.eigh(σ1, overwrite_a=True)
            sqrtσ1 = (U1 * np.sqrt(np.abs(λ1))) @ U1.T.conj()
            λ, U = scipy.linalg.eigh(sqrtσ1 @ σ2 @ sqrtσ1, overwrite_a=True)
            return np.sum(np.sqrt(np.abs(λ)))**2
    elif σ2.ndim == 1:
        #
        # Both are pure states
        F = abs(np.vdot(σ1, σ2))**2
        if normalize:
            return F / (np.linalg.norm(σ1)*np.linalg.norm(σ2))
        else:
            return F
    elif σ2.ndim == 2:
        #
        # σ1 is a pure state, σ2 a density 
        if normalize:
            σ2 /= np.trace(σ2)
            σ1 = σ1 / np.linalg.norm(σ1)
        return abs(np.vdot(σ1, σ2 @ σ1))
    raise ValueException(f'state_fidelity() got neither a pure state nor a density matrix')

def avg_unitary_fidelity(U, W=None):
    """How close U is to W (which defaults to identiy)"""
    if W is not None:
        U = U * W.T.conj()
    d = len(U)
    Fe = np.abs(np.trace(U)/d)**2
    F = (d*Fe+1)/(d+1)
    return F

def avg_superoperator_fidelity(E):
    """Return the average fidelity of superoperator E, represented as a four
    dimensional tensor with indices of size d, where 'd' is the size of the
    Hilbert space."""
    if E.ndim == 4:
        d = E.shape[0]
        E = E.reshape(d*d,d*d)
    else:
        raise ValueException('Not a valid representation for a superoperator.')
    Fe = abs(np.trace(E))/(d*d)
    F = (d*Fe+1)/(d+1)
    return F

def avg_fidelity(T):
    """Return the average fidelity of a transformation T.
    
    Arguments
    ---------
    T  -- Either a 4 dimensional tensor, representing a positive map, or
          a 2 dimensional tensor or matrix, representing a unitary operation.

    Output
    ------
    F  -- A value in [0,1] representing the average fidelity.
    """
    if T.ndim == 2:
        return avg_unitary_fidelity(T)
    elif T.ndim == 4:
        return avg_superoperator_fidelity(T)
    else:
        raise ValueException('Not a valid superoperator or matrix.')

def leakage(S):
    """Compute the leakage outside the computational space, for a matrix
    S that connects input and output states in the computational basis,
    and which is in general not unitary."""
    if S.ndim == 2:
        d = S.shape[0]
        return np.abs(1 - np.vdot(S, S)/d)
    elif S.ndim == 4:
        d = S.shape[0]
        return np.abs(1 - np.einsum('iijj', S)/d)
    else:
        raise ValueError('Not a valid unitary or positive map')

def extract_phases(T):
    """Extract diagonal phases from a unitary operation or superoperator"""
    if T.ndim == 2:
        # Unitary operator
        v = np.diag(T) / T[0,0]
        v /= np.abs(v)
        return np.diag(v)
    elif T.ndim == 4:
        # Superoperator
        return extract_phases(T[:,0,:,0])
    raise ValueError('Not a valid unitary or positive map.')

def remove_phases(T):
    """Eliminate diagonal phases from a unitary operation or superoperator"""
    inv = extract_phases(T).conj()
    if T.ndim == 4:
        d = inv.shape[0]
        return np.einsum('ij,kl,jlmn', inv, inv.conj(), T)
    return inv @ T

def avg_fidelity_no_phases(T):
    return avg_fidelity(remove_phases(T))
