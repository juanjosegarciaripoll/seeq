import numpy as np
import scipy.sparse as sp

def lowest_eigenstates(H, neig):
    """Compute the lowest eigenstates of the Hamiltonian H.
    
    Arguments:
    ----------
    neig -- An integer denoting the number of lowest energy eigenstates
            to compute, or a list of numbers denoting the eigenstates
            to select (e.g. [0, 3, 5] for the ground state, 3rd excited, etc)
    Returns:
    --------
    λ    -- The eigenergies, sorted
    ψ    -- The corresponding eigenstates, as columns of this matrix.
    """
    λ, ψ = sp.linalg.eigsh(H, neig, which='SA', return_eigenvectors=True)
    ndx = np.argsort(λ)
    return λ[ndx], ψ[:,ndx]

def lowest_eigenvalues(H, neig):
    """Compute the lowest eigenstates of the Hamiltonian H.
    
    Arguments:
    ----------
    neig -- An integer denoting the number of lowest energy eigenstates
            to compute, or a list of numbers denoting the eigenstates
            to select (e.g. [0, 3, 5] for the ground state, 3rd excited, etc)
    Returns:
    --------
    λ    -- The eigenergies, sorted
    """
    λ = sp.linalg.eigsh(H, neig, which='SA', return_eigenvectors=False)
    return np.sort(λ)
