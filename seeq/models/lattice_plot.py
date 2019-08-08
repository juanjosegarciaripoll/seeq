
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_lattice(L, ax=None, dot='.'):
    """Plot a 2D or 3D representation of the lattice on the given
    axis, or create one if none is given.
    
    Parameters
    ----------
    L    -- A Regular3DLattice() object
    ax   -- Axis to plot on. If None, create a new one.
    dot  -- Symbol to plot on the vertices of the lattice.
    
    Returns
    -------
    ax   -- Axis on which the figure is plot.
    """
    if ax is None:
        if L.dimension <= 2:
            fig, ax = plt.subplots()
        else:
            import mpl_toolkits.mplot3d
            fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    #
    # We plot all the connections
    #
    H = L.hamiltonian()
    coord = L.coord
    aux = sp.coo_matrix(H)
    if L.dimension == 3:
        import mpl_toolkits.mplot3d.art3d
        lines = [(coord[a,:], coord[b,:])
                 for (a,b) in zip(aux.row, aux.col)
                 if H[a,b] != 0]
        lc = mpl_toolkits.mplot3d.art3d.Line3DCollection(lines, linewidths=0.2)
    else:
        import matplotlib.collections
        lines = [(coord[a,0:2], coord[b,0:2])
                 for (a,b) in zip(aux.row, aux.col)
                 if H[a,b] != 0]
        lc = matplotlib.collections.LineCollection(lines, linewidths=0.2)
    ax.add_collection(lc)
    #
    # First we plot all the dots that are connected to others
    #
    ndx, _ = np.nonzero(np.sum(np.abs(H), 1))
    points = coord[ndx,:]
    if L.dimension == 3:
        ax.plot(points[:,0], points[:,1], points[:,2], '.')
    else:
        ax.plot(points[:,0], points[:,1], '.')
    return ax
