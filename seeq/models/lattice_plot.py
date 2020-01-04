
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.sparse as sp
import numpy as np

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

def plot_field2d(lattice, field, ax=None, x=None, y=None,
                 Lx=100, Ly=100, σ=1/2.0, cmap='Greys'):
    """Plot a field that lives in a 2D lattice.
    
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
        fig, ax = plt.subplots()
    if x is None:
        x = np.linspace(lattice.Xmin-σ, lattice.Xmax+σ, Lx)
    else:
        Lx = len(x)
    if y is None:
        y = np.linspace(lattice.Ymin-σ, lattice.Ymax+σ, Ly)
    else:
        Ly = len(y)
    extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])/1
    aspect = 'auto'
    x = np.reshape(x, (1,Lx))
    y = np.reshape(y, (Ly,1))
    dty = np.zeros((Ly, Lx))
    for (n, (X, Y, Z)) in zip(field, lattice.coord):
        dty += n * np.exp(-((x-X)**2+(y-Y)**2)/σ**2)
    ax.imshow(dty, extent=extent, aspect=aspect, interpolation='none',
              origin='lower', cmap=cmap)
    return ax, dty
