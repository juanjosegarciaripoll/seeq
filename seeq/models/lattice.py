import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

class Lattice(LinearOperator):
    """Generic Lattice model with hopping/local frequencies.
    
    Properties
    ----------
    H         -- NxN matrix of tunnelings and local frequencies.
    dimension -- Dimensionality of the model.
    """
    
    def __init__(self, H, dimension=1):
        super(Lattice, self).__init__(np.float64, H.shape)
        self._H = H
        self.size = H.shape[0]
        self.dimension = dimension

    def hamiltonian(self):
        return self._H

    def coupling_at(self, r):
        """Return a vector of couplings for an impurity at position
        `r` in this generic lattice. The interpretation of `r` is
        left up to the subclass."""
        pass

    def _matvec(self, v):
        # Implement the matrix-vector multiplication
        # Inherited interface from LinearOperator()
        return self._H @ v

    def _matmat(self, A):
        # Implement the matrix-matrix multiplication
        # Inherited interface from LinearOperator()
        return self._H @ A

class Regular3DLattice(Lattice):
    """Regular lattice of equispaced points, with arbitrary connectivity and boundary.
    
    Parameters
    ----------
    sizes     -- tuple (Lx, Ly, Lz) of sizes for the lattice
    hopping   -- function of three arguments (X,Y,Z), returning a list of pairs
                 [(J, (X,Y,Z))] with the hopping and the neighbor's coordinate.
    r0        -- location of the corner (defaults to (0,0,0))
    condition -- function of three arguments (X,Y,Z), returning true if the point
                 belongs to the lattice.
    """
    
    def __init__(self, sizes, hopping, g=None, r0=(0,0,0), condition=None):
        
        if condition == None:
            condition = lambda X, Y, Z: True

        # Construct all points in this lattice
        X0, Y0, Z0 = r0
        coord = [(X,Y,Z)
                 for X in range(X0,X0+sizes[0])
                 for Y in range(Y0,Y0+sizes[1])
                 for Z in range(Z0,Z0+sizes[2])
                 if condition(X, Y, Z)]
        #
        # Assign an index to each point
        ndx_map = {vector: ndx for ndx, vector in enumerate(coord)}
        #
        # Construct the list of neighbors for each point, together
        # with their hoppings
        hops = np.array([(J, i, ndx_map[dest])
                         for i, orig in enumerate(coord)
                         for J, dest in hopping(*orig)
                         if dest in ndx_map])
        #
        # Use this information to build the sparse matrix `H` of
        # hoppings and frequencies.
        L = len(ndx_map)
        H = sp.csr_matrix((hops[:,0],(hops[:,1],hops[:,2])), shape=(L,L))
        #
        # Determine the dimensionality 
        if sizes[2] > 1:
            dimension = 3
        elif sizes[1] > 1:
            dimension = 2
        else:
            dimension = 1
        super(Regular3DLattice, self).__init__(H, dimension)
        self.coord = coord = np.array(coord)
        self.ndx_map = ndx_map
        self.Xmin = min(coord[:,0])
        self.Xmax = max(coord[:,0])
        self.Ymin = min(coord[:,1])
        self.Ymax = max(coord[:,1])
        self.Zmin = min(coord[:,2])
        self.Zmax = max(coord[:,2])

    def coupling_at(self, r, g=1.0):
        """Return vector of couplings at given position."""
        ndx = ndx_map.get(r, None)
        if ndx is None:
            raise Exception(f'Emitter position {r} is not in the lattice.')
        gr = np.zeros(self.sizeL)
        gr[ndx] = g
        return gr

    def vertex_index(self, r):
        """Return the index of position 'r' in the lattice."""
        return self.ndx_map[r]

class Lattice1D(Regular3DLattice):
    """Latice for a 1D model with nearest-neighbor hoppings.
    
    Parameters
    ----------
    L         -- Lattice length (number of vertices)
    J         -- Hopping amplitude
    ω         -- Local energy on each site
    r0        -- location of the corner (defaults to (0,0,0))
    """

    def __init__(self, L, J=1, ω=1, **kwdargs):
        
        def hopping1d(X, Y, Z):
            return [(ω, (X,Y,Z)), (J, (X+1,Y,Z)), (J, (X-1,Y,Z))]

        super(Lattice1D, self).__init__([L,1,1], hopping1d, **kwdargs)
        self.J = J
        self.ω = ω

class SquareLattice(Regular3DLattice):
    """Latice for a 2D model with nearest-neighbor hoppings.
    
    Parameters
    ----------
    Lx, Ly    -- Lattice length (number of vertices). Ly defaults to Lx
    Jx, Jy    -- hopping amplitudes (Jy defaults to Jx, Jx defaults to 1)
    ω         -- Local energy on each site
    r0        -- location of the corner (defaults to (0,0,0))
    """

    def __init__(self, Lx, Ly=None, Jx=1.0, Jy=None, ω=1, **kwdargs):
        
        if Ly is None:
            Ly = Lx
        if Jy is None:
            Jy = Jx

        def hopping2d(X, Y, Z):
            return [(ω, (X,Y,Z)), (Jx, (X+1,Y,Z)), (Jx, (X-1,Y,Z)), (Jy, (X,Y+1,Z)), (Jy, (X,Y-1,Z))]

        super(SquareLattice, self).__init__([Lx,Ly,1], hopping2d, **kwdargs)
        self.J = (Jx, Jy)
        self.ω = ω

class RhombusLattice(Regular3DLattice):
    """Latice for a 2D model with nearest-neighbor hoppings and a boundary
    that resembles a Rhombus
    
    Parameters
    ----------
    L         -- Lattice length (number of vertices)
    Jx, Jy    -- hopping amplitudes (Jy defaults to Jx, Jx defaults to 1)
    ω         -- Local energy on each site
    r0        -- Center of the rombus
    """

    def __init__(self, L, Jx=1.0, Jy=None, ω=1, r0=(0,0,0), **kwdargs):
        if Jy is None:
            Jy = Jx

        X0, Y0, Z0 = r0
        r0 = (X0 - L, Y0 - L, Z0)

        def hopping2d(X, Y, Z):
            return [(ω, (X,Y,Z)), (Jx, (X+1,Y,Z)), (Jx, (X-1,Y,Z)), 
                    (Jy, (X,Y+1,Z)), (Jy, (X,Y-1,Z))]

        def condition(X, Y, Z):
            X -= X0
            Y -= Y0
            return ((np.abs(X-Y) <= L) & (np.abs(X+Y) <= L))+0

        super(RhombusLattice, self).__init__([2*L+1,2*L+1,1], hopping2d, r0=r0,
                                             condition=condition, **kwdargs)
        self.J = (Jx, Jy)
        self.ω = ω

class CubicLattice(Regular3DLattice):
    """Latice for a 3D model with a cubic lattice.
    
    Parameters
    ----------
    Lx, Ly, Lz -- Lattice length (number of vertices, Ly and Lz default to Lx)
    Jx, Jy, Jx -- hopping amplitudes (Jx defaults to 1, Jy and Jz default to Jx)
    ω          -- Local energy on each site
    r0         -- Corner of the cube
    """

    def __init__(self, Lx, Ly=None, Lz=None, Jx=1.0, Jy=None, Jz=None, ω=1, **kwdargs):
        if Ly is None:
            Ly = Lx
        if Lz is None:
            Lz = Lx
        if Jy is None:
            Jy = Jx
        if Jz is None:
            Jz = Jx

        def hopping3d(X, Y, Z):
            return [(ω, (X,Y,Z)),
                    (Jx, (X+1,Y,Z)), (Jx, (X-1,Y,Z)),
                    (Jy, (X,Y+1,Z)), (Jy, (X,Y-1,Z)),
                    (Jz, (X,Y,Z+1)), (Jz, (X,Y,Z-1))]

        super(CubicLattice, self).__init__([Lx,Ly,Lz], hopping3d, **kwdargs)
        self.J = (Jx, Jy, Jz)
        self.ω = ω

class BCCLattice(Regular3DLattice):
    """Latice for a 3D model with a BCC lattice.
    
    Parameters
    ----------
    Lx, Ly, Lz -- Lattice length (number of vertices, Ly and Lz default to Lx)
    J          -- hopping amplitudes (defaults to 1)
    ω          -- Local energy on each site
    r0         -- Corner of the cube
    """

    def __init__(self, Lx, Ly=None, Lz=None, J=1.0, ω=1, **kwdargs):
        if Ly is None:
            Ly = Lx
        if Lz is None:
            Lz = Lx

        def bcc_hopping(X, Y, Z):
            return [(ω, (X,Y,Z)),
                     (J, (X+1,Y+1,Z+1)), (J, (X-1,Y-1,Z+1)),
                     (J, (X+1,Y-1,Z+1)), (J, (X-1,Y+1,Z+1)),
                     (J, (X+1,Y+1,Z-1)), (J, (X-1,Y-1,Z-1)),
                     (J, (X+1,Y-1,Z-1)), (J, (X-1,Y+1,Z-1))]

        def bcc_condition(X, Y, Z):
            dX = X % 2
            dY = Y % 2
            dZ = Z % 2
            return (dX == dY) and (dX == dZ)

        super(BCCLattice, self).__init__([Lx, Ly, Lz], bcc_hopping,
                                         condition=bcc_condition, **kwdargs)
        self.J = J
        self.ω = ω
