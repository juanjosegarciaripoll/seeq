
import unittest

class TestOperators(unittest.TestCase):
    
    @staticmethod
    def commute(x, y):
        return x @ y - y @ x
    
    def assertEqualMatrix(self, a, b):
        if (a.ndim == b.ndim):
            if (a.shape == b.shape):
                if sp.issparse(a):
                    a = a.todense()
                if sp.issparse(b):
                    b = b.todense()
                if (a == b).all():
                    return True
        raise self.failureException(f'Matrices not equal:\na={a}\nb={b}')

    def test_commutation_Pauli(self):
        """For a qubit to remain the same, we do nothing."""
        for (a, b, c) in [(σx, σy, 2.j*σz)]:
            self.assertEqualMatrix(self.commute(σx, σy), 2j*σz)
            self.assertEqualMatrix(self.commute(σy, σz), 2j*σx)
            self.assertEqualMatrix(self.commute(σz, σx), 2j*σy)

    def test_boson_creation(self):
        """For a qubit to remain the same, we do nothing."""
        a2 = np.array([[0,1.0],
                       [0,0]])
        n2 = np.array([[0,0],[0,1]])
        self.assertEqualMatrix(boson_creation(2), a2.T)
        self.assertEqualMatrix(boson_anihilation(2), a2)
        self.assertEqualMatrix(boson_number(2), n2)
        
        a3 = np.array([[0,1.0,0],
                       [0,0,np.sqrt(2.)],
                       [0,0,0]])
        n3 = np.array([[0,0,0],[0,1.,0],[0,0,2.]])
        self.assertEqualMatrix(boson_creation(3), a3.T)
        self.assertEqualMatrix(boson_anihilation(3), a3)
        self.assertEqualMatrix(boson_number(3), n3)
        
        a4 = np.array([[0,1.0,0,0],
                       [0,0,np.sqrt(2.),0],
                       [0,0,0,np.sqrt(3.)],
                       [0,0,0,0]])
        n4 = np.array([[0,0,0,0],[0,1.,0,0],[0,0,2.,0],[0,0,0,3.]])
        self.assertEqualMatrix(boson_creation(4), a4.T)
        self.assertEqualMatrix(boson_anihilation(4), a4)
        self.assertEqualMatrix(boson_number(4), n4)
