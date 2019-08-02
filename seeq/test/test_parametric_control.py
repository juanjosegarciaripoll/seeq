
import numpy as np
import scipy.sparse as sp
from seeq.states import *

from seeq.control import *

import unittest

class TestQControl(unittest.TestCase):
    π = np.pi
    σz = np.array([[1., 0.],[0., -1.]])
    σx = np.array([[0., 1.],[1., 0.]])
    σy = np.array([[0., -1.j],[1.j, 0.]])
    ψ0 = np.eye(2)

    def test_nothing(self):
        """For a qubit to remain the same, we do nothing."""
        Ug = np.eye(2)
        H = lambda t, x, ψ: x * (self.σx @ ψ)
        r = parametric_control([1.0], H, self.ψ0, Ug, T=1.0, tol=1e-8, method='expm')
        self.assertEqual(len(r.x), 1)
        self.assertAlmostEqual(r.x[0], 0.0, delta=1e-7)

    def test_nothing2(self):
        """For a qubit to remain the same, we cancel the frequency."""
        Ug = np.eye(2)
        H = lambda t, x, ψ: x[0] * (self.σx @ ψ) + (1.0 - x[1]) * (self.σz @ ψ)
        r = parametric_control([1.0, 0.1], H, self.ψ0, Ug, T=1.0, tol=1e-8, method='expm')
        self.assertEqual(len(r.x), 2)
        self.assertAlmostEqual(r.x[0], 0.0, delta=1e-7)
        self.assertAlmostEqual(r.x[1], 1.0, delta=1e-7)

    def test_qubit_flip(self):
        """Construct a π/2 pulse."""
        Ug = -1j*self.σy
        H = lambda t, x, ψ: (x * self.σy) @ ψ
        r = parametric_control([1.0], H, self.ψ0, Ug, T=1.0, tol=1e-9, method='expm')
        self.assertEqual(len(r.x), 1)
        self.assertAlmostEqual(r.x[0], self.π/2., delta=1e-7)

    def test_nothing_derivative(self):
        """For a qubit to remain the same, we do nothing (with gradients)."""
        Ug = np.eye(2)
        H = lambda t, x, ψ: x * (self.σx @ ψ)
        dH = lambda t, x, ψ: [self.σx @ ψ]
        r = parametric_control([1.0], H, self.ψ0, Ug, T=1.0, dH=dH, tol=1e-8, method='expm')
        self.assertEqual(len(r.x), 1)
        self.assertAlmostEqual(r.x[0], 0.0, delta=1e-7)

    def test_qubit_flip_derivative(self):
        """Construct a π/2 pulse (with gradients)."""
        Ug = -1j*self.σy
        H = lambda t, x, ψ: (x * self.σy) @ ψ
        dH = lambda t, x, ψ: [self.σy @ ψ]
        r = parametric_control([1.0], H, self.ψ0, Ug, T=1.0, dH=dH, tol=1e-9, method='expm')
        self.assertEqual(len(r.x), 1)
        self.assertAlmostEqual(r.x[0], self.π/2., delta=1e-7)
