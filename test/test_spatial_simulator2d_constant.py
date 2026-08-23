import numpy as np
from .testing_tools import unittest, TestCase
from modgen2d.spatial_simulator2d import ConstantSimulator

class TestConstantSimulator(TestCase):
    def setUp(self):
        self.points = np.array([[0., 0.], [1., 1.], [2., 3.]])
    def test_constant_and_linear_means(self):
        sim = ConstantSimulator()
        self.assertArrayEqual(sim.simulate(self.points, mean=10), np.full(3, 10.))
        self.assertArrayEqual(sim.simulate(self.points, mean=[5, 2]),
                              5 + 2 * self.points[:, 1])
    def test_sigma_is_intentionally_ignored(self):
        sim = ConstantSimulator()
        self.assertArrayEqual(sim.simulate(self.points, 4, 100),
                              sim.simulate(self.points, 4, 0))
    def test_disallowed_simulation_raises(self):
        sim = ConstantSimulator(); sim.allow_simulation = False
        with self.assertRaises(RuntimeError): sim.simulate(self.points)
    
if __name__ == "__main__":
    unittest.main()