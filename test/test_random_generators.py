# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import numpy as np
from testing_tools import unittest, TestCase
from modgen2d import random_generators

class TestDomain2D(TestCase):
    @classmethod
    def setUp(self):
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)

    def test_constant_generate(self):
        val = 5.0
        c = random_generators.Constant(val, rng=self.rng)
        self.assertEqual(c.value, val)
        result = c.generate(size=(10,))
        self.assertTrue(np.all(result == val))
        self.assertEqual(result.shape, (10,))

    def test_constant_invalid(self):
        with self.assertRaises(ValueError):
            random_generators.Constant(None)
        with self.assertRaises(ValueError):
            random_generators.Constant(float('nan'))
        with self.assertRaises(ValueError):
            random_generators.Constant(np.nan)

    def test_uniform(self):
        low, high = 1.0, 3.0
        u = random_generators.Uniform(low, high, rng=self.rng)
        result = u.generate(size=(1000,))
        self.assertTrue(np.all((result >= 1) & (result <= 3)))
        self.assertEqual(result.shape, (1000,))

        self.assertEqual(u.low, low)
        self.assertEqual(u.high, high)
        
    def test_uniform_invalid(self):
        self.assertRaises(ValueError, random_generators.Uniform, 5, 2)
        self.assertRaises(TypeError, random_generators.Uniform, 'a', 2)
        self.assertRaises(ValueError, random_generators.Uniform, np.nan, 2)
        self.assertRaises(ValueError, random_generators.Uniform, 2,  np.nan)
    
    def test_loguniform_generate(self):
        lu = random_generators.LogUniform(1, 10, rng=self.rng)
        result = lu.generate(size=(1000,))
        self.assertTrue(np.all((result >= 1) & (result <= 10)))
        self.assertEqual(lu.low, 1)
        self.assertEqual(lu.high, 10)
        self.assertEqual(result.shape, (1000,))

    def test_loguniform_invalid(self):
        self.assertRaises(ValueError, random_generators.LogUniform, 0, 2)
        self.assertRaises(ValueError, random_generators.LogUniform, -1, 2)
        self.assertRaises(ValueError, random_generators.LogUniform, 5, 2)
        self.assertRaises(ValueError, random_generators.LogUniform, np.nan, 2)
        self.assertRaises(ValueError, random_generators.LogUniform, 2,  np.nan)

    def test_normal_generate(self):
        mean = 1
        stdev = 3
        n = random_generators.Normal(mean=mean, stdev=stdev, rng=self.rng)
        result = n.generate(size=(1000,))
        self.assertEqual(result.shape, (1000,))

        self.assertEqual(n.mean, mean)
        self.assertEqual(n.stdev, stdev)
        
        self.assertAlmostEqual(np.mean(result), mean, delta=0.1)
        self.assertAlmostEqual(np.std(result), stdev, delta=0.1)

    def test_normal_invalid(self):
        self.assertRaises(TypeError, random_generators.Normal, 'a', 2)
        self.assertRaises(TypeError, random_generators.Normal, 0, 'a')
        self.assertRaises(ValueError, random_generators.Normal, np.nan, 2)
        self.assertRaises(ValueError, random_generators.Normal, 2, np.nan)
        
        
    def test_discrete_choice_generate(self):
        x = ['a', '2', 'c'] #allowed as nupy upcast is used. 2 treated as '2'
        p = [0.2, 0.3, 0.5]
        d = random_generators.DiscreteChoice(x, p, rng=self.rng)
        self.assertArrayEqual(d.x, x)
        self.assertArrayAlmostEqual(d.p, p)
        
        result = d.generate(size=(1000,))
        self.assertTrue(set(result).issubset(set(np.asarray(x))))
        
        # Check if probabilities roughly match
        counts = [np.sum(result == xi)/len(result) for xi in np.asarray(x)]
        for c, expected in zip(counts, p):
            self.assertAlmostEqual(c, expected, delta=0.02)
            
    def test_discrete_choice_uniform_prob(self):
        x = [1, 2, 3]
        d = random_generators.DiscreteChoice(x, rng=self.rng)
        result = d.generate(size=(1000,))
        self.assertTrue(set(result).issubset(set(x)))

        self.assertArrayEqual(d.x, x)
        self.assertArrayAlmostEqual(d.p, np.ones(3)/3)

        # Check if probabilities roughly match
        counts = [np.sum(result == xi)/len(result) for xi in x]
        
        p=[1/3, 1/3, 1/3]
        for c, expected in zip(counts, p):
            self.assertAlmostEqual(c, expected, delta=0.02)
            
    def test_discrete_choice_invalid(self):
        x = [1, 2]
        p = [0.5, 0.4]  # does not sum to 1
        with self.assertRaises(ValueError):
            random_generators.DiscreteChoice(x, p)
        
        p = [1, np.nan]
        with self.assertRaises(ValueError):
            random_generators.DiscreteChoice(x, p)
            
        x = [1, 2]
        p = [1]  # does not sum to 1
        with self.assertRaises(TypeError):
            random_generators.DiscreteChoice(x, p)

    def test_discrete2continuous_pdf_basic(self):
        x = [0, 1, 2]
        p = [0.2, 0.3, 0.5]
        new_del_x = 0.5
        d2c = random_generators.Discrete2ContinuousPDF(x, p, new_del_x,rng=self.rng)
        self.assertTrue(np.isclose(np.sum(d2c.p), 1.0))
    
if __name__ == "__main__":
    unittest.main()
