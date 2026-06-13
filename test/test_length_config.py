"""Create basic sanity checks for LengthConfig."""

from .testing_tools import unittest, TestCase
from modgen2d import length_config

class TestLengthConfig(TestCase):
    @classmethod
    def setUpClass(cls):
        # cls.length_config1 = length_config.LengthConfig()
        cls.length_config1 = length_config.LengthConfig(
            physical_length_unit="m",
            min_dl=0.000001,
        )
        cls.length_config2 = length_config.LengthConfig(
            physical_length_unit="cm",
            max_grid_density=100,
        )

    def test_config_values(self):
        self.assertEqual(self.length_config1.physical_length_unit, "m")
        self.assertEqual(self.length_config1.min_dl, 0.000001)
        self.assertEqual(self.length_config1.max_grid_density, 1000000)

        self.assertEqual(self.length_config2.physical_length_unit, "cm")
        self.assertEqual(self.length_config2.min_dl, 0.01)
        self.assertEqual(self.length_config2.max_grid_density, 100)

    def test_to_domain_length_unit(self):
        self.assertEqual(self.length_config1.to_domain_length_unit(2.200004), 2200004)
        self.assertEqual(self.length_config1.to_domain_length_unit(0.000025), 25)
        self.assertEqual(self.length_config2.to_domain_length_unit(2.5), 250)

    def test_to_physical_length_unit(self):
        self.assertEqual(self.length_config1.to_physical_length_unit(2200004), 2.200004)
        self.assertEqual(self.length_config1.to_physical_length_unit(25), 0.000025)
        self.assertEqual(self.length_config2.to_physical_length_unit(250), 2.5)

    def test_equality(self):
        same = length_config.LengthConfig("m", 0.000001)
        self.assertEqual(self.length_config1, same)

        diff_unit = length_config.LengthConfig("cm", 0.000001)
        self.assertNotEqual(self.length_config1, diff_unit)

        diff_density = length_config.LengthConfig("m", 0.0001)
        self.assertNotEqual(self.length_config1, diff_density)

    def test_invalid_config_error(self):
        with self.assertRaises(ValueError):
            length_config.LengthConfig()
            
        with self.assertRaises(TypeError):
            length_config.LengthConfig(1, 0.0001)

        with self.assertRaises(ValueError):
            length_config.LengthConfig("meter_long", 0.0001)

        with self.assertRaises(ValueError):
            length_config.LengthConfig("m", None, None)

        with self.assertRaises(ValueError):
            length_config.LengthConfig("m", 0)

        with self.assertRaises(ValueError):
            length_config.LengthConfig("m", -0.1)

        with self.assertRaises(ValueError):
            length_config.LengthConfig("m", 0.0001, 100)

        with self.assertRaises(ValueError):
            length_config.LengthConfig("m", 0.0003)

    def test_invalid_conversion_error(self):
        with self.assertRaises(TypeError):
            self.length_config1.to_domain_length_unit("1")

        with self.assertRaises(ValueError):
            self.length_config1.to_domain_length_unit(-1)

        with self.assertRaises(ValueError):
            self.length_config1.to_domain_length_unit(0.0000015)

    def test_immutability(self):
        with self.assertRaises(AttributeError):
            self.length_config1.min_dl = 0.01

    def test_get_config_and_from_config(self):
        config = self.length_config1.get_config
        config_result = {
            "physical_length_unit": "m",
            "min_dl": 0.000001,
            "max_grid_density": 1000000,
        }

        self.assertDictDeepAlmostEqual(config, config_result, tol=1e-12)

        recreated = length_config.LengthConfig.from_config(config)
        self.assertEqual(self.length_config1, recreated)


if __name__ == "__main__":
    unittest.main()