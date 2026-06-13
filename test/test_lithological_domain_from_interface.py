import numpy as np
from .testing_tools import unittest, TestCase

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from modgen2d.interface import DiscretizedInterfaces2D, GlobalSoilInterfaceConfig
from modgen2d.lithological_domain2d import LithologicalDomain2D

class TestLithologicalDomain2DFromInterface(TestCase):

    def setUp(self):
        GlobalSoilInterfaceConfig.reset()

        self.domain = DiscretizedDomain2D(span_x=5, span_z=4, dx=1, dz=1)

    def tearDown(self):
        GlobalSoilInterfaceConfig.reset()

    def _make_interface(self, generate_surface=True):
        interface = DiscretizedInterfaces2D(
            domain=self.domain,
            n_soil_layers=3,
            generate_surface=generate_surface,
            rng=np.random.default_rng(2),
        )

        if generate_surface:
            mat = np.tile([0.0, 1.5, 3.0], (7, 1))
        else:
            mat = np.tile([0.0, 1.5, 3.0], (7, 1))

        interface.set_interfaces_matrix(mat)
        interface.lock_interfaces()
        return interface

    def test_global_soil_interface_config(self):
        interface = self._make_interface(generate_surface=True)

        with self.assertRaises(TypeError):
            GlobalSoilInterfaceConfig.set_soil_interface(None)

        with self.assertRaises(TypeError):
            GlobalSoilInterfaceConfig.set_soil_interface(self.domain)

        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        self.assertIs(GlobalSoilInterfaceConfig.get_interface_instance(), interface)
        self.assertNotEqual(GlobalSoilInterfaceConfig.get_revision_id(), 0)

        with self.assertRaises(RuntimeError):
            GlobalSoilInterfaceConfig.set_soil_interface(interface)

        old_revision = GlobalSoilInterfaceConfig.get_revision_id()
        GlobalSoilInterfaceConfig.set_soil_interface(interface, force_set=True)

        self.assertNotEqual(old_revision, GlobalSoilInterfaceConfig.get_revision_id())

    def test_lithological_domain_from_surface_interface(self):
        interface = self._make_interface(generate_surface=True)
        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        lit = LithologicalDomain2D(
            domain=self.domain,
            gwt_depth=1.8,
            name="soil",
        )

        self.assertEqual(lit.name, "soil")
        self.assertEqual(lit.gwt_depth, 1.8)
        self.assertEqual(lit.lm_type, "from_interface_config")
        self.assertTupleEqual(lit.lithological_matrix.shape, self.domain.shape)
        self.assertFalse(lit.check_for_Xs(lit.lithological_matrix))
        self.assertDictEqual(lit.get_feature_id_and_lit_val(), {"def": [0, 1, 2, 3]})

    def test_lithological_domain_from_no_surface_interface(self):
        interface = self._make_interface(generate_surface=False)
        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        lit = LithologicalDomain2D(
            domain=self.domain,
            gwt_depth=1.8,
            name="soil",
        )

        self.assertEqual(lit.lm_type, "from_interface_config")
        self.assertTupleEqual(lit.lithological_matrix.shape, self.domain.shape)
        self.assertFalse(lit.check_for_Xs(lit.lithological_matrix))
        self.assertDictEqual(lit.get_feature_id_and_lit_val(), {"def": [1, 2, 3]})

    def test_lithological_domain_remeshed_from_interface(self):
        interface = self._make_interface(generate_surface=True)
        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        fine_domain = DiscretizedDomain2D(
            span_x=5,
            span_z=4,
            dx=0.5,
            dz=0.5,
        )

        lit = LithologicalDomain2D(
            domain=fine_domain,
            gwt_depth=1.8,
            name="soil_fine",
        )

        self.assertTupleEqual(lit.domain.shape, (10, 8))
        self.assertTupleEqual(lit.lithological_matrix.shape, (10, 8))
        self.assertEqual(lit.init_domain, None)

    def test_lithological_domain_invalid_domain_span(self):
        interface = self._make_interface(generate_surface=True)
        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        bad_domain = DiscretizedDomain2D(
            span_x=6,
            span_z=4,
            dx=1,
            dz=1,
        )

        with self.assertRaises(ValueError):
            LithologicalDomain2D(bad_domain, 1.8, "bad")

    def test_get_config_and_from_config(self):
        interface = self._make_interface(generate_surface=True)
        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        lit = LithologicalDomain2D(self.domain, 1.8, "soil")

        config = lit.get_config
        recreated = LithologicalDomain2D.from_config(config)

        self.assertEqual(recreated.domain, lit.domain)
        self.assertEqual(recreated.name, lit.name)
        self.assertEqual(recreated.lm_type, lit.lm_type)
        self.assertArrayEqual(recreated.lithological_matrix, lit.lithological_matrix)

    def test_from_config_rejects_wrong_lm_type(self):
        interface = self._make_interface(generate_surface=True)
        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        lit = LithologicalDomain2D(self.domain, 1.8, "soil")
        config = lit.get_config
        config["lm_type"] = "bad_type"

        with self.assertRaises(ValueError):
            LithologicalDomain2D.from_config(config)

if __name__ == "__main__":
    unittest.main()
