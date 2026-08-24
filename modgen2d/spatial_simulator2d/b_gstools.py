"""Concrete spatial simulators for 2D lithological domains."""

import numpy as np
import gstools as gs
from .a_abstract import SpatialSimulator2DAbstract

class GSToolsSimulator(SpatialSimulator2DAbstract):
    """
    Spatial simulator using GSTools' exponential covariance model.

    The GSTools length scales are adjusted internally to remain consistent
    with the ``theta_x`` and ``theta_z`` convention used by
    :class:`CovarianceDecompositionSimulator`.

    Despite the adjustment, these two simulators do not define identical 
    two-dimensional covariance models. ``CovarianceDecompositionSimulator``
    uses a separable, Manhattan-style exponential correlation:

    ``exp(-2*|dx|/theta_x) * exp(-2*|dz|/theta_z)``

    GSTools uses an anisotropic Euclidean-distance exponential correlation.
    Consequently, the correlations are consistent along the x- and z-axes
    but differ when both ``dx`` and ``dz`` are nonzero. The generated
    realizations also differ because GSTools uses approximate spectral
    randomization instead of direct Cholesky decomposition.

    Parameters
    ----------
    theta_x : float
        Correlation length in the x-direction, following the convention
        used by ``CovarianceDecompositionSimulator``.
    theta_z : float
        Correlation length in the z-direction, following the convention
        used by ``CovarianceDecompositionSimulator``.
    mode_no : int, default=1000
        Number of spectral modes used by ``RandMeth``.
    simulated_val_for_ignored_lit_property : int, default=-99999
        Value assigned to ignored lithological IDs.
    rng : numpy.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        theta_x,
        theta_z,
        mode_no=1000,
        simulated_val_for_ignored_lit_property=-99999,
        rng=None,
        verbose=False
    ):
        # Validate correlation lengths.
        for name, value in {
            "theta_x": theta_x,
            "theta_z": theta_z,
        }.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
            ):
                raise TypeError(
                    f"{name} must be a numeric scalar."
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        # Validate number of spectral modes.
        if (
            isinstance(mode_no, (bool, np.bool_))
            or not isinstance(mode_no, (int, np.integer))
        ):
            raise TypeError(
                "mode_no must be a positive integer."
            )

        if mode_no <= 0:
            raise ValueError(
                "mode_no must be greater than zero."
            )

        params = {
            "theta_x": float(theta_x),
            "theta_z": float(theta_z),
            "mode_no": int(mode_no),
        }

        super().__init__(
            params=params,
            simulated_val_for_ignored_lit_property=simulated_val_for_ignored_lit_property,
            rng=rng, verbose=verbose
        )

    def simulate(self, points, mean=0, sigma=1):
        """
        Generate a Gaussian random field at the supplied coordinates.

        Parameters
        ----------
        points : array-like, shape (n, 2)
            Coordinates in ``(x, z)`` order.
        mean : float or sequence of two floats, default=0
            Constant mean or ``[a_m, b_m]`` defining
            ``a_m + b_m * z``.
        sigma : float, default=1
            Marginal standard deviation.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Simulated values in the same order as ``points``.
        """
        if not self.allow_simulation:
            raise RuntimeError(self._invalid_simulation_message())
            
        pts = self.check_points(points)
        a_m, b_m = self.get_means_am_bm(mean)
        mean_vector = a_m + b_m * pts[:, 1]

        if sigma == 0:
            return mean_vector

        # Convert the package's exp(-2) directional correlation lengths
        # to GSTools' exponential len_scale convention.
        gstools_len_scale_x = (
            self.params["theta_x"] / 2.0
        )
        gstools_len_scale_z = (
            self.params["theta_z"] / 2.0
        )
        
        model = gs.Exponential(
            dim=2,
            var=1.0,
            len_scale=[
                gstools_len_scale_x,
                gstools_len_scale_z,
            ],
            angles=0.0,
        )

        # Generate a GSTools seed from the simulator's NumPy RNG.
        gstools_seed = int(
            self.rng.integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )

        srf = gs.SRF(
            model,
            mean=0.0,
            seed=gstools_seed,
            generator="RandMeth",
            mode_no=self.params["mode_no"],
        )

        standardized_field = np.asarray(
            srf.unstructured(
                (pts[:, 0], pts[:, 1])
            ),
            dtype=float,
        ).reshape(-1)

        if standardized_field.size != pts.shape[0]:
            raise RuntimeError(
                "GSTools returned an unexpected number of values: "
                f"expected {pts.shape[0]}, "
                f"received {standardized_field.size}."
            )

        return mean_vector + float(sigma) * standardized_field
