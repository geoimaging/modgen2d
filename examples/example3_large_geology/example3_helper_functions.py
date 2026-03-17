from scipy.ndimage import gaussian_filter
import modgen2d as mg2d
import modgen2d.general_functions as f
import numpy as np


def add_features_from_pd(rng, main_property_instance, main_property_name, feature_id, pd_dataframe, cov_distribution = None, cov_type='cov'):
    wet_a_colname, wet_b_colname = f"{main_property_name}_wet_a", f"{main_property_name}_wet_b" 
    dry_a_colname, dry_b_colname = f"{main_property_name}_dry_a", f"{main_property_name}_dry_b" 
    for material_name in pd_dataframe.index.tolist():
        # print(material_name)
        wet_a, wet_b = pd_dataframe[wet_a_colname].loc[material_name], pd_dataframe[wet_b_colname].loc[material_name]
        dry_a, dry_b = pd_dataframe[dry_a_colname].loc[material_name], pd_dataframe[dry_b_colname].loc[material_name]

        assert not pd.isna(wet_a), f"wet_a for {material_name} must be a number"
        assert not pd.isna(wet_b), f"wet_a for {material_name} must be a number"
        wet_mean_distribution = mg2d.random_generators.Uniform(wet_a, wet_b, rng)
        wet_prop = mg2d.PropertyDistribution(main_property_name, wet_mean_distribution, cov_distribution, stdev_type=cov_type)

        dry_mean_distribution = None
        if pd.isna(dry_a) or pd.isna(dry_b): 
            dry_prop = None
        else:
            dry_mean_distribution = mg2d.random_generators.Uniform(dry_a, dry_b, rng)
            dry_prop = mg2d.PropertyDistribution(main_property_name, dry_mean_distribution, cov_distribution, stdev_type=cov_type)
        
        main_property_instance.add_material_property_of_feature(feature_id, material_name, wet_prop, dry_prop)
    return main_property_instance

def add_layer0(rng, main_property_instance, main_property_name, air_val, water_val):
    material_name = 'layer0'
    feature_id = 'def'
    wet_mean_distribution = mg2d.random_generators.Constant(water_val, rng)
    wet_prop = mg2d.PropertyDistribution(main_property_name, wet_mean_distribution)

    dry_mean_distribution = mg2d.random_generators.Constant(air_val, rng)
    dry_prop = mg2d.PropertyDistribution(main_property_name, dry_mean_distribution)
    main_property_instance.add_material_property_of_feature(feature_id, material_name, wet_prop, dry_prop)
    return main_property_instance


#================USER DEFINED OBSTRUCTION2D

class ManualObstruction2D(mg2d.Obstruction2D):
    def __init__(self, dl:float, ref_xz_symbolic = ['c', 'c'], snap_to_dl:bool=True):
        super().__init__(dl, ref_xz_symbolic, snap_to_dl)
       
    def karst_circle_perlin(self, d, noise_amplitude=0.1, smooth_sigma=3.0, obstruction_id=1, rng=np.random.default_rng()):
        assert self.shape is False, "ERROR: utils shape has already been defined"
        if d<0:
            raise ValueError("Diameter of the circle must be zero/positive")
        
        obstruction_id = f.check_obstruction_id(obstruction_id)

        d_adj = d
        r_nominal = d_adj / 2
        # Determine maximum radius including positive noise
        r_max = r_nominal * (1 + noise_amplitude)
        n_grid = int(np.ceil(2 * r_max / self.dl))
        
        grid = np.zeros((n_grid, n_grid), dtype=int)
        
        x = np.arange(n_grid) * self.dl + self.dl/2
        z = np.arange(n_grid) * self.dl + self.dl/2
        xx, zz = np.meshgrid(x, z, indexing='ij')
        
        # Generate smooth noise
        noise_smooth = smooth_noise((n_grid, n_grid), sigma=smooth_sigma, rng=rng)

        # Compute noisy radius
        r_grid = r_nominal * (1 + noise_amplitude * noise_smooth)

        # Compute distance from center
        center = r_max
        dist2 = (xx - center)**2 + (zz - center)**2
        grid[dist2 <= r_grid**2] = obstruction_id

        # Update object attributes
        self.grid2d = grid
        self.description = f'Karst circle (smooth noise) 2D, base diameter {d:.6g}'
        self.shape = True
        self.center_in_unit_length = [center, center]

    def tunnel_shape(self, lx, lz, obstruction_id=1, rng=np.random.default_rng()):

        if lx/2 > lz:
            raise ValueError("Height of the tunnel cannot be less than radius of the circle (roof) at top")

        ## Create the base first
        self.rectangle_2d(lx, lz, obstruction_id = obstruction_id)

        ## Create the circular top 
        circle_top = mg2d.obstruction2d.Obstruction2D(dl = self.dl, ref_xz_symbolic = ['c','c'], snap_to_dl=self.snap_to_dl)
        circle_top.circle_2d(d=lx,  obstruction_id = obstruction_id)
        
        ## Truncate the circle to semi-circle.
        r_to_grid_length = int(np.round(circle_top.center_in_unit_length[1]/self.dl,0))
        circle_top.expand_grid(new_grid_xlen = None, new_grid_zlen = r_to_grid_length, warn_truncate=False)


        ## Merge the base and top
        self.merge_shapes(circle_top)
        
        self.description = f'Tunnel2D of size (lx x lz) = ({lx} x {lz})'

def smooth_noise(shape, sigma=3, rng=np.random.default_rng()):
    noise = rng.random(shape) * 2 - 1  # random in [-1,1]
    noise_smooth = gaussian_filter(noise, sigma=sigma)
    # Normalize to [-1,1]
    noise_smooth = noise_smooth / np.max(np.abs(noise_smooth))
    return noise_smooth

#================USER DEFINED INTERFACE CREATOR EXAMPLE
class FBMInterfaceGen(mg2d.interface.rough_interface_generator.AbstractRoughInterfaceGenerator):
    """
    Generate rough interfaces using fractional Brownian motion (fBM).
    """
    from fbm import FBM  # Requires fbm only if user need to use FBMInterfaceGen #Lazyimport
    
    def __init__(self, H, length, method, generate_surface:bool, roughness_multipliers:list):
        """
        Initialize the fractional Brownian motion interface generator.

        Parameters
        ----------
        H : float
            Hurst exponent controlling roughness.
        length : float
            Total horizontal length of the interface.
        method : str
            fBM generation method supported by ''fbm.FBM''.
        generate_surface:bool
            Whether a surface interface is present.
        roughness_multipliers : array-like
            Scaling factor applied to each interface.
        """
        generator_params = {'H': H,
                            'length': length,
                            'method': method}
        super().__init__(generator_params, generate_surface, roughness_multipliers)
       
    def generate_rough_interfaces(self, discretized_interface2d_instance):
        """
        Generate rough interfaces using fractional Brownian motion.

        Parameters
        ----------
        discretized_interface2d_instance:DiscretizedInterfaces2D
            Initial DiscretizedInterfaces2D.

        Returns
        -------
        numpy.ndarray
            Interface elevation matrix of shape ''(nx, n_interfaces)''.
        """
        nx, _ = discretized_interface2d_instance.interfaces_matrix.shape
        n_soil_layers = discretized_interface2d_instance.n_soil_layers
        interfaces_matrix = np.zeros((nx, n_soil_layers))

        # rng = discretized_interface2d_instance.rng
        adj_roughness_multipliers = self.get_adjusted_roughness_multipliers(discretized_interface2d_instance, self.roughness_multipliers)

        H = self.generator_params['H']
        L = self.generator_params['length'] # *surface_scaling_factor While this gives approx scaling
        method = self.generator_params['method']
    
        n = nx - 1
        for j in range(n_soil_layers):
            scale = adj_roughness_multipliers[j]
            #generates n+1 data ie n increments
            rnd_layer = FBM(n=n, hurst=H, length=L, method=method).fbm() * scale  
            interfaces_matrix[:, j]= rnd_layer
        return interfaces_matrix, adj_roughness_multipliers
    
    