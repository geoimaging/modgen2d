import modgen2d as mg2d
import numpy as np
from scipy.stats import multivariate_normal, norm

def add_features_from_pd(rng, ec_main_property_instance, dc_main_property_instance, corr_coeff, feature_id, pd_dataframe, cov_distribution = None, cov_type='cov'):
    
    # Get col_name for relevant limit values for both property X (ec) and Y (dc)
    X_wet_a_colname, X_wet_b_colname = f"ec_wet_a", f"ec_wet_b" 
    X_dry_a_colname, X_dry_b_colname = f"ec_dry_a", f"ec_dry_b" 
    Y_wet_a_colname, Y_wet_b_colname = f"dc_wet_a", f"dc_wet_b" 
    Y_dry_a_colname, Y_dry_b_colname = f"dc_dry_a", f"dc_dry_b" 

    # Running the loop and adding property of each material for that feature
    for material_name in pd_dataframe.index.tolist():
        # print(material_name)

        # 
        # NOTE: This generator takes properties of both main_properties (ec, and dc), and produces two values (Ec and dc) in a single go, unlike default random generators.
        # Hence, additional post-processing is needed to adjust to format later.
        
        X_wet_a, X_wet_b = pd_dataframe[X_wet_a_colname].loc[material_name], pd_dataframe[X_wet_b_colname].loc[material_name]
        X_dry_a, X_dry_b = pd_dataframe[X_dry_a_colname].loc[material_name], pd_dataframe[X_dry_b_colname].loc[material_name]
        
        Y_wet_a, Y_wet_b = pd_dataframe[X_wet_a_colname].loc[material_name], pd_dataframe[X_wet_b_colname].loc[material_name]
        Y_dry_a, Y_dry_b = pd_dataframe[X_dry_a_colname].loc[material_name], pd_dataframe[X_dry_b_colname].loc[material_name]

        assert not pd.isna(wet_a), f"wet_a for {material_name} must be a number"
        assert not pd.isna(wet_b), f"wet_a for {material_name} must be a number"
        # wet_mean_distribution = mg2d.random_generators.Uniform(wet_a, wet_b, rng)
        wet_mean_distribution = hCorrelatedUniformBivariateXLogY(X_wet_a, X_wet_b, Y_wet_a, Y_wet_b, corr_coeff, rng)  
        wet_prop = mg2d.PropertyDistribution(main_property_name, wet_mean_distribution, cov_distribution, stdev_type=cov_type)

        dry_mean_distribution = None
        if pd.isna(X_dry_a) or pd.isna(X_dry_b) or pd.isna(Y_dry_b) or pd.isna(Y_dry_b): 
            dry_prop = None
        else:
            # dry_mean_distribution = mg2d.random_generators.Uniform(dry_a, dry_b, rng)
            dry_mean_distribution = CorrelatedUniformBivariateXLogY(X_dry_a, X_dry_b, Y_dry_a, Y_dry_b, corr_coeff, rng)
            dry_prop = mg2d.PropertyDistribution(main_property_name, dry_mean_distribution, cov_distribution, stdev_type=cov_type)
        
        ec_main_property_instance.add_material_property_of_feature(feature_id, material_name, wet_prop, dry_prop)


        ### Repeat for 'dc'
        ## But since 'dc' is already generated from the bivariate random generator above, so this is defined for a placeholder in sampled properties only.
        # Note: right now, ec_main_property_instance will generate ['ec', 'dc'], but we need to post-process to have 'ec' and 'dc' on relevant properties. So, need a placeholder.

        wet_mean_distribution = mg2d.random_generators.Constant(-999, rng)  # Filler to be replaced later
        wet_prop = mg2d.PropertyDistribution(main_property_name, wet_mean_distribution, cov_distribution, stdev_type=cov_type)
        dry_mean_distribution = None
        if pd.isna(X_dry_a) or pd.isna(X_dry_b) or pd.isna(Y_dry_b) or pd.isna(Y_dry_b): 
            dry_prop = None
        else:
            # dry_mean_distribution = mg2d.random_generators.Uniform(dry_a, dry_b, rng)
            dry_mean_distribution = mg2d.random_generators.Constant(-999, rng)  # Filler to be replaced later
            dry_prop = mg2d.PropertyDistribution(main_property_name, dry_mean_distribution, cov_distribution, stdev_type=cov_type)
        
        dc_main_property_instance.add_material_property_of_feature(feature_id, material_name, wet_prop, dry_prop)

    return ec_main_property_instance, dc_main_property_instance

#================USER DEFINED RANDOM GENERATORS
class CorrelatedUniformBivariateXLogY(mg2d.random_generators.RandomGeneratorAbstract):
    def __init__(self, u_low_x, u_high_x, u_low_logy, u_high_logy, r_x_logy, rng=None):
        """
        Initializes a correlated uniform random generator using a Gaussian copula.

        X  ~ Uniform(u_low_x, u_high_x)
        log10(Y) ~ Uniform(log10(u_low_logy), log10(u_high_logy))
        Corr(X, log10(Y)) = r_x_logy
        
        Parameters
            u_low_x, u_high_x : float :  Range for Variable1.
            u_low_logy, u_high_logy : float : Range for Variable2.
            r : float : Correlation coefficient between Variable 1 and Variable 2.
            r_comp_axis: string : Axes scale for correlation computation.
                First element must be in {"x", "log10x"}, second in {"y", "log10y"}.
                That is Options: ['x', 'y'], ['log10x', 'log10y'], ['log10x', 'y'], ['x', 'log10y']]
        """
        self.u_low_x = u_low_x
        self.u_high_x = u_high_x
        self.u_low_logy = u_low_logy
        self.u_high_logy = u_high_logy
        
        # --- means checks ---
        if u_low_x > u_high_x:
            raise ValueError("u_low_x must be < u_high_x")
        if u_low_logy <= 0 or u_high_logy <= 0:
            raise ValueError("Log-uniform variable must be strictly positive.")
        if u_low_logy > u_high_logy:
            raise ValueError("u_low_logy must be < u_high_logy")

        # --- correlation checks ---
        if not isinstance(r_x_logy, (int, float)):
            raise TypeError("Correlation coefficient r must be a float or int.")
        if not -1 <= r_x_logy <= 1:
            raise ValueError(f"Correlation coefficient r must be between -1 and 1, got {r_x_logy}")
            
        self.r_x_logy = r_x_logy

    def generate(self, size=1):
        if isinstance(size, int):
            n = size
            out_shape = (size, 2)
        else:
            raise TypeError("size must be int")
            
        cov = [[1, self.r_x_logy], [self.r_x_logy, 1]]
        z = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=size)
        u = norm.cdf(z)
        u = u.reshape(-1, 2)
        
        x = self.u_low_x + (self.u_high_x - self.u_low_x) * u[:, 0]
        logy = np.log10(self.u_low_logy) + (np.log10(self.u_high_logy) - np.log10(self.u_low_logy)) * u[:, 1]
        y = 10 ** logy
       
        samples = np.column_stack((x, y))
        return samples.reshape(out_shape)

class Discrete2ContinuousPDF(mg2d.random_generators.DiscreteChoice):
    """
    Convert a discrete probability distribution into a continuous-like PDF
    using linear interpolation.
    """
    def __init__(self, x, p, new_del_x:float, new_x_min:float=None, new_x_max:float=None, incr_x_init:float=0, rng=None):
        """
        Refine a discrete PDF into a continuous approximation by linear interpolation at each of new_x_array.

        Parameters
        ----------
        x : array-like
            Discrete support values (numeric only).
        p : array-like
            Probabilities associated with ``x`` (must sum to 1). If provided empty list; then uniform distribution over all x
        new_del_x : float
            Step size for the refined continuous grid.
        new_x_min : float, optional
            Minimum bound of the refined grid.
        new_x_max : float, optional
            Maximum bound of the refined grid.
        incr_x_init : float, optional
            Constant shift applied to ``x`` before interpolation.
        rng : numpy.random.Generator, optional
            Random number generator.
        refines discrete pdf to more of continuous one by linear interpolation at each of new_x_array
        
        Example
        -------
        x = [0,1,5]; p = [0, 0.2, 0.4] -> means p(0) = 0; p(1) = 0.2; p[5]=0.4
           
        if new_del_x = 0.5
        then, {0:0, 0.5:0.1, 1:0.2, 1.5:0.225, 2:0.25 .... ,5:0.4}. i.e. linear interpolation in between
        then, pdf_values readjusted such that sum is 1.
        """
        # Convert to numpy array if not already
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float) if p is not None else np.ones_like(x) / len(x)
        assert not np.isnan(p).any(), "p contains NaN values"
        
        # Ensure both x and p are 1D
        if x.ndim != 1 or p.ndim != 1:
            raise ValueError("Both x and p must be 1D lists or 1D numpy arrays.")

        # Ensure all elements in `x` are numbers only
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("All elements in x must be numeric (int or float).")

        # Apply shift to x
        x = x+incr_x_init

        if new_x_max is None:
            new_x_max = np.max(x)
        if new_x_min is None:
            new_x_min = np.min(x)

        # Ensure x_min ≤ min(x) and x_max ≥ max(x)
        if new_x_min > np.min(x):
            raise ValueError(f"x_min ({new_x_min}) must be less than or equal to the minimum value in shifted x ({x}).")
        if new_x_max < np.max(x):
            raise ValueError(f"x_max ({new_x_max}) must be greater than or equal to the maximum value in shifted x ({x}).")

        new_x_array = np.arange(new_x_min, new_x_max + new_del_x, new_del_x)
        new_p = np.interp(new_x_array, x, p)
        
        # Normalize probabilities so they sum to 1
        new_p = new_p / np.sum(new_p)
        
        super().__init__(x = new_x_array, p = new_p, rng=rng)
