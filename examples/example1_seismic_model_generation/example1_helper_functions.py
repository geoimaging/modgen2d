import modgen2d as mg
import pandas as pd
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
        wet_mean_distribution = mg.random_generators.Uniform(wet_a, wet_b, rng)
        wet_prop = mg.PropertyDistribution(main_property_name, wet_mean_distribution, cov_distribution, stdev_type=cov_type)

        dry_mean_distribution = None
        if pd.isna(dry_a) or pd.isna(dry_b): 
            dry_prop = None
        else:
            dry_mean_distribution = mg.random_generators.Uniform(dry_a, dry_b, rng)
            dry_prop = mg.PropertyDistribution(main_property_name, dry_mean_distribution, cov_distribution, stdev_type=cov_type)
        
        main_property_instance.add_material_property_of_feature(feature_id, material_name, wet_prop, dry_prop)
    return main_property_instance

def add_layer0(rng, main_property_instance, main_property_name, air_val, water_val):
    material_name = 'layer0'
    feature_id = 'def'
    wet_mean_distribution = mg.random_generators.Constant(water_val, rng)
    wet_prop = mg.PropertyDistribution(main_property_name, wet_mean_distribution)

    dry_mean_distribution = mg.random_generators.Constant(air_val, rng)
    dry_prop = mg.PropertyDistribution(main_property_name, dry_mean_distribution)
    main_property_instance.add_material_property_of_feature(feature_id, material_name, wet_prop, dry_prop)
    return main_property_instance

def Vp_profile(Vs_profile, miu_profile):
    """
    Computes P-wave velocity (Vp) from S-wave velocity (Vs) and Poisson's ratio (μ)
    for element-wise 3D NumPy arrays.

    Formula:
        Vp = Vs * sqrt((1 / (1 - 2*μ)) + 1)
    """
    return Vs_profile * np.sqrt((1 / (1 - 2 * miu_profile)) + 1)
