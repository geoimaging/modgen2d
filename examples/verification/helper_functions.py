import modgen2d as mg2d
import modgen2d.general_functions as f
import numpy as np

def get_properties_and_lit_domain(n_layers, x_span, z_span, gwt_depth, rng=None, plot=False):
    
    # Step 1: Length Configuration
    length_config = mg2d.LengthConfig("m", max_grid_density=100) 
   
    # Step 2: Properties Definition
    # Grid spacing
    del_xz_interface = 1       # Base spacing for the interfaces (Note: for Unifrom, too small dx might make interface flat)
    del_xz_final = 0.2       # Base spacing for the domain for obstacles and final generated model.

    domain_interface = mg2d.DiscretizedDomain2D(x_span, z_span, del_xz_interface, del_xz_interface, length_config)
    domain_final = mg2d.DiscretizedDomain2D(x_span, z_span, del_xz_final, del_xz_final, length_config)

    # Interface interpolation method
    remesh_interp_method = 'linear'
    
    # Random number generator (for reproducibility)
    if rng is None:
        rng = np.random.default_rng(seed=42)
    
    # Initialize feature configuration
    feature_config_instance =  mg2d.FeaturesConfig()
    
    # Define material distributions for this simple example
    soil_materials_distribution = mg2d.random_generators.Constant(val = 'soil', rng=rng)
    
    # Add features to feature_config_instance
    feature_config_instance.add_feature('def', soil_materials_distribution, feature_description = 'def means soil.')
    
    #2.3.1 Main Properties config definition
    main_properties_config_instance =  mg2d.MainPropertiesConfig(feature_config_instance, layer0_flag=True)
    
    #2.3.2 Define each MainProperty instance
    main_property_name = 'vs'
    property_desc = 'shear wave velocity'
    main_property_instance = mg2d.MainProperty(main_property_name, feature_config_instance, layer0_flag=True, description=property_desc)
    
    #2.3.3  Define wet and dry properties for each features' each materials (including layer 0  for 'def' if flag is True)
    ## For Feature 'def'; material type 'soil'
    wet_mean_distribution = mg2d.random_generators.Uniform(80, 900, rng)  
    cov_distribution = mg2d.random_generators.Constant(0.05, rng)
    cov_type = 'cov'
    wet_prop = mg2d.PropertyDistribution(main_property_name,  wet_mean_distribution, cov_distribution, stdev_type=cov_type)
    
    dry_prop = None     # If None, GWT does not matter for this material's property (of interest).
    main_property_instance.add_material_property_of_feature(feature_id='def', material_name='soil', #Must match feature_id and material name (as defined in features_config)
                                                            property_distribution_instance=wet_prop,  
                                                            property_distribution_instance_if_dry=dry_prop 
                                                           )
    
    ## For Feature 'def'; material type 'layer0'
    wet_mean_distribution = mg2d.random_generators.Constant(10, rng)  
    cov_distribution = None  # None means 'cov' is 0 (i.e. no spatial correlation)
    cov_type = 'cov'
    wet_prop = mg2d.PropertyDistribution(main_property_name,  wet_mean_distribution, cov_distribution, stdev_type=cov_type)
    
    dry_prop = None
    main_property_instance.add_material_property_of_feature(feature_id='def', material_name='layer0', 
                                                            property_distribution_instance=wet_prop,
                                                            property_distribution_instance_if_dry=dry_prop)
    
    # 2.3.4 Add MainProperty to MainPropertiesConfig instance
    main_properties_config_instance.add_main_property(main_property_instance)
    # main_properties_config_instance.print()

    
    # Step 3: Define Soil Interfaces and Lithological Domain
    has_surface = False
    surface_factor = 1.5 if has_surface else 0
    
    # If number of layers > length of list, last value is reused
    roughness_multipliers = [surface_factor,1.3,1.2,1]
    
    interface_sett= {
        'generate_surface':has_surface,  # Generate ground surface
    
        # Parameters for step 1: Generation of rough interfaces
        'rough_interface_generator_instance':mg2d.interface.rough_interface_generator.UniformInterfaceGen(1, has_surface, roughness_multipliers),
    
        # Parameters for step 2: Filtering
        'savgol2d_smoother_settings': {
                     'filter_window_length':21, # must be odd
                     'filter_polyorder':7,
                            },
    
        # Parameter for Step 3: Interface Initial Points Generation
        'interfaces_depths_updater':'random',  # Can be 'random', 'equidistant', or np.ndarray (skips zs generation.). Default: 'random',
        'interfaces_depth_reference_point_x':None,  # Default: None,
        
        # Parameter for Step 4: Handling the overlapping and adjust surface_top_to_zero
        'overlapping_resolver_technique': 'erosion', # Options: 'erosion', 'reverse_erosion'. Default: 'erosion'
        'adjust_surface_top_to_zero': True,  # Default: True
        }
    
    # DiscretizedInterfaces2D from dictionary definition
    soil_interface = mg2d.interface.DiscretizedInterfaces2DFromDict(domain_interface, n_layers, interface_sett, 
                                                                    remesh_interp_method=remesh_interp_method, rng=rng)
    # soil_interface.plot()
    
    # Reset global soil interface configuration (safety step)
    mg2d.GlobalSoilInterfaceConfig.reset()   # For safety only
    mg2d.GlobalSoilInterfaceConfig.set_soil_interface(soil_interface)
    
    ## Get lithological domain from interface
    name = 'soil_lit'
    lit = mg2d.LithologicalDomain2D(domain_final, gwt_depth, name)

    if plot:
        lit.plot(discrete_point_size = 10, plot_interfaces=True)
    
    # Initialize lithological domain collection
    lit_collection = mg2d.LithologicalDomain2DCollection(main_properties_config_instance.get_feature_ids(), interface_set_name="soil") 
    
    # Add soil-based lithological domain
    lit_collection.add_lithological_domain_from_soil_interface_config(lit)
    
    # Finalize and lock the lithological domain collection
    lit_collection.lock()

    return main_properties_config_instance, lit_collection