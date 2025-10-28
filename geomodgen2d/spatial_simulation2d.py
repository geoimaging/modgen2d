"""All functions to perform spatial simulation"""
import numpy as np
import pandas as pd
import geomodgen2d.general_functions as f

## All in One (Detailed demo in Random_Field_simulation_v1 and v2::)
def cov_decom_matrix(a_m, b_m, sigma, theta_x, theta_z, points, rnd_no=np.random.default_rng(), print_op=False):
    # covariance decomposition matrix
    """
    # mean_f = a_m+b_m*z :: Note here x -> Vs; and z-> depth
    # sigma_f = sigma
    # theta_x, theta_z = settings for spatial random field generation: relates the correlation in x and z direction
    # points = lists of points (x,z coordinates) (n*2 array)
    # rnd_no = random_number generator with seed
    # Print_op = if op to be printed
    #
    """
    df = pd.DataFrame(points, columns = ['z', 'x'])
        
    # Calculating mean vector
    df['mean_f'] = a_m+b_m*df['z'] # Mean Vector: m(x,z) = a_m + b_m*z
    
    
    if sigma==0:
        df['simulated_f'] = df['mean_f']
        return df
    
    # Calculating covariance matrix and correlation matrix
    cov_mat = pd.DataFrame(np.zeros([df.shape[0], df.shape[0]]))
    cov_mat = pd.DataFrame(np.exp(-2 * np.abs(df['x'].values[:, np.newaxis] - df['x'].values) / theta_x) * np.exp(-2 * np.abs(df['z'].values[:, np.newaxis] - df['z'].values) / theta_z), columns=df.index, index=df.index)
    corr_mat = cov_mat * sigma**2
    
    # Transformation matrix
    L = pd.DataFrame(np.linalg.cholesky(corr_mat))  # Cholesky Decomposition: E_xx (corr_matrix) = L*L_transpose:: Transformation Matrix
    
    # Simulation Vector
    df['simulated'] = rnd_no.normal(size = df.shape[0])  #Generate random normally distributed numbers

    # Simulated random field
    df['simulated_f'] = np.matmul(L, df['simulated'])+df['mean_f']   #simulated_vector = L * simulated_rand + Mean
    df = df.drop(['simulated', 'mean_f'], axis=1)
    
    if print_op:
        print("Correlation_Matrix: First five rows\n")
        print(corr_mat.head())
        print("\n df_variable")
        print(df.head())
    return df

def spatial_profile_gen(theta_x, theta_z, lithologicalDomain_class, gwt_depth=None, rnd_no=np.random.default_rng(), processed_property_dict=None, ignore_lithological_ids=[]):
    """
    Generates a 2D spatially correlated random field based on a layered matrix representation.

    Parameters:
    theta_x, theta_y : float
        Correlation lengths in x and y directions.
    lithological_domain2d_instance : object
        An instance containing the 2D layered matrix and coordinate ranges.
    gwt_depth : float, optional
        Ground water table depth (used for wet/dry classification).
    rnd_no : numpy.random.Generator
        Random number generator.
    processed_property_dict : dict, optional
        Dictionary mapping layer IDs to their mean and stddev (wet/dry or both) properties.
    ignore_lithological_ids : list
        List of lithological IDs to ignore. These will be assigned -99999.

    Returns:
    numpy.ndarray
        A 2D array representing the simulated spatially correlated random field.
    """
    
    #if porcessed_property_dict is None: Then simulated profiles with mean 0 and standard dev 1.

    layer_mat = lithologicalDomain_class.layered_matrix
    xcoord = lithologicalDomain_class.x_ranges
    zcoord = lithologicalDomain_class.z_ranges
    
    gwt_z, _ = np.meshgrid(zcoord, xcoord, indexing='ij')
    
    if gwt_depth is None:
        gwt_z = np.zeros_like(layer_mat, dtype=bool)
    else:
        gwt_z = gwt_z >= gwt_depth  # Use y or z based on your case
        
    assert gwt_z.shape == layer_mat.shape, "The shapes does not match. Check x_ranges, z_ranges, layer_mat of lithological_domain2d_instance."         
    
    vectorized_format = np.vectorize(f.format_value)
    layer_mat = vectorized_format(layer_mat)
    unique_layers = np.unique(layer_mat)
    # print(unique_layers)
    
    # Validate processed_property_dict
    if processed_property_dict is not None:
        processed_property_dict = validate_processed_property_dict(processed_property_dict)
        processed_unique_layers = set(unique_layers) - set(ignore_lithological_ids)
        if not set(processed_unique_layers).issubset(processed_property_dict.keys()):
            missing = set(processed_unique_layers) - set(processed_property_dict.keys())
            raise ValueError(f"Missing keys in processed_property_dict: {missing}")
        assert not (set(processed_property_dict.keys()) & set(ignore_lithological_ids)), \
            f"Keys {set(processed_property_dict.keys()) & set(ignore_lithological_ids)} are in ignore list!"

        if len(set(processed_property_dict.keys())-set(processed_unique_layers)) != 0:
            print(f"WARNING: Some extra ids in property set. Extras: ({set(processed_property_dict.keys())-set(unique_layers)})")

    # print(unique_layers, layer_mat)
    simulated_pd = pd.DataFrame()
    z_val2param = False
    
    for layer_id in unique_layers:
        z_idx, x_idx = np.where(layer_mat == layer_id)
        if z_idx.size == 0:
            continue
            # skip the remaining step if no matches.
        if layer_id in ignore_lithological_ids:
            a_m, b_m, sigma = -99999, 0, 0
        elif processed_property_dict is not None:
            z_val2param = True
            
            if layer_id not in processed_property_dict.keys():
                raise ValueError(f"{layer_id} not in available keys: {processed_property_dict.keys()}")
            if check_for_zero_sigma(processed_property_dict[layer_id]):
                a_m, b_m, sigma = 0, 0, 0
            else:
                a_m, b_m, sigma = 0, 0, 1
        else:
            a_m, b_m, sigma = 0, 0, 1
            
        print(f"Simulating z-vals for Layer ID: {layer_id}")
        
        coordinates = [[z,x] for z,x in zip(zcoord[z_idx], xcoord[x_idx])]
        simulated_pd_each = cov_decom_matrix(a_m=a_m, b_m=b_m, sigma=sigma, theta_x=theta_x, theta_z=theta_z, points=coordinates, rnd_no=rnd_no, print_op=False)

        if z_val2param and layer_id not in ignore_lithological_ids:
            if 'both' in processed_property_dict[layer_id].keys():
                a_m = processed_property_dict[layer_id]['both']['mean']  
                b_m = processed_property_dict[layer_id]['both']['mean_bm']
                sigma = processed_property_dict[layer_id]['both']['stdev/cov']
                if processed_property_dict[layer_id]['both']['stdev_type'] == 'cov':
                    sigma*=a_m
                    
                simulated_pd_each['simulated_f'] = (a_m + b_m*simulated_pd_each['z']) + simulated_pd_each['simulated_f'] * sigma
            else:
                gwt_z = simulated_pd_each['z'].to_numpy()
                if gwt_depth is None:
                    gwt_z = np.zeros_like(gwt_z, dtype=bool)
                else:
                    gwt_z = gwt_z>=gwt_depth
                    
                a_m = processed_property_dict[layer_id]['wet']['mean']  
                b_m = processed_property_dict[layer_id]['wet']['mean_bm']
                sigma = processed_property_dict[layer_id]['wet']['stdev/cov']
                if processed_property_dict[layer_id]['wet']['stdev_type'] == 'cov':
                    sigma*=a_m
                
                a_m_dry = processed_property_dict[layer_id]['dry']['mean']  
                b_m_dry = processed_property_dict[layer_id]['dry']['mean_bm']
                sigma_dry = processed_property_dict[layer_id]['dry']['stdev/cov']
                if processed_property_dict[layer_id]['dry']['stdev_type'] == 'cov':
                    sigma_dry*=a_m_dry

                df_z = pd.DataFrame()
                df_z['a_m'] = np.where(gwt_z, a_m, a_m_dry)
                df_z['b_m'] = np.where(gwt_z, b_m, b_m_dry)
                df_z['sigma'] = np.where(gwt_z, sigma, sigma_dry)
                    
                simulated_pd_each['simulated_f'] = (df_z['a_m'] + df_z['b_m']*simulated_pd_each['z']) + simulated_pd_each['simulated_f'] * df_z['sigma']

        simulated_pd=pd.concat([simulated_pd,simulated_pd_each], ignore_index=True)
    simulated_2d = simulated_pd.pivot(index='z', columns='x', values='simulated_f')
    # print(simulated_2d.to_numpy().shape)
    assert (len(simulated_pd) - layer_mat.shape[0]*layer_mat.shape[1]) == 0, f"ERROR: ALL coordinates are not covered. {(len(simulated_pd))} - {(layer_mat.shape[0]*layer_mat.shape[1])} = {(len(simulated_pd) -  layer_mat.shape[0]*layer_mat.shape[1])} remaining"
    return simulated_2d.to_numpy()

def non_spatial_simulation(lithological_domain2d_instance, gwt_depth, processed_property_dict, ignore_lithological_ids=[], prof_type='mean', warn_non_zero_stdev=True):
    """
    Generates a non-spatial profile for a given layer matrix. Note: no spatial correlation, just replacing the values.

    Args:
    lithological_domain2d_instance : object
        An instance containing the layered matrix and coordinate ranges.
    gwt_depth:
        ground water depth value. (Used for wet and dry)
    processed_property_dict : dict
        A dictionary of processed properties.
        {
        'layer_ID': {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},
        'layer_ID2': {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},
        'layer_ID3': {'both': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},
         ....
        
    ignore_lithological_ids: list
        Lithological ids to ignore during simulation. All values at these ids will have value -99999. 
    prof_type : str, optional
        The type of profile to generate, either 'mean' or 'stdev/cov'. Default is 'mean'.
    warn_non_zero_stdev: bool, optional
        Show warning if non-zero standard deviation for non-spatial simulation
        
    Returns:
    np.ndarray
        A profile array based on the specified property type.
        
    ## Note: Here, the run is done twice with wet/dry even for both cases. Should not be a problem, as we just replaces the vals.
    """
    assert prof_type in ['mean', 'stdev/cov'], f"prof_type must be either 'mean' or 'stdev/cov'. Provided {prof_type}"
    layer_matrix = lithological_domain2d_instance.layered_matrix
    x_coord = lithological_domain2d_instance.x_ranges
    z_coord = lithological_domain2d_instance.z_ranges
    
    new_layer_matrix = np.empty_like(layer_matrix, dtype='<U20')
    vectorized_format = np.vectorize(f.format_value)
    layer_matrix = vectorized_format(layer_matrix)
    unique_layers = np.unique(layer_matrix)
    
    gwt_z, _ = np.meshgrid(z_coord, x_coord, indexing='ij')
    
    if gwt_depth is None:
        gwt_z = np.zeros_like(layer_matrix, dtype=bool)
    else:
        gwt_z = gwt_z >= gwt_depth  # Use y or z based on your case
        
    assert gwt_z.shape == layer_matrix.shape, "The shapes does not match. Check x_ranges, z_ranges, layer_mat of lithological_domain2d_instance."         
    
    # Validate processed_property_dict if provided
    processed_property_dict = validate_processed_property_dict(processed_property_dict)
    processed_unique_layers = set(unique_layers) - set(ignore_lithological_ids)
    # Check if processed_property_dict covers all unique layers.
    if not set(processed_unique_layers).issubset(processed_property_dict.keys()):
        missing_keys = set(processed_unique_layers) - set(processed_property_dict.keys())
        raise ValueError(f"Missing keys in processed_property_dict: {missing_keys}")

    extra_keys = set(processed_property_dict.keys()) - set(processed_unique_layers)
    if extra_keys:
        print(f"WARNING: Some extra IDs in processed_property_dict. Extras: {extra_keys}")

    if len(set(processed_property_dict.keys())-set(processed_unique_layers)) != 0:
        print(f"WARNING: Some extra ids in property set. Extras: ({set(processed_property_dict.keys())-set(unique_layers)})")

    replace_list = []
    replace_with = []
    for layer_id in unique_layers:
        # Extract coordinates for the given layer
        for gwt_condn in ['wet', 'dry']:
            gwt_idx = "T" if gwt_condn == 'wet' else "F"
            
            # x_indices, z_indices = np.where(layer_mat == layer_id)
            z_idx, x_idx = np.where((layer_matrix == layer_id) & (gwt_z == (gwt_idx=="T")))
            
            new_id = f'{layer_id}_{gwt_idx}'
            new_layer_matrix[z_idx, x_idx] = new_id
            # print(layer_matrix)
            if z_idx.size == 0:
                continue
                # skip the remaining step if no matches.
                
            # Assign mean and standard deviation based on processed_property_dict
            if layer_id in ignore_lithological_ids:
                a_m = -99999
            elif layer_id in processed_property_dict.keys():
                if 'both' in processed_property_dict[layer_id].keys():
                    gwt_key = 'both'
                else:
                    gwt_key = gwt_condn

                a_m = processed_property_dict[layer_id][gwt_key][prof_type]  
                if processed_property_dict[layer_id][gwt_key]['stdev_type'] == 'cov' and prof_type == 'stdev/cov':
                    a_m*=processed_property_dict[layer_id][gwt_key]['mean']
                            
                if warn_non_zero_stdev:
                    sigma = processed_property_dict[layer_id][gwt_key]['stdev/cov']  
                    if sigma!=0:
                        print(f"Warning: {layer_id}:{gwt_key} has non-zero stdev/cov - {sigma}, but no spatial correlation is used in the simulation.")
            else:
                raise ValueError(f"{layer_id} not in available keys: {processed_property_dict.keys()} or in ignore_lithological_ids: {ignore_lithological_ids}")
            replace_list.append(new_id)
            replace_with.append(a_m)
    assert '' not in np.unique(new_layer_matrix), "Not all layer_id covered for gwt_condition and value determination."
    simulated_profile = f.replace_vals_in_array(new_layer_matrix, replace_list = replace_list, replace_with = replace_with)
    return simulated_profile

def constant_field(a_m, b_m, points, rnd_no):
    df = pd.DataFrame(points, columns = ['z', 'x'])
    df['simulated_f'] = a_m+b_m*df['z'] # Mean Vector: m(y,z) = a_m + b_m*z
    return df

def check_for_zero_sigma(processed_property_dict_layer_id):
    """
    Check if the layer_id has zero stdev/cov value in all its possible case.
        {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}, 
         'dry': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},
    Or,
        {'both': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},                      

    Parameters:
    processed_property_dict : dict
        Dictionary wet/dry/both and their corresponding mean and standard deviation values as nested dictionaries.
    
    Returns:
    True if zero stdev/cov in all cases, else False
    """
    
    # Validate the dictionary through validate_processed_property_dict
    keys = set(processed_property_dict_layer_id.keys())
    optional_key = 'layer0_air'
    keys.discard(optional_key)

    valid_sets = [{'wet', 'dry'}, {'both'}]
    assert keys in valid_sets, f"Invalid key combination: {keys}"   ## Though already in more detail validated using validate_processed_property_dict
     
    for k in keys:
        stdev_val = processed_property_dict_layer_id[k]['stdev/cov']
        if stdev_val!=0:
            return False
    return True

def validate_processed_property_dict(processed_property_dict):
    """
    Validates if the given dictionary follows the required format:
    
    {
        'layer_ID': {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},
        'layer_ID2': {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},
        'layer_ID3': {'both': {'mean': float, 'mean_bm': float (optional), 'stdev/cov': float, 'stdev_type':string}},                      
        ...
    }

    Parameters:
    property_dict : dict
        Dictionary containing layer IDs as keys and their corresponding mean and standard deviation values as nested dictionaries.

    Returns:
    processed_property dict; adjusted if mean_bm (optional) is not provided.
    """
    if processed_property_dict is not None:
        assert isinstance(processed_property_dict, dict), "Input must be a dictionary."
        
        for key, value in processed_property_dict.items():
            assert isinstance(value, dict), f"Value for key '{key}' must be a dictionary."
            
            # Assert that layer_ID contains either 'wet' and 'dry' or 'both'
            assert ('wet' in value and 'dry' in value) or 'both' in value, f"Layer '{key}' must contain both 'wet' and 'dry', or 'both'."

            # Check for wet, dry, or all layer types
            for subkey, subvalue in value.items():
                if subkey!= 'layer0_air':
                    assert subkey in ['wet', 'dry', 'both'], f"Subkey '{subkey}' for key '{key}' is invalid. Expected 'wet', 'dry', or 'both'."
                    assert isinstance(subvalue, dict), f"Subvalue for '{subkey}' in layer '{key}' must be a dictionary."
                    
                    # Validate mean and stdev for each layer type
                    assert 'mean' in subvalue, f"'{subkey}' layer for key '{key}' must contain 'mean'."
                    assert 'stdev/cov' in subvalue, f"'{subkey}' layer for key '{key}' must contain 'stdev/cov'."
                    assert 'stdev_type' in subvalue, f"'{subkey}' layer for key '{key}' must contain 'stdev_type'."
                    assert isinstance(subvalue['mean'], (int, float)), f"'mean' for '{subkey}' in layer '{key}' must be a number."
                    assert isinstance(subvalue['stdev/cov'], (int, float)), f"'stdev/cov' for '{subkey}' in layer '{key}' must be a number."
                    assert subvalue['stdev_type'] in ['stdev', 'cov'], "stdev_type must be either 'stdev', or 'cov'."
                    
                    # 'mean_bm' is optional, but if present, must be a number
                    if 'mean_bm' in subvalue:
                        assert isinstance(subvalue['mean_bm'], (int, float)), f"'mean_bm' for '{subkey}' in layer '{key}' must be a number (if provided)."
                    else:
                        processed_property_dict[key][subkey]['mean_bm'] = 0.0

    return processed_property_dict
