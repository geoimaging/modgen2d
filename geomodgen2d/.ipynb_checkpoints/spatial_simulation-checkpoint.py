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

def spatial_profile_gen(theta_x, theta_z, lithologicalDomain_class, processed_property_dict=None, rnd_no=np.random.default_rng()):
    #if porcessed_property_dict is None: Then simulated profiles with mean 0 and standard dev 1.

    layer_mat = lithologicalDomain_class.layered_matrix
    xcoord = lithologicalDomain_class.x_ranges
    zcoord = lithologicalDomain_class.z_ranges
    
    vectorized_format = np.vectorize(f.format_value)
    layer_mat = vectorized_format(layer_mat)
    unique_layers = np.unique(layer_mat)
    # print(unique_layers)
    if processed_property_dict is not None:
        assert len(set(unique_layers)-set(processed_property_dict.keys())) == 0
        if len(set(processed_property_dict.keys())-set(unique_layers)) != 0:
            print(f"WARNING: Some extra ids in property set. Extras: ({set(processed_property_dict.keys())-set(unique_layers)})")

    # print(unique_layers, layer_mat)
    simulated_pd = pd.DataFrame()
    for layer_id in unique_layers:
        if processed_property_dict is not None:
            if layer_id in processed_property_dict.keys():
                a_m = processed_property_dict[layer_id][0]
                sigma = processed_property_dict[layer_id][1]
                # print(layer_id, a_m, sigma)
            else:
                raise ValueError(f"{layer_id} not in available keys: {processed_property_dict.keys()}")
        else:
            a_m = 0
            sigma = 1

        x_indices, z_indices = np.where(layer_mat == layer_id)
        # print(zcoord, z_indices)
        coordinates = [[z,x] for z,x in zip(zcoord[z_indices], xcoord[x_indices])]
        simulated_pd_each = cov_decom_matrix(a_m=a_m, b_m=0, sigma=sigma, theta_x=theta_x, theta_z=theta_z, points=coordinates, rnd_no=rnd_no, print_op=False)
        simulated_pd=pd.concat([simulated_pd,simulated_pd_each], ignore_index=True)
    simulated_2d = simulated_pd.pivot(index='x', columns='z', values='simulated_X')
    # print(simulated_2d.to_numpy().shape)
    
    assert (len(simulated_pd) - layer_mat.shape[0]*layer_mat.shape[1]) == 0, f"ERROR: ALL coordinates are not covered. {(len(simulated_pd))} - {(layer_mat.shape[0]*layer_mat.shape[1])} = {(len(simulated_pd) -  layer_mat.shape[0]*layer_mat.shape[1])} remaining"
    return simulated_2d.to_numpy()
    
def constant_field(a_m, b_m, points, rnd_no):
    df = pd.DataFrame(points, columns = ['z', 'x'])
    df['simulated_X'] = a_m+b_m*df['z'] # Mean Vector: m(y,z) = a_m + b_m*z
    return df
