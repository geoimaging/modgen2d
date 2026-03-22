import modgen2d as mg2d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_features_from_pd(pd_dataframe, main_property_instance, main_property_name, feature_id, rng, cov_distribution = None, cov_type='cov'):
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

def Vp_profile(Vs_profile, miu_profile):
    """
    Computes P-wave velocity (Vp) from S-wave velocity (Vs) and Poisson's ratio (μ)
    for element-wise 3D NumPy arrays.

    Formula:
        Vp = Vs * sqrt((1 / (1 - 2*μ)) + 1)
    """
    return Vs_profile * np.sqrt((1 / (1 - 2 * miu_profile)) + 1)

def plot_profile(gen_profile_like, main_property_like, new_simulated_profile, ax=None, discrete_point_size=0, plot_gwt = True,
               vlog = False, vmin=None, vmax=None, cmap='gist_earth_r', 
               title = 'auto', legend = True, legend_label = None, legendkwargs_dict={}):
    """
    Plots a 2D property profile.

    Parameters
    ----------
    gen_profile_like:
        replicates domain and properties shape from gen_profile_like
    main_property_like : str
        replicates simulated profile shape from this main_property name.
    new_simulated_profile:
        array of same shape as main_property_like, to plot.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on.
    discrete_point_size : float, default 0
        Size of scatter points.
    plot_gwt : bool, default True
        Plot groundwater table.
    vlog : bool, default False
        Apply logarithmic normalization.
    vmin, vmax : float, optional
        Color scale limits.
    cmap : str or Colormap, default 'gist_earth_r'
        Colormap.
    title : str, default 'auto'
        Plot title.
    legend : bool, default True
        Show colorbar.
    legend_label : str, optional
        Colorbar label.
    legendkwargs_dict : dict, optional
        Extra keyword arguments for colorbar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    vmin : float
        Minimum value used for colormap.
    vmax : float
        Maximum value used for colormap.
    """
    if ax is None:
        fig,ax = plt.subplots()

    simulated_profile_set = gen_profile_like.simulated_profiles
    domain = gen_profile_like.lit_domain.domain
    z_centers, x_centers = domain.z_centers, domain.x_centers
    span_x, span_z = domain.spans
    
    if main_property_like not in simulated_profile_set.keys():
        raise ValueError(f"main_property_name: {main_property_like} not generated yet.")
    
    data = simulated_profile_set[main_property_like]

    if data.shape != new_simulated_profile.shape:
        raise ValueError(f"Shape mismatch. Expected: {data.shape}. Got {new_simulated_profile.shape}")

    data = new_simulated_profile.T
    extent=[0, span_x, span_z, 0]
    # Create a colormap from the color mapping
    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)
        
    if vlog:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cax = ax.imshow(data, norm=norm, cmap=cmap, extent=extent, interpolation='none') 
    else:
        cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='none') 
    
    # Plot gwt
    if gen_profile_like.lit_domain.gwt_depth is not None and plot_gwt:
        edges_kw = dict(color='r', linestyle='dashed', linewidth=2, zorder=4000)
        ax.plot([0, span_x], [gen_profile_like.lit_domain.gwt_depth, gen_profile_like.lit_domain.gwt_depth], **edges_kw)

    x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
    if discrete_point_size!=0:
        ax.scatter(x_data.flatten(), z_data.flatten(), c = 'k', s=discrete_point_size)
    
    if title=='auto':
        ax.set_title(f"Main_property_name:{main_property_like}")
    
    # Colorbar
    # if legendkwargs_dict is None:
    #     legendkwargs_dict = {
    #         'shrink':0.6,
    #         'aspect':20,
    #         'pad':0.1
    #     }
    
    if legend:
        cbar = plt.colorbar(cax, ax=ax, **legendkwargs_dict)
        cbar.set_label(legend_label)
    
    ax.axis('scaled')
    ax.set(
        xlim= [0, span_x],
        ylim= [span_z, 0],
        xlabel='X',
        ylabel='Z',
    )

    return ax, vmin, vmax
