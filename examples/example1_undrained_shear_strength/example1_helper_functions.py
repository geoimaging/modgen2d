import modgen2d as mg2d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
