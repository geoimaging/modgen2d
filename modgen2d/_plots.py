import matplotlib.pyplot as plt
import numpy as np
import warnings
from .discretized_domain2d import DiscretizedDomain2D
import matplotlib.colors as mcolors

def _plot_property_profile(domain:DiscretizedDomain2D, simulated_profile_np:np.array, gwt_depth, ax=None, discrete_point_size=0, plot_gwt = True,
               vlog = False, vmin=None, vmax=None, cmap='gist_earth_r', 
               legend = True, legend_label = None, legendkwargs_dict={},
               origin_x = 0, origin_z = 0):
        """
        Plots a 2D property profile.

        Parameters
        ----------
        domain: DiscretizedDomain2D
            domain instance for profile
        simulated_profile_np : np array
            simulated_profile in np.array
        gwt_depth:
            gwt depth (None if not available (assumes at the bottom))
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
        origin_x, origin_z: dict, float
            Change origin for plotting only. (All plot elements are shifted based on provided origin.) 
        

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

        if origin_z!=0 or origin_x!=0:
           warnings.warn(f"Plot origins are set to [{origin_x}, {origin_z}].  Note that this origin shift applies only to visualization; all computations are performed assuming an origin at (0, 0).")
        
        z_centers, x_centers = domain.z_centers, domain.x_centers
        span_x, span_z = domain.spans
        data = simulated_profile_np.T
        
        extent = [0 + origin_x, span_x + origin_x, span_z + origin_z, 0 + origin_z]
        # Create a colormap from the color mapping
        if vmin is None:
            vmin = np.min(data)

        if vmax is None:
            vmax = np.max(data)
            
        if vlog:
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cax = ax.imshow(data, norm=norm, cmap=cmap, extent=extent, interpolation='none') 
        else:
            cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='none') 
        
        # Plot gwt
        if gwt_depth is not None and plot_gwt:
            edges_kw = dict(color='r', linestyle='dashed', linewidth=2, zorder=4000)
            ax.plot([0 + origin_x, span_x + origin_x], [gwt_depth + origin_z, gwt_depth + origin_z], **edges_kw)

        x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
        if discrete_point_size!=0:
            ax.scatter(x_data.flatten() + origin_x, z_data.flatten() + origin_z, c = 'k', s=discrete_point_size)
        
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
            xlim= [0 + origin_x, span_x + origin_x],
            ylim= [span_z + origin_z, 0 + origin_z],
            xlabel='X',
            ylabel='Z',
        )

        return ax, vmin, vmax
    
def _plot_lit_domain(domain:DiscretizedDomain2D, lithological_matrix:np.array, gwt_depth, ax=None, discrete_point_size=0, legend=True,
                    id2material_dict = None, title='Lithological Domain',
                    color_map = {
                            'def': plt.get_cmap('tab20', 10),      # For integer values
                            'U_': plt.get_cmap('Set3', 10)   # For "U-{x}" values
                    },
                    origin_x = 0, origin_z = 0):
    """
    Plot the lithological domain.

    Parameters
    ----------
    domain: DiscretizedDomain2D
        domain instance for profile
    lithological_matrix : np array
        lithological_matrix in np.array
    gwt_depth:
        gwt depth (None if not available (assumes at the bottom))
        
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. Creates a new figure if None.
    discrete_point_size : float, optional
        Size of discrete scatter points; 0 disables scatter.
    legend : bool, optional
        Whether to display the legend.
    id2material_dict : dict, optional
        Maps IDs to material names.
    title : str, optional
        Plot title.
    color_map : dict, optional
        Prefix-to-colormap dictionary.
    origin_x, origin_z: dict, float
        Change origin for plotting only. (All plot elements are shifted based on provided origin.) 
    """
    if ax is None:
        fig,ax = plt.subplots()

    if origin_z!=0 or origin_x!=0:
        warnings.warn(f"Plot origins are set to [{origin_x}, {origin_z}].  Note that this origin shift applies only to visualization; all computations are performed assuming an origin at (0, 0).")
    
    z_centers, x_centers = domain.z_centers, domain.x_centers
    span_x, span_z = domain.spans
    
    unique_values, color_mapping, integer_mapped_array, fixed_cmap = __get_unique_lithological_color_map(lithological_matrix, color_map)
    
    # Plot the data using imshow with the fixed colormap
    extent = [0 + origin_x, span_x + origin_x, span_z + origin_z, 0 + origin_z]
    cax = ax.imshow(integer_mapped_array, cmap=fixed_cmap, extent=extent, interpolation='none')
    
    # Plot gwt
    if gwt_depth is not None:
        edges_kw = dict(color='r', linestyle='dashed', linewidth=2, zorder=4000)
        ax.plot([0 + origin_x, span_x + origin_x], [gwt_depth + origin_z, gwt_depth + origin_z], **edges_kw)


    x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
    if discrete_point_size!=0:
        ax.scatter(x_data.flatten() + origin_x, z_data.flatten() + origin_z, 
                    c = [color_mapping[value] for value in lithological_matrix.flatten()], 
                    edgecolors='white',  # thin white borders
                    linewidths=0.3,   
                    marker='s',          # square marker
                    s=discrete_point_size)
        
    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]
    gwt_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='GWT')
    handles.append(gwt_handle)
    labels = list(unique_values) + ['GWT']

    if id2material_dict is not None:
        labels = [id2material_dict[label][1] if label in id2material_dict else label for label in labels]
        labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in labels]
    
    if legend:
        ax.legend(handles, unique_values, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if title is not None:
        ax.set_title(title)
        
    ax.axis('scaled')
    ax.set(
        xlim= [0 + origin_x, span_x + origin_x],
        ylim= [span_z + origin_z, 0 + origin_z],
        xlabel='X',
        ylabel='Z',
    )
    return ax

def __get_unique_lithological_color_map(
    lithological_matrix,
    color_map = {
    'def': plt.get_cmap('tab20', 10),      # For integer values
    'U_': plt.get_cmap('Set3', 10)   # For "U-{x}" values
}):
    unique_values = np.unique(lithological_matrix)
    color_mapping = {} 
    for value in unique_values:
        assigned = False
        # print(value, f.is_integer_value(value))
        for prefix, cmap in color_map.items():
            if value == 'X':
                color_mapping[value] = (1.,1.,1.,1.)#'#ffffff'
                assigned = True
                
            elif prefix == 'def' and is_integer_value(value):
                if value == 0 or value == '0':
                    color_mapping[value] = (1.,1.,1.,1.)#'#ffffff'
                else:
                    # If no prefix, assume integer or digit
                    index = int(float(value)) % 10
                    color_mapping[value] = cmap(index)
                assigned = True
                break
            elif isinstance(value, str) and value.startswith(prefix):
                # For prefixed values
                index = int(float(value[len(prefix):])) % 10
                color_mapping[value] = cmap(index)
                assigned = True
                break
        
        # If no pattern matched, assign a random color
        if not assigned:
            color_val = "#" + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)])
            color_mapping[value] = mcolors.to_rgba(color_val)
            
    # Create a colormap from the color mapping
    colors = [color_mapping[value] for value in unique_values]
    int_map = {value: color_mapping[value] for idx, value in enumerate(unique_values)}
    integer_mapped_array = np.array(np.vectorize(int_map.get)(lithological_matrix), dtype='float')
    integer_mapped_array = np.transpose(integer_mapped_array, axes=(2, 1, 0))  #Adjusting for imshow
    fixed_cmap = mcolors.ListedColormap(colors)

    return unique_values, color_mapping, integer_mapped_array, fixed_cmap

def is_integer_value(value):
    try: 
        # Convert value to float, then to int, and back to string to check if integer-like
        return float(value).is_integer()
    except ValueError:
        return False
