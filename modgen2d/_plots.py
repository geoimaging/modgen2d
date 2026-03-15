import matplotlib.pyplot as plt
import numpy as np
import warnings
from .discretized_domain2d import DiscretizedDomain2D
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

def _plot_property_profile(domain:DiscretizedDomain2D, simulated_profile_np:np.array, gwt_depth, ax=None, 
                           discrete_point_size=0, white_edges_size = 0, plot_gwt = True, gwt_kw={},
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
        white_edges_size : float, default 0
            Size of white edges of pixels.
        plot_gwt : bool, default True
            Plot groundwater table.
        gwt_kw : dict,
            keywords for controlling gwt_plot
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
        gwt_handle = None
        if gwt_depth is not None and plot_gwt:
            gwt_handle, gwt_label = _draw_water_table(
                ax, y_level = gwt_depth + origin_z, x_min = 0 + origin_x, x_max=span_x + origin_x,
                **gwt_kw)

        x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
        if discrete_point_size!=0:
            ax.scatter(x_data.flatten() + origin_x, z_data.flatten() + origin_z, c = 'k', s=discrete_point_size)

        if white_edges_size != 0:
            edges_x = np.arange(len(x_centers)) * domain.dhs[0]
            edges_z = np.arange(len(z_centers)) * domain.dhs[1]
        
            for e in edges_x:
                ax.axvline(e, color='white', linewidth=white_edges_size)
        
            for e in edges_z:
                ax.axhline(e, color='white', linewidth=white_edges_size)

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
    
def _plot_lit_domain(domain:DiscretizedDomain2D, lithological_matrix:np.array, gwt_depth, ax=None, 
                     discrete_point_size=0, white_edges_size=1, plot_gwt = True, gwt_kw={}, legend=True, try_clean_legend=False,
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
    white_edges_size : float, default 0
        Size of white edges of pixels.
    plot_gwt : bool, default True
        Plot groundwater table.
    gwt_kw : dict,
        keywords for controlling gwt_plot
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
    gwt_handle = None
    gwt_label = None
    if gwt_depth is not None and plot_gwt:
        gwt_handle, gwt_label = _draw_water_table(
            ax, y_level = gwt_depth + origin_z, x_min = 0 + origin_x, x_max=span_x + origin_x,
            **gwt_kw)

    x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
    if discrete_point_size!=0:
        ax.scatter(x_data.flatten() + origin_x, z_data.flatten() + origin_z, 
                    c = [color_mapping[value] for value in lithological_matrix.flatten()], 
                    edgecolors='white',  # thin white borders
                    linewidths=0.3,   
                    marker='s',          # square marker
                    s=discrete_point_size)

    if white_edges_size != 0:
        edges_x = np.arange(len(x_centers)) * domain.dhs[0]
        edges_z = np.arange(len(z_centers)) * domain.dhs[1]
    
        for e in edges_x:
            ax.axvline(e, color='white', linewidth=white_edges_size)
    
        for e in edges_z:
            ax.axhline(e, color='white', linewidth=white_edges_size)

    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]
    labels = unique_values

    if id2material_dict is not None:
        labels = [id2material_dict[label][1] if label in id2material_dict else label for label in labels]
        labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in labels]
    elif try_clean_legend:
        labels = _get_clean_legend_labels(labels)

    if gwt_handle is not None:
        handles.append(gwt_handle)
        labels = list(labels) + [gwt_label]

    if legend:
        ax.legend(handles, labels, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
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

def _draw_water_table(
    ax, y_level, x_min, x_max,
    triangle_positions=None, triangle_size=10, triangle_hollow=False, triangle_offset_factor=0.03,
    linewidth=.5, linestyle='--', color='r', triangle_color=None, label = 'GWT',
    zorder=4000):

    """
    Draw geotechnical groundwater table symbol (line + inverted triangles).

    Parameters
    ----------
    ax : matplotlib axis
    y_level : float
        Elevation of water table
    x_min, x_max : float
        Horizontal extent of water table line
    triangle_positions : list or None
        x locations for triangles (default evenly spaced)
    triangle_size : int
        Size of triangle markers
    linewidth : float
        Line thickness
    linestyle : str or tuple
        Line dash type ('-', '--', ':', '-.', or custom dash)
    color : str
        Color of symbol
    """

    # draw water table line
    ax.plot([x_min, x_max], [y_level, y_level],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder)

    # default triangle positions
    if triangle_positions is None:
        triangle_positions = np.linspace(x_min, x_max, 3)[1:2]

    # offset so triangle tip sits on line
    triangle_offset = triangle_offset_factor * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # draw inverted triangles
    y_tri = [y_level + triangle_offset] * len(triangle_positions)

    if triangle_color is None:
        triangle_color = color
    
    if triangle_hollow:
        ax.scatter(
            triangle_positions,
            y_tri,
            marker='v',
            s=triangle_size,
            facecolors='none',
            edgecolors=triangle_color,
            linewidths=linewidth,
            zorder=zorder+1
        )
    else:
        ax.scatter(
            triangle_positions,
            y_tri,
            marker='v',
            s=triangle_size,
            color=triangle_color,
            zorder=zorder+1
        )

    handle = None
    if label is not None:
        handle = Line2D([0], [0],
                        color=color,
                        marker='v',
                        markersize=np.sqrt(triangle_size),
                        linestyle=linestyle,
                        markerfacecolor='none' if triangle_hollow else triangle_color,
                        markeredgecolor=triangle_color,
                        linewidth=linewidth)
    return handle, label

def _get_clean_legend_labels(init_legend_labels):
    """
    Convert original layer/anomaly IDs to descriptive legend labels.
    Numeric → geomaterials (G)
    Underscore-prefixed → anomalies/utilities (A)
    If any existing values have prefix G_ (like "G_1"), numeric layers get "_G" prefix.
    """
    # conflict check: any value with "_" whose prefix is "G"
    conflict = False
    for v in init_legend_labels:
        s = str(v)
        if "_" in s:
            prefix = s.split("_")[0]
            if prefix == "G":
                conflict = True
                break

    labels = []
    for v in init_legend_labels:
        s = str(v)
        if "_" in s:           # utility / anomaly
            labels.append(s.replace("_", ""))
        elif s.isnumeric():             # geomaterial / layer
            labels.append(f"_G{s}" if conflict else f"G{s}")
        else:
            labels.append(s)            # leave other strings as-is

    return labels
    
def is_integer_value(value):
    try: 
        # Convert value to float, then to int, and back to string to check if integer-like
        return float(value).is_integer()
    except ValueError:
        return False
