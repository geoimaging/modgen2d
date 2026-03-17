# This file is part of geomodgen3d a Python package for 3D model generation.
# Copyright (C) 2025 Bhochhibhoya S. and Vantassel, J.P. (joseph.p.vantassel@gmail.com)
#
# LICENSE

import h5py, warnings
import numpy as np
import modgen2d.interface.global_soil_interface_config as global_soil_interface_config
from modgen2d.generated_model2d import GeneratedProfileCollection2D, GeneratedProfileCollection2DReadOnly
from modgen2d.metadata import __version__

def load_dict_from_hdf5(group):
    """
    Recursively loads the contents of an HDF5 group into a Python dictionary.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group object to load.

    Returns
    -------
    dict
        Nested dictionary containing the data from the HDF5 group. 
        Supports integers, floats, strings, arrays, and nested groups.
    """
    loaded_dict = {}
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            # Recursively load nested dictionary
            loaded_dict[key] = load_dict_from_hdf5(item)
        elif isinstance(item, h5py.Dataset):
            data = item[()]
            if key in ("state", "inc"):
                loaded_dict[key] = int(data)
                
            elif isinstance(data, bytes):
                loaded_dict[key] = None if data == b'__None__' else data.decode('utf-8')
            
            elif isinstance(data, np.ndarray):
                if data.dtype.kind in {'S', 'O'}:
                    loaded_dict[key] = data.astype(str)
                    if loaded_dict[key].ndim == 1:
                        loaded_dict[key] = loaded_dict[key].tolist()
                else:
                    loaded_dict[key] = data.tolist() if data.ndim == 1 else data
                    
            elif isinstance(data, np.generic):
                if np.issubdtype(type(data), np.integer):
                    loaded_dict[key] = int(data)
                elif np.issubdtype(type(data), np.floating):
                    loaded_dict[key] = float(data)
                else:
                    loaded_dict[key] = data.item()
            else:
                loaded_dict[key] = data
    return loaded_dict

def read_hdf5_file(to_file, read_only=False, check_merged = False):
    """
    Reads a saved HDF5 file containing a modgen2d model collection and reconstructs
    the corresponding Python objects.

    Parameters
    ----------
    to_file : str
        Path to the HDF5 file to read.
    read_only : bool, default False
        Whether to load the model collection in read-only mode. Raises an error if the file
        was saved with read-only but read_only=False is requested.
    check_merged : bool, default False
        Reserved for future use. Currently not used in this function.

    Returns
    -------
    GeneratedProfileCollection2D or GeneratedProfileCollection2DReadOnly
        Loaded model collection instance corresponding to the saved HDF5 data.

    Raises
    ------
    ValueError
        If a read-only file is attempted to be opened with read_only=False.
    """
    with h5py.File(to_file, 'r') as hf:
        full_config = load_dict_from_hdf5(hf)
    
    read_only_flag = full_config['save_read_only']
    version = full_config['modgen2d_version']
    
    if read_only_flag is True and read_only is False:
        raise ValueError("The hdf5 file was saved with read_only purpose, but attempted to open with non-read only purpose (i.e., read_only=False).")
    
    if version != __version__:
        warnings.warn(f"The modgen2d version mismatch. Saved at version {version}, but attempting loading at version {__version__}.")
        
    global_soil_interface_config.GlobalSoilInterfaceConfig.set_soil_interface_from_config(full_config['global_interface_config'])
    if read_only:
        generated_profiles3d_instance = GeneratedProfileCollection2DReadOnly.from_config(full_config['gen_model_2d_collection'])
    else:
        generated_profiles3d_instance = GeneratedProfileCollection2D.from_config(full_config['gen_model_2d_collection'])
    return generated_profiles3d_instance
    
# def subplot_prep():
#     fig = plt.figure(figsize=(14/1.2, 10/1.2), constrained_layout=True)
#     gs = gridspec.GridSpec(
#         2, 2, figure=fig,
#         width_ratios=[2, 2],  
#         height_ratios=[2, 2]  
#     )

#     ax3d = fig.add_subplot(gs[0, 0], projection='3d')  # Large 3D plot
#     ax2d_left = fig.add_subplot(gs[1, 0])              # Bottom-left
#     ax2d_bottom = fig.add_subplot(gs[0, 1])            # Top-right

#     return fig, ax3d, ax2d_left, ax2d_bottom

# def find_h5_file(base_folder, file_slider):
#     target_filename = f'{file_slider:08.0f}.h5'

#     # Step 1: Direct check in base folder
#     direct_path = os.path.join(base_folder, target_filename)
#     if os.path.isfile(direct_path):
#         return direct_path

#     # Step 2: Search only in immediate subdirectories
#     for sub in os.listdir(base_folder):
#         sub_path = os.path.join(base_folder, sub)
#         if os.path.isdir(sub_path):
#             candidate = os.path.join(sub_path, target_filename)
#             if os.path.isfile(candidate):
#                 return candidate

#     # Step 3: Not found
#     raise FileNotFoundError(f"{target_filename} not found in {base_folder} or its immediate subfolders.")

# # Create a function to update the plot
# def make_update_plot_box():
#     prev_file = [None]  # use mutable container to hold state
#     prev_instance = [None]
    
#     def update_plot_box(to_generated_hdf5_folder, file_slider, plot_type, plot_case, vmin_max, slice_x_at_m=None, slice_y_at_m=None, slice_z_at_m=None, check_merged=False):
#         start_time = time.time()
#         # Only reload if file has changed
#         if file_slider != prev_file[0]:
#             # print(f"Loading new file: {file_slider}")
#             to_file = find_h5_file(to_generated_hdf5_folder, file_slider)
#             generated_profiles3d_instance = read_hdf5_file(to_file, read_only=True, check_merged=check_merged)
#             prev_file[0] = file_slider
#             prev_instance[0] = generated_profiles3d_instance
#         else:
#             # print(f"Using cached file: {file_slider}")
#             generated_profiles3d_instance = prev_instance[0]
        
#         if plot_case == 'Soil Only':
#             instance_key = 'gen_order_1'
#             gen_order = 1
#         elif plot_case == 'Obstructions Only':
#             instance_key = 'gen_order_2'
#             gen_order = 2
#         else:
#             gen_order = None
#             if 'merged' in generated_profiles3d_instance.lithological_domain3D_instance_dict.keys():
#                 instance_key = 'merged'
#             else:
#                 instance_key = 'gen_order_1'
                
#         # Clear axes for fresh plotting
#         fig, ax3d, ax2d_left, ax2d_bottom = subplot_prep()
#         ax3d.cla()
#         ax2d_bottom.cla()
#         ax2d_left.cla()
        
#         if instance_key not in generated_profiles3d_instance.lithological_domain3D_instance_dict.keys():
#             print(f"No {plot_type}, i.e., {instance_key}, loaded/found in the hdf5 file")
#             return
#         else:
#             lit_domain = generated_profiles3d_instance.lithological_domain3D_instance_dict[instance_key]
                    
#             discretizedinterfaces3d_instance = lit_domain.discretizedInterfaces3D_instance
#             scatter_point_size = 0
            
#             if plot_type == 'Boundary':
#                 if discretizedinterfaces3d_instance is None:
#                     print("No boundary loaded/found in the hdf5 file")
#                 else:
#                     fig_kwargs = {'title':'auto', 'section_colors': ['b', 'g', 'r']}
#                     discretizedinterfaces3d_instance.plot_boundary(ax3d, slice_x_at_m=slice_x_at_m, slice_y_at_m=slice_y_at_m, warning = True, **fig_kwargs)
#                     discretizedinterfaces3d_instance.plot2d(ax = ax2d_bottom, section = ['x',slice_x_at_m], warning = False, **fig_kwargs)
#                     discretizedinterfaces3d_instance.plot2d(ax = ax2d_left, section = ['y',slice_y_at_m], warning = False, **fig_kwargs)

#             elif plot_type == 'Lithological Domain 3D':
#                 if lit_domain is None:
#                     print("No lithological domain loaded/found in the hdf5 file")
#                 else:
#                     if isinstance(generated_profiles3d_instance.lit_id2material_dict,dict):
#                         id2material = generated_profiles3d_instance.lit_id2material_dict
#                     else:
#                         id2material = None
#                     fig_kwargs = {'section_colors': ['b', 'g', 'r']}
#                     lit_domain.plot3d(ax = ax3d, slice_x_at_m=slice_x_at_m, slice_y_at_m=slice_y_at_m, slice_z_at_m=slice_z_at_m, warning = True, title='auto', legend=True, id2material_dict = id2material, **fig_kwargs)
#                     lit_domain.plot2d(ax = ax2d_bottom, section = ['x',slice_x_at_m], warning = False, scatter_point_size=scatter_point_size, title='auto', legend=True, **fig_kwargs)
#                     lit_domain.plot2d(ax = ax2d_left, section = ['y',slice_y_at_m], warning = False, scatter_point_size=scatter_point_size, title='auto', legend=True, **fig_kwargs)
#                     # print(lit_domain.utils_description)
#             else:
#                 plot_actual = {'Shear Wave Velocity (m/s)':'vs', 
#                             "Poisson's Ratio":'miu', 
#                             'Density (kg/m3)':'rho', 
#                             'Dielectric Constant':'dc',
#                             'Electric Conductivity (mS/s)':'ec',
#                             'z_vals_seismic': 'z_vals_seismic',
#                             'z_vals_electromagnetic': 'z_vals_electromagnetic'}

#                 if plot_type not in plot_actual.keys():
#                     raise ValueError("Invalid plot_type.")

#                 property_name = plot_actual[plot_type]
                
#                 try:
#                     vmin, vmax, vlog = vmin_max[property_name]
#                 except:
#                     vmin, vmax, vlog = None, None, False

#                 fig_kwargs = {'section_colors': ['b', 'g', 'r'], 'legend_label':plot_type}
#                 _, vmin, vmax = generated_profiles3d_instance.plot3d(property_name, gen_order, slice_x_at_m, slice_y_at_m, slice_z_at_m, vlog=vlog, vmin=vmin, vmax=vmax, ax=ax3d, fig=fig, title='auto', legend=True, **fig_kwargs)
                
#                 # mappables = [m for m in ax3d.get_children() if hasattr(m, 'get_clim')]
#                 # # Pick the first (or appropriate) one and get limits
#                 # if mappables:
#                 #     vmin, vmax = mappables[0].get_clim()
#                 # print(vmin, vmax)            
#                 generated_profiles3d_instance.plot2d(property_name, gen_order, section = ['x',slice_x_at_m], vlog=vlog, vmin=vmin, vmax=vmax, ax=ax2d_bottom, scatter_point_size=scatter_point_size, title='auto', legend=True, **fig_kwargs)
#                 generated_profiles3d_instance.plot2d(property_name, gen_order, section = ['y',slice_y_at_m], vlog=vlog, vmin=vmin, vmax=vmax, ax=ax2d_left, scatter_point_size=scatter_point_size, title='auto', legend=True, **fig_kwargs)
#         ax3d.view_init(elev=30, azim=-45) 
#         # plt.tight_layout()
#         # plt.show()
#         # # display(fig)
#         # total_duration = time.time() - start_time
#         # print(f"Total update time: {total_duration:.2f} seconds")
#     return update_plot_box