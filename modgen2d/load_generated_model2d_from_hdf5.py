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
