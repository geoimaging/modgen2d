import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from fbm import FBM

from geomodgen2d.domain2d import Domain2D, check_for_remeshing_coordinate_compatibility
import geomodgen2d.general_functions as f

class BoundaryCreator(Domain2D):

    def __init__(self, span_x: float, span_z: float, del_x: float, del_z: float, n_layers: int):
        """
        Initializes the BoundaryCreator instance with given spatial limits, spacing, boundary settings,
        and number of layers.
        
        Parameters:
        span_x, span_z : float
            The upper limit for the x, and z-coordinate range.
        del_x, del_z : float
            The spacing interval for x, and z-coordinates.
        n_layers: float, 
            Number of soil layers in the model
        """
        
        super().__init__(span_x, span_z, del_x, del_z)
        # Step1: Create empty matrix
        self.n_layers = int(n_layers)
        self.boundary_array = np.zeros((self.n_layers-1, len(self.x_ranges)))    # depths of each boundary interfaces. Note: values in format (z,x)

    def print(self):
        """Prints the dimensions of the boundary array and its content."""
        print(f"N_interfaces = {self.boundary_array.shape[0]}, N_x_coord = {self.boundary_array.shape[1]}")
        print("Boundary array \n", self.boundary_array) #No need to transpose as formatted already in z,x 

    #GENERAL PROCEDURE TASKS:
    def gen_using_def_process(self, boundary_settings_dict, rnd_no=np.random.default_rng()):
        self.generating_boundary(boundary_settings_dict['generator_settings_dict'], rnd_no = rnd_no)
        self.filtering_boundary(**boundary_settings_dict['filter_settings_dict'])
        self.boundary_init_points(init_boundary=boundary_settings_dict['random_init_boundary_option'] , rnd_no = rnd_no) #None means get one from 
        self.processing_boundary(boundary_settings_dict['boundary_overlap_bottom_priority'])
                        
    def edit_boundary_matrix(self, boundary_array):
        """
        Updates the boundary_array with a new matrix.
        
        Parameters:
        boundary_array : numpy array: 
            A NumPy array of the same shape as the existing boundary array
        """
        assert self.boundary_array.shape == boundary_array.shape, "New array must have same shape as previous array"
        self.boundary_array = boundary_array

    def generating_boundary(self, random_generator_settings_dict, rnd_no=np.random.default_rng()):
        """
        Generates a boundary matrix with randomized boundary points.

        Parameters:
        random_generator_option: str
            Method for random generation of boundary. Options: 'uniform', 'normal', 'fbm'.

        random_generator_settings_dict: dict
            Dictionary containing all relevant settings for respective option.

            Option: a) 'uniform' (original), next boundary point at x+del_x = depth at x + uniformly generated random no.
                random_generator_settings_dict keys:
                    1) 'z_max_change_per_m' required: maximum change in z allowed per m
                            
            Option: b) 'normal', next boundary point at x+del_x = depth at x + gaussian generated random no.
                random_generator_settings_dict keys:
                    1)'std'
                        
            Option: c) 'fbm', boundary generated based on fractional brownian motion
                random_generator_settings_dict keys:
                    1) 'H' : Hurst index (1/2) for classical brownian motion
                    2) 'method': Solver for fbm

        rnd_no: 
            Optional random number generator instance
        """
        if self.dim == 2:  #If 2D only.
            b_array = self.boundary_array
            n_layers = self.n_layers
            random_generator_option = random_generator_settings_dict['generator_option']
            if random_generator_option=='uniform':
                z_b_change_per_m = random_generator_settings_dict['z_max_change_per_m']
                del_x = self.del_x
                z_max_change_per_del_x = (z_b_change_per_m*del_x)
                rnd_numbers = (rnd_no.random((self.n_layers-1, (len(self.x_ranges)-1)))-0.5)*2 #Numbers ranging from 1 and -1
                b_array[:,1:] = rnd_numbers*z_max_change_per_del_x
                b_array = np.cumsum(b_array, axis=1)
                
            elif random_generator_option=='normal':
                mean = 0
                std = random_generator_settings_dict['std']
                rnd_numbers = rnd_no.normal(loc=mean, scale=std, size=(self.n_layers-1, len(self.x_ranges)-1)) #Numbers ranging from 0 to 1
                b_array[:,1:] = rnd_numbers
                b_array = np.cumsum(b_array, axis=1)
            
            elif random_generator_option=='fbm':
                H = random_generator_settings_dict['H']
                L = random_generator_settings_dict['length']
                method = random_generator_settings_dict['method']
                n = len(self.x_ranges) - 1
                for j in range(self.n_layers-1):
                    rnd_layer = FBM(n=n, hurst=H, length=L, method=method).fbm() #generates n+1 data ie n increments
                    init_layer = b_array[j,0]
        
                    rnd_layer+=init_layer
                
                    b_array[j,:]= rnd_layer
            else:
                raise ValueError("random_generator_options can only be either 'uniform', or 'normal', or 'fbm'")

            self.edit_boundary_matrix(b_array)
        
    
    def boundary_init_points(self, init_boundary='random_sum', ref_x_val=0, rnd_no=np.random.default_rng()):
        """
        Initializes boundary points at reference coordinates. Shifting the boundaries' reference points so that they match the values in boundary_init_points.

        Parameters:
        init_boundary: str or list
            Mode of initialization ('equidistant', 'random_sum', 'random_sort', or numPy array of floats)
        ref_x_val: float
            Reference x-coordinate for initialization
        rnd_no: 
            Optional random number generator instance
        """
        #init_boundary = 'equidistant', 'random_sum', 'random_sort', [.., .., ..]


        # Find nearest value to x-ref to points in array
        x_array = self.x_ranges
        nearest_index = np.abs(x_array - ref_x_val).argmin()
        nearest_ref_x_val = x_array[nearest_index]
        b_array = self.boundary_array
        
        if np.abs(nearest_ref_x_val-ref_x_val)>=1e-2:
            print(f"Provided reference x_val ({ref_x_val}) is not available in boundary x_ranges: {x_array}. So using nearest value (i.e., {nearest_ref_x_val})")

        # Getting the z-values at the reference point.
        if isinstance(init_boundary, np.ndarray):
            assert init_boundary.dtype == np.float64 or init_boundary.dtype == np.float32, "init_boundary must contain float values"
            assert init_boundary.ndim == 1, "init_boundary must be one-dimensional"
            assert self.n_layers-1 == init_boundary.shape[0], f"The provided reference boundaries's #boundaries ({init_boundary.shape[0]}) != provided no of interfaces ({self.n_layers}-1)"
            init_z_vals = init_boundary

        elif init_boundary == 'equidistant':
            # Equal thickness at y = 0  eg. [1/4,2/4,3/4] for n_layers = 4
            init_z_vals= [i*self.span_z/self.n_layers for i in np.arange(1, self.n_layers)]
            
        elif init_boundary == 'random_sum':
            #Random depths, such that the total thickness is span_z... say [0.1, 0.2, 0.05, 0.15] => [1/5, 2/5, 0.5/5, 1.5/5] => [1/5, 3/5, 3.5/5, 5/5]
            rndm_numbers = rnd_no.random(self.n_layers)
            thickness = (rndm_numbers/sum(rndm_numbers))*self.span_z
            depths = np.cumsum(thickness)
            init_z_vals = depths[:-1]

        elif init_boundary == 'random_sort':
            # Randomized and sorted, directly randomized the layer points
            rndm_numbers = rnd_no.random(self.n_layers-1)
            rndm_numbers.sort()
            init_z_vals = rndm_numbers*self.span_z
        else:
            raise ValueError("init_boundary must be a NumPy array of float values, if not 'equidistant', 'random_sum', or 'random_sort'")


        # computing the shift
        shift_z = init_z_vals - b_array[:, nearest_index]  
        print(shift_z)
        shift_matrix = np.ones_like(b_array) * shift_z[:,np.newaxis]
        self.edit_boundary_matrix(b_array+shift_matrix)
      
    def filtering_boundary(self, filter_window_length=21, filter_polyorder=3):
        """
        Applies a Savitzky-Golay filter to smooth the boundary.

        Parameters:
        filter_window_length: int
            Window size for the filter. If the value is zero, then it means no filtering.
        filter_polyorder: int
            Polynomial order for the filter
        """
        if self.dim == 2:  #If 2D only.
            if filter_window_length!=0:
                b_array = scipy.signal.savgol_filter(self.boundary_array, window_length=filter_window_length, polyorder=filter_polyorder, axis=1)#, window_length, polyorder
                self.edit_boundary_matrix(b_array)

    def processing_boundary(self, bottom_erosion):
        """
        Processes boundaries to prevent overlapping and limit values.
        
        bottom_erosion: Boolean flag to determine priority in overlapping layers
        """
        #b_line_filtered_dict, zlim, top_priority=True):
        # Process 1: Limiting the boundaries to 0 and zlim
        # Process 2: Boundary crossing handling - Currently, priority given to lower boundary (v3: option to reverse the priority)
        if self.dim == 2:  #If 2D only.            
            b_array = self.boundary_array
            n_interface = b_array.shape[0]
            if bottom_erosion:
                for i in np.arange(n_interface):
                    b_array[i]=np.clip(b_array[i], -1, self.span_z)  #clip between -1 and span_z
                    if i!=0:
                        b_array[i] = np.maximum(b_array[i], b_array[i-1])
            else:
                for i in np.arange(n_interface-1, -1, -1):
                    b_array[i]=np.clip(b_array[i], -1, self.span_z)
                    if i!=n_interface-1:
                        b_array[i] = np.minimum(b_array[i], b_array[i-1])
                    
            self.edit_boundary_matrix(b_array)

    def remeshing_boundary(self, new_del_x, interp_method='linear'):
        """
        Remeshes the boundary coordinates using interpolation.

        Parameters:
        new_del_x: float
            Refined spacing
        interp_method: str
            Interpolation method (default: 'linear')
        """

        if self.dim == 1:
            print("Warning: No effect of remeshing on 1D boundary generation (depth only)")

        else:
            n_interface = self.boundary_array.shape[0]
            if n_interface!=0:
                remeshed_3D_domain = check_for_remeshing_coordinate_compatibility(self, self.span_x, self.span_z, new_del_x, self.del_z)
                x_ranges_old = self.x_ranges
                zs = range(n_interface)
                b_array = f.remeshing_2D_matrix(x_old = x_ranges_old, x_new = remeshed_3D_domain.x_ranges, z_old = zs, z_new = zs, matrix_2d = self.boundary_array.T, interp_method = interp_method)
                self_copy = BoundaryCreator(remeshed_3D_domain.span_x, remeshed_3D_domain.span_z, remeshed_3D_domain.del_x, remeshed_3D_domain.del_z, self.n_layers)
                self_copy.edit_boundary_matrix(b_array.T)
                self.__dict__.update(self_copy.__dict__)
        
    def plot_boundary(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=[8,8])

        n_interface = self.boundary_array.shape[0]
        x_ranges = self.x_ranges
        x_ranges_plt = x_ranges
        b_array = self.boundary_array
        if self.dim == 1:
            x_ranges = [-self.span_z/10, self.span_z/10]
            x_ranges_plt = [0]
            b_array = np.ones([n_interface, 2])*b_array[:,None]
        for i in np.arange(n_interface-1, -1, -1):
            ax.plot(x_ranges_plt,
                    b_array[i,:],label=i,
                    linestyle='-',
                   # color=color_code[i],
                   )

        ax.plot([x_ranges[0],x_ranges[-1]], [0,0], 'k--',[x_ranges[0],x_ranges[-1]], [self.span_z, self.span_z],'k--')
        ax.set(ylim=[self.span_z, 0],
               xlim=[x_ranges[0],x_ranges[-1]],
               xlabel='Distance',
               ylabel='Depth',
               )
        ax.axis('scaled')
        # ax.invert_yaxis()
        return ax

class SurfaceBoundaryCreator(BoundaryCreator):
    def __init__(self, span_x: float, span_z: float, del_x: float, del_z: float, generator_option:str = 'uniform', gen_param:float = 0, filter_window=21, filter_order=3, bottom_erosion=True, rnd_no=np.random.default_rng()):
        """
        Creates a surface_boundary layer. It will have 2 layers (i.e., 1 interface). 
        Only have z_max_change_per_m_top_option (i.e., only created using uniform option).
        
        Parameters:
        span_x, span_z : float
            The upper limit for the x, and z-coordinate range.
        del_x, del_z : float
            The spacing interval for x, and z-coordinates.
        generator_option: str
            Only options: 'uniform', 'normal'  ('fbm' not yet included).
        gen_param: float
            if 'uniform', it refers to 'z_max_change_per_m'
            if 'normal', it refers to 'std'
        filter_window, filter_order:
            parameters for filtering (filter_window=0 if no filtering)
        bottom_erosion,
            parameters of processing
        
        rnd_no: 
            Optional random number generator instance
        """
        
        super().__init__(span_x, span_z, del_x, del_z, n_layers = 2)

        assert generator_option in ['uniform', 'normal'], f"Only allowed options yet are uniform and normal. Provided {generator_option}"
        generator_settings_dict = {
                 'generator_option':generator_option,    # options: 'uniform', 'normal'
                 'z_max_change_per_m':gen_param,   # Required for 'uniform' only
                 'std':gen_param     # Required for 'normal' only Mean has to be zero
                }
        
        self.generating_boundary(generator_settings_dict, rnd_no = rnd_no)
        self.filtering_boundary(filter_window, filter_order)
        top_depth = np.min(self.boundary_array)
        self.boundary_init_points(init_boundary=np.array([-top_depth+self.boundary_array[0][0]])) #None means get one from 
        self.processing_boundary(bottom_erosion)
        assert np.abs(np.min(self.boundary_array)) <= 1e-2