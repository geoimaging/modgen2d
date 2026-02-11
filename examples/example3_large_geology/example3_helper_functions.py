from scipy.ndimage import gaussian_filter
import modgen2d as mg
import modgen2d.general_functions as f
import numpy as np

def smooth_noise(shape, sigma=3, rng=np.random.default_rng()):
    noise = rng.random(shape) * 2 - 1  # random in [-1,1]
    noise_smooth = gaussian_filter(noise, sigma=sigma)
    # Normalize to [-1,1]
    noise_smooth = noise_smooth / np.max(np.abs(noise_smooth))
    return noise_smooth

class ManualObstruction2D(mg.Obstruction2D):
    def __init__(self, dl:float, ref_xz_symbolic = ['c', 'c'], snap_to_dl:bool=True):
        super().__init__(dl, ref_xz_symbolic, snap_to_dl)
       
    def karst_circle_perlin(self, d, noise_amplitude=0.1, smooth_sigma=3.0, obstruction_id=1, rng=np.random.default_rng()):
        assert self.shape is False, "ERROR: utils shape has already been defined"
        if d<0:
            raise ValueError("Diameter of the circle must be zero/positive")
        
        obstruction_id = f.check_obstruction_id(obstruction_id)

        d_adj = d
        r_nominal = d_adj / 2
        # Determine maximum radius including positive noise
        r_max = r_nominal * (1 + noise_amplitude)
        n_grid = int(np.ceil(2 * r_max / self.dl))
        
        grid = np.zeros((n_grid, n_grid), dtype=int)
        
        x = np.arange(n_grid) * self.dl + self.dl/2
        z = np.arange(n_grid) * self.dl + self.dl/2
        xx, zz = np.meshgrid(x, z, indexing='ij')
        
        # Generate smooth noise
        noise_smooth = smooth_noise((n_grid, n_grid), sigma=smooth_sigma, rng=rng)

        # Compute noisy radius
        r_grid = r_nominal * (1 + noise_amplitude * noise_smooth)

        # Compute distance from center
        center = r_max
        dist2 = (xx - center)**2 + (zz - center)**2
        grid[dist2 <= r_grid**2] = obstruction_id

        # Update object attributes
        self.grid2d = grid
        self.description = f'Karst circle (smooth noise) 2D, base diameter {d:.6g}'
        self.shape = True
        self.center_in_unit_length = [center, center]

    def tunnel_shape(self, lx, lz, obstruction_id=1, rng=np.random.default_rng()):

        if lx/2 > lz:
            raise ValueError("Height of the tunnel cannot be less than radius of the circle (roof) at top")

        ## Create the base first
        self.rectangle_2d(lx, lz, obstruction_id = obstruction_id)

        ## Create the circular top 
        circle_top = mg.obstruction2d.Obstruction2D(dl = self.dl, ref_xz_symbolic = ['c','c'], snap_to_dl=self.snap_to_dl)
        circle_top.circle_2d(d=lx,  obstruction_id = obstruction_id)
        
        ## Truncate the circle to semi-circle.
        r_to_grid_length = int(np.round(circle_top.center_in_unit_length[1]/self.dl,0))
        circle_top.expand_grid(new_grid_xlen = None, new_grid_zlen = r_to_grid_length, warn_truncate=False)


        ## Merge the base and top
        self.merge_shapes(circle_top)
        
        self.description = f'Tunnel2D of size (lx x lz) = ({lx} x {lz})'