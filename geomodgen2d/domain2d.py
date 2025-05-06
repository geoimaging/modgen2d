import numpy as np
import geomodgen2d.general_functions as f

class Domain2D():
    def __init__(self, span_x: float, span_z: float, del_x: float, del_z: float, name: str=''):
        """
        A class to check and generate coordinate ranges based on given limits and spacing.
        For 1-D models, put span_x = None, and del_x = any number
        
        Parameters:
        span_x, span_z : float
            The upper limit for the x, and z-coordinate range.
        del_x, del_z : float
            The spacing interval for x, and z-coordinates.
        name : str, optional
            Name identifier for the instance (default is an empty string).
        """
        self.check = False
        self.domain3d_name = name
        span_z = float(span_z)
        del_z = float(del_z)
        
        if span_x is not None and span_x != 0:
            span_x = float(span_x)
            del_x = float(del_x)
            self._dim = 2 #2d
        else:
            span_x = span_z/10
            del_x = span_x  # Making sure there is only one point in x axis.
            self._dim = 1 #2d
            
        self._span_x = span_x
        self._span_z = span_z
        self._del_x = del_x
        self._del_z = del_z
        
        assert f.is_divisible(span_x, del_x), f"{name}:span_x ({span_x}) must be divisible by del_x({del_x})."
        assert f.is_divisible(span_z, del_z), f"{name}:span_z ({span_z}) must be divisible by del_z({del_z})."

        self._x_ranges = np.arange(del_x/2,span_x+del_x/4,del_x)
        self._z_ranges = np.arange(del_z/2,span_z+del_z/4,del_z)
            
        self.check = True

    @property
    def x_ranges(self):
        return self._x_ranges

    @property
    def z_ranges(self):
        return self._z_ranges

    @property
    def span_x(self):
        return self._span_x

    @property
    def span_z(self):
        return self._span_z

    @property
    def del_x(self):    
        return self._del_x

    @property
    def del_z(self):
        return self._del_z
    
    @property
    def dim(self):
        return self._dim

def check_spacing_compability(current_del, new_del):
    if current_del>new_del:
        larger_del = current_del
        smaller_del = new_del
    else:
        larger_del = new_del
        smaller_del = current_del

    # assert larger_del>=smaller_del, f"Required Condition: {larger_del}>={smaller_del}. NOT CORRECT"  #Well should always be true
    if larger_del == 0. and smaller_del == 0:
        pass
    else:
        if not f.is_divisible(larger_del, smaller_del):
            print(f"Spacings i.e. new_del(={new_del}) and del_org(={current_del}) are not divisible with each other, i.e new mesh might not include original points. {larger_del%smaller_del}")     
            return False
    return True

def check_for_remeshing_coordinate_compatibility(current_domain2D, new_span_x, new_span_z, new_del_x, new_del_z):
    ## Changed from check_refined_coordinate_compatibility ():
    """
    Checks the compatibility of a remeshing coordinate grid with the original grid specifications.

    Parameters:
    current_domain2D : object - 
        An instance of any class that's root is domain2D class that contains the grid properties.
    new_span_x, new_span_z, new_del_x, new_del_z: float
        new coordinates to check    
    """

    remeshed_2D_domain = Domain2D(new_span_x, new_span_z, new_del_x, new_del_z)  #Check init compabilities
    
    assert current_domain2D.span_x == remeshed_2D_domain.span_x, f"Required Condition: {current_domain2D.span_x}=={new_span_x}. NOT CORRECT"
    if current_domain2D.span_z is not None and remeshed_2D_domain.span_z is not None:
        assert current_domain2D.span_z == remeshed_2D_domain.span_z, f"Required Condition: {current_domain2D.span_z}=={remeshed_2D_domain.span_z}. NOT CORRECT"

    ## No flagging, but print warning if not compatible (exactly divisible)
    check_spacing_compability(current_domain2D.del_x, remeshed_2D_domain.del_x)
    check_spacing_compability(current_domain2D.del_z, remeshed_2D_domain.del_z)

    return remeshed_2D_domain