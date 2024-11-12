import numpy as np
from math import sin, cos, atan2, sqrt
import jax
from functools import partial
import jax.numpy as jnp

def set_shapes_matrix(shapes_matrix_):
    '''
    This function imports the shapes_matrix object used in
    the create_multiple_arbitrary_windowpanes function from
    create_windowpanes.py into this module as global variables
    for use with the jax integrated function multiple_sin_cos_components_to_xyz.
    Input: shapes_matrix
    Output: None
    '''
    global shapes_matrix
    shapes_matrix = shapes_matrix_
    return None

def set_opt_coil_parameters(ntoroidalcoils_, npoloidalcoils_, nfp_, stellsym_, unique_shapes_, winding_surface_function_, order_, curves_, num_currents_):
    '''
    This function imports the parameters used in the
    create_multiple_arbitrary_windowpanes function from
    create_windowpanes.py into this module as global
    variables for use with the jax integrated functions
    sin_cos_components_to_xyz and multiple_sin_cos_components_to_xyz.
    Input: ntoroidalcoils, npoloidalcoils, nfp, stellsym, unique_shapes,
           winding_surface_function, order, curves, num_currents
    Output: None
    '''
    global ntoroidalcoils, npoloidalcoils, nfp, stellsym, unique_shapes, winding_surface_function, order, curves, num_currents
    ntoroidalcoils = ntoroidalcoils_
    npoloidalcoils = npoloidalcoils_
    nfp = nfp_
    stellsym = stellsym_
    unique_shapes = unique_shapes_
    winding_surface_function = winding_surface_function_
    order = order_
    curves = curves_
    num_currents = num_currents_
    return None

@partial(jax.jit, static_argnames=['ntoroidalcurves','npoloidalcurves', 'nfp', 'stellsym', 'winding_surface_function', 'normaltowinding', 'order'])
def sin_cos_components_to_xyz(dofsarr, ntoroidalcurves, npoloidalcurves, nfp, stellsym,
                                 winding_surface_function = None, normaltowinding=False, order=5):
    '''
    This function takes a single set of planar sine and
    cosine dofs and creates the corresponding x, y,
    and z dofs for use with Simsopt. It is created as
    a jax function in order to allow for quick computation
    of a jacobian for the transformation between the two
    sets of variables.
    Input: array of planar sin/cos comps, ntoroidalcurves, 
           npoloidalcurves, nfp, stellsym, winding_surface_function, 
           normaltowinding, order
    Output: xyz dofs for all desired curves
    '''
    from create_windowpanes import planar_vector_list
    sin_components = dofsarr[:2*order].reshape(order,2)
    cos_components = dofsarr[2*order:].reshape(order,2)
    curves_xyz_dofs = jnp.array([])
    for i in range(int(ntoroidalcurves)):
        tor_angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        for j in range(int(npoloidalcurves)):
            pol_angle = j*(2*np.pi)/npoloidalcurves
            # Initialize the components
            components = jnp.array([],float)
            if normaltowinding:
                _, normal_vec = winding_surface_function(tor_angle, pol_angle)
                tor_winding = atan2(normal_vec[1], normal_vec[0])
                pol_winding = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
                planar_vectors = planar_vector_list(tor_winding, pol_winding)
            else:
                planar_vectors = planar_vector_list(tor_angle, pol_angle)
            # Set the components
            for ord in range(order):
                scomponent = sin_components[ord][0]*planar_vectors[0] + sin_components[ord][1]*planar_vectors[1]
                ccomponent = cos_components[ord][0]*planar_vectors[0] + cos_components[ord][1]*planar_vectors[1]
                components = jnp.append(components, scomponent)
                components = jnp.append(components, ccomponent)
            curve_xyz_dofs = components.reshape((2*order,3)).T.flatten()
            curves_xyz_dofs = jnp.append(curves_xyz_dofs, curve_xyz_dofs)
    return curves_xyz_dofs

@partial(jax.jit, static_argnames=['ntoroidalcurves','npoloidalcurves', 'nfp', 'stellsym', 'unique_shapes', 'winding_surface_function', 'normaltowinding', 'order'])
def multiple_sin_cos_components_to_xyz(dofsarr, ntoroidalcurves, npoloidalcurves, nfp, stellsym, unique_shapes,
                                       winding_surface_function = None, normaltowinding=False, order=5):
    '''
    This function takes an arbitrary number of sets
    of planar sine and cosine dofs corresponding to
    an arbitrary number of independent shapesand creates 
    the corresponding x, y, and z dofs for use with
    Simsopt. It is created as a jax function in order to
    allow for quick computation of a jacobian for the
    transformation between the two sets of variables.
    Input: array of planar sin/cos comps, ntoroidalcurves, 
           npoloidalcurves, nfp, stellsym, nuniqueshapes,
           winding_surface_function, normaltowinding, order
    Output: xyz dofs for all desired curves
    '''
    from create_windowpanes import planar_vector_list
    sin_components_all = [None]*unique_shapes
    cos_components_all = [None]*unique_shapes
    for i in range(unique_shapes):
        sin_components_all[i] = dofsarr[2*order*(2*i):2*order*(2*i+1)].reshape(order,2)
        cos_components_all[i] = dofsarr[2*order*(2*i+1):2*order*(2*i+2)].reshape(order,2)
    curves_xyz_dofs = jnp.array([])
    for i in range(int(ntoroidalcurves)):
        tor_angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        for j in range(int(npoloidalcurves)):
            shape = shapes_matrix[i][j]
            if not shape == -1:
                pol_angle = j*(2*np.pi)/npoloidalcurves
                sin_components_this = sin_components_all[shape]
                cos_components_this = cos_components_all[shape]
                # Initialize the components
                components = jnp.array([],float)
                if normaltowinding:
                    _, normal_vec = winding_surface_function(tor_angle, pol_angle)
                    tor_winding = atan2(normal_vec[1], normal_vec[0])
                    pol_winding = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
                    planar_vectors = planar_vector_list(tor_winding, pol_winding)
                else:
                    planar_vectors = planar_vector_list(tor_angle, pol_angle)
                # Set the components
                for ord in range(order):
                    scomponent = sin_components_this[ord][0]*planar_vectors[0] + sin_components_this[ord][1]*planar_vectors[1]
                    ccomponent = cos_components_this[ord][0]*planar_vectors[0] + cos_components_this[ord][1]*planar_vectors[1]
                    components = jnp.append(components, scomponent)
                    components = jnp.append(components, ccomponent)
                curve_xyz_dofs = components.reshape((2*order,3)).T.flatten()
                curves_xyz_dofs = jnp.append(curves_xyz_dofs, curve_xyz_dofs)
    return curves_xyz_dofs

def multiple_sin_cos_components_to_xyz_2(dofsarr, ntoroidalcurves, npoloidalcurves, nfp, stellsym, unique_shapes,
                                       winding_surface_function = None, normaltowinding=False, order=5):
    '''
    This function takes an arbitrary number of sets
    of planar sine and cosine dofs corresponding to
    an arbitrary number of independent shapesand creates 
    the corresponding x, y, and z dofs for use with
    Simsopt. It is created as a jax function in order to
    allow for quick computation of a jacobian for the
    transformation between the two sets of variables.
    Input: array of planar sin/cos comps, ntoroidalcurves, 
           npoloidalcurves, nfp, stellsym, nuniqueshapes,
           winding_surface_function, normaltowinding, order
    Output: xyz dofs for all desired curves
    '''
    from create_windowpanes import planar_vector_list
    sin_components_all = [None]*unique_shapes
    cos_components_all = [None]*unique_shapes
    for i in range(unique_shapes):
        sin_components_all[i] = dofsarr[2*order*(2*i):2*order*(2*i+1)].reshape(order,2)
        cos_components_all[i] = dofsarr[2*order*(2*i+1):2*order*(2*i+2)].reshape(order,2)
    curves_xyz_dofs = np.array([])
    for i in range(int(ntoroidalcurves)):
        tor_angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        for j in range(int(npoloidalcurves)):
            shape = shapes_matrix[i][j]
            if not shape == -1:
                pol_angle = j*(2*np.pi)/npoloidalcurves
                sin_components_this = sin_components_all[shape]
                cos_components_this = cos_components_all[shape]
                # Initialize the components
                components = np.array([],float)
                if normaltowinding:
                    _, normal_vec = winding_surface_function(tor_angle, pol_angle)
                    tor_winding = atan2(normal_vec[1], normal_vec[0])
                    pol_winding = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
                    planar_vectors = planar_vector_list(tor_winding, pol_winding)
                else:
                    planar_vectors = planar_vector_list(tor_angle, pol_angle)
                # Set the components
                for ord in range(order):
                    scomponent = sin_components_this[ord][0]*planar_vectors[0] + sin_components_this[ord][1]*planar_vectors[1]
                    ccomponent = cos_components_this[ord][0]*planar_vectors[0] + cos_components_this[ord][1]*planar_vectors[1]
                    components = np.append(components, scomponent)
                    components = np.append(components, ccomponent)
                curve_xyz_dofs = components.reshape((2*order,3)).T.flatten()
                curves_xyz_dofs = np.append(curves_xyz_dofs, curve_xyz_dofs)
    return curves_xyz_dofs

def change_jacobian():
    '''
    This function creates a jacobian object
    for the sin_cos_components_to_xyz
    function via jax.
    Input: None
    Output: jacobian of sin_cos_components_to_xyz
    '''
    return jax.jacfwd(sin_cos_components_to_xyz)

def multiple_change_jacobian():
    '''
    This function creates a jacobian object
    for the multiple_sin_cos_components_to_xyz
    function via jax.
    Input: None
    Output: jacobian of multiple_sin_cos_components_to_xyz
    '''
    return jax.jacfwd(multiple_sin_cos_components_to_xyz)

def multiple_change_jacobian_2(ntoroidalcurves, npoloidalcurves, nfp, stellsym, unique_shapes,
                                       winding_surface_function = None, normaltowinding=False, order=5):
    """Creates the jacobian between the simsopt dofs and the planar windowpane coil dofs.

    Args:
        ntoroidalcurves (int): Number of rings of windowpane coils
        npoloidalcurves (int): Number of windowpane coils per ring.
        nfp (int): Number of field periods
        stellsym (bool): Boolean defining stellarator symmetry
        unique_shapes (int): Number of unique windowpane shapes across all windowpane coils
        winding_surface_function (func, optional): A function specifying where the centers of the coils were initialized. Defaults to None.
        normaltowinding (bool, optional): Boolean defining whether windowpane area normals are pointed at the center axis (False) or along the normal of the winding surface (True). Defaults to False.
        order (int, optional): The maximum order of Fourier components included in the coils. Defaults to 5.

    Returns:
        2d_np.ndarray: The jacobian determining the transformation from the simsopt dofs basis to the planar windowpane coil dofs basis.
    """
    from create_windowpanes import planar_vector_list
    for i in range(ntoroidalcurves):
        tor_angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        for j in range(npoloidalcurves):
            # Set the shape
            shape = shapes_matrix[i][j]
            if not shape == -1:
                pol_angle = j*(2*np.pi)/npoloidalcurves
                if normaltowinding:
                    _, normal_vec = winding_surface_function(tor_angle, pol_angle)
                    tor_winding = atan2(normal_vec[1], normal_vec[0])
                    pol_winding = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
                    planar_vectors = planar_vector_list(tor_winding, pol_winding)
                else:
                    planar_vectors = planar_vector_list(tor_angle, pol_angle)
                xvec = planar_vectors.T[0]
                yvec = planar_vectors.T[1]
                zvec = planar_vectors.T[2]
                half_jacobian = np.zeros((2*3*order - 1,2*order))
                for ord in range(order):
                    half_jacobian[2*ord][2*ord:2*ord+2] = xvec
                    half_jacobian[2*ord+2*order][2*ord:2*ord+2] = yvec
                    half_jacobian[2*ord+4*order][2*ord:2*ord+2] = zvec
                left_half_jacobian = np.vstack([half_jacobian, np.zeros(2*order)])
                right_half_jacobian = np.vstack([np.zeros(2*order), half_jacobian])

                single_jacobian = np.hstack([left_half_jacobian, right_half_jacobian])
                shapes_row_jacobian = np.zeros((3*2*order, 2*2*order*unique_shapes))
                shapes_row_jacobian[:,2*2*order*shape:2*2*order*(shape+1)] = single_jacobian
                try:
                    jacobian = np.vstack([jacobian, shapes_row_jacobian])
                except:
                    jacobian = shapes_row_jacobian
    return jacobian

def change_arbitrary_windowpanes(curves, curves_xyz_dofs):
    '''
    This function takes a set of curves and a set of x, y,
    and z dofs and sets the dofs of the curves to be
    the input dofs by index.
    Input: set of curves, desired xyz dofs for curves
    Output: None
    '''
    curves_xyz_dofs = curves_xyz_dofs.reshape(((ntoroidalcoils*npoloidalcoils - np.sum(shapes_matrix == -1)),6*order))
    list_dofs = curves_xyz_dofs.tolist()
    for i in range(len(list_dofs)):
        curves[i].x = list_dofs[i]
    return None
