import numpy as np
from math import sin, cos, sqrt, atan2
import jax.numpy as jnp

def maximum_coil_radius(ntoroidalcurves, npoloidalcurves, nfp, stellsym, R0=1.0, R1=0.5):
    '''
    This function provides the maximum radius of the windowpane coils assuming
    they lie on a cirular torus.
    Input the number curves per toroidal spacing of 2pi/nfp/(1+int(stellsym)),
    the number of curves per 2pi of poloidal spacing, the number of field periods,
    whether or not there is stellarator symmetry, the major radius, and the
    minor radius.
    Output the maximum radius a coil can have on a cirucular torus winding
    surface with major radius R0 and minor radius R1 before colliding with
    another coil.
    '''
    poloidal_max_radius = R1*(1-cos((2*np.pi)/npoloidalcurves))/sin((2*np.pi)/npoloidalcurves)
    toroidal_max_radius = (R0-R1)*(1-cos((2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)))/sin((2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves))
    coil_radius = min(poloidal_max_radius, toroidal_max_radius)
    return coil_radius

def create_windowpane_curves(ntoroidalcurves, npoloidalcurves, nfp, stellsym, R0=1.0, R1=0.5, max_radius_prop=0.8, order=1, numquadpoints=None, fixed=True):
    """
    This function is deprecated with comparison to create_arbitrary_windowpanes.
    Create ``ntoroidalcurves * npoloidalcurves`` curves of type
    :obj:`~simsopt.geo.curvexyzfourier.CurveXYZFourier` of order
    ``order`` that will result in circular equally angularly spaced windowpane coils
    with radius ''max_radius_prop'' * {the maximum radius the coils can be without colliding}
    lying on a torus of major radius ``R0`` and minor radius ``R1``
    after applying :obj:`~simsopt.field.coil.coils_via_symmetries`.
    If ''fixed=True'' is specified, the center point of the coils will
    remain fixed in place and will not be dofs.

    Usage example: create 4 rings of 12 base poloidal curves,
    which are then rotated 3 times and
    flipped for stellarator symmetry:

    .. code-block::

        base_curves = create_equally_spaced_windowpane_curves(4, 12, 3, stellsym=True)
        base_currents = [Current(1e5) for c in base_curves]
        coils = coils_via_symmetries(base_curves, base_currents, 3, stellsym=True)
    """
    assert order >= 1, "''order'' must be at least 1 to fully define the windowpane coils"
    if numquadpoints is None:
        numquadpoints = 75
    #Coils have a maximum possible radius before colliding found from geometric considerations
    coil_radius = max_radius_prop*maximum_coil_radius(ntoroidalcurves, npoloidalcurves, nfp, stellsym, R0=R0, R1=R1)
    curves = []
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i in range(ntoroidalcurves):
        tor_angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        for j in range(npoloidalcurves):
            pol_angle = j*(2*np.pi)/npoloidalcurves
            curve = CurveXYZFourier(numquadpoints, order)
            curve.set('xc(0)', (R0+R1*cos(pol_angle))*cos(tor_angle))
            curve.set('xs(1)', coil_radius*sin(tor_angle))
            curve.set('xc(1)', -coil_radius*cos(tor_angle)*sin(pol_angle))
            curve.set('yc(0)', (R0+R1*cos(pol_angle))*sin(tor_angle))
            curve.set('ys(1)', -coil_radius*cos(tor_angle))
            curve.set('yc(1)', -coil_radius*sin(tor_angle)*sin(pol_angle))
            curve.set('zc(0)', R1*sin(pol_angle))
            curve.set('zs(1)', 0)
            curve.set('zc(1)', coil_radius*cos(pol_angle))
            curve.x = curve.x
            if fixed:
                curve.fix('xc(0)')
                curve.fix('yc(0)')
                curve.fix('zc(0)')
            curves.append(curve)
    return curves

def create_windowpane_with_modular_curves(ntoroidalcurves, npoloidalcurves, nfp, stellsym, R0=1.0, R1=0.5, max_radius_prop=0.8, fixed=True, order=1, numquadpoints=None):
    """
    Create ``ntoroidalcurves * npoloidalcurves`` curves of type
    :obj:`~simsopt.geo.curvexyzfourier.CurveXYZFourier` of order
    ``order`` that will result in circular equally angularly spaced coils
    with radius ''max_radius_prop'' * {the maximum radius the coils can be without colliding}
    lying on a torus of major radius ``R0`` and minor radius ``R1``
    after applying :obj:`~simsopt.field.coil.coils_via_symmetries`.
    If ''fixed=True'' is specified, the center point of the coils will
    remain fixed in place and will not be dofs.
    
    Additionally creates ''ntoroidalcurves'' curves of type
    :obj:`~simsopt.geo.curvexyzfourier.CurveXYZFourier` of order
    ``order`` that will result in circular equally spaced coils (major
    radius ``R0`` and minor radius ``R1``) after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries` which are
    in line with the created windowpane coils.

    Usage example: create 4 rings of 12 base poloidal curves
    and 4 fully fixed modular coils, 
    which are then rotated 3 times and
    flipped for stellarator symmetry:

    .. code-block::
    
        windowpane_curves, modular_curves = create_windowpane_with_modular_curves(4, 12, 3, stellsym = True)
        windowpane_currents = [Current(1e5) for c in windowpane_curves]
        modular_currents = [Current(1e3) for c in modular_curves]
        base_curves = windowpane_curves + modular_curves
        base_currents = windowpane_currents + modular_currents
        coils = coils_via_symmetries(base_curves, base_currents, 3, stellsym=True)
    """
    if numquadpoints is None:
        numquadpoints = 75
    windowpane_curves = create_windowpane_curves(ntoroidalcurves, npoloidalcurves, nfp, stellsym, R0=R0, R1=R1, max_radius_prop=max_radius_prop, fixed=fixed, order=order, numquadpoints=numquadpoints)
    max_windowpane_radius = maximum_coil_radius(ntoroidalcurves, npoloidalcurves, nfp, stellsym, R0=R0, R1=R1)
    modular_curves = []
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i in range(ntoroidalcurves):
        curve = CurveXYZFourier(numquadpoints, order=1)
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        curve.set("xc(0)", cos(angle)*R0)
        curve.set("xc(1)", cos(angle)*(R1+max_windowpane_radius))
        curve.set("yc(0)", sin(angle)*R0)
        curve.set("yc(1)", sin(angle)*(R1+max_windowpane_radius))
        # The the next line, the minus sign is for consistency with
        # Vmec.external_current(), so the coils create a toroidal field of the
        # proper sign and free-boundary equilibrium works following stage-2 optimization.
        curve.set("zs(1)", -(R1+max_windowpane_radius))
        curve.x = curve.x  # need to do this to transfer data to C++
        curve.fix_all()
        modular_curves.append(curve)
    return windowpane_curves, modular_curves

def circular_torus_winding(tor_angle, pol_angle, R0=1.0, R1=0.5):
    '''
    This function creates a circular torus. It is the default winding surface
    used by the create_arbitrary_windowpanes function.
    Input a toroidal angle, a poloidal angle, a major radius, and a minor radius.
    Output the x,y,z coordinates of a circular torus with major radius R0
    and minor raidus R1 at the given toroidal and poloidal angles.
    '''
    return [(R0+R1*cos(pol_angle))*cos(tor_angle),
            (R0+R1*cos(pol_angle))*sin(tor_angle),
            R1*sin(pol_angle)]

def clean_components(components, order):
    '''
    This function takes the input component matrix or nested list
    and outputs a cleaned up version of the components.
    Input sine or cosine components and the order of the windowpane
    coils. Confirm the components are the correct shape and
    an allowable size. Then, manipulate them to be a standard
    size and the correct type.
    Output the cleaned up and standardized components.
    '''
    # Make sure the components are numpy arrays
    components = np.array(components)
    # Confirm the shape of the arrays is correct
    assert components.shape[1] == 2, 'Components shape improperly specified'
    assert len(components) <= order, 'Components overspecified compared to order'
    components = np.append(components, np.zeros((order-len(components),2)),axis=0)
    return components

def rotate_windowpane_shapes(sin_components, cos_components, rotation_angle, order):
    '''
    This function rotates the input windowpane coil shape around it's normal vector.
    Input the sine and cosine components (in the format of the components
    for the create_arbitrary_windowpanes function), a rotation angle, and
    the order of the windowpane coils.
    Output the sine and cosine components with a rotation applied of
    rotation angle.
    '''
    for i in range(order):
        rot_mat = np.append(sin_components[i],cos_components[i]).reshape(2,2)
        rot_mat = rot_mat * cos(rotation_angle) + np.fliplr(rot_mat)*np.array([[1,-1],[1,-1]])*sin(rotation_angle)
        sin_components[i] = rot_mat[0]
        cos_components[i] = rot_mat[1]
    return sin_components, cos_components

def planar_vector_list(tor_angle, pol_angle):
    '''
    This function outputs the two vectors needed to create planar coils
    which are each orthogonal to each other and the normal vector
    specified by the input toroidal and poloidal angles.
    Input toroidal angle and poloidal angle.
    Output two vectors which span the plane with normal vector
    defined by the toroidal angle and the poloidal angle.
    These vectors are:
    (sin(toroidal angle), -cos(toroidal angle), 0)
    (-cos(toroidal angle) * sin(poloidal angle), -sin(toroidal angle) * sin(poloidal angle), cos(poloidal angle))
    '''
    return jnp.array([[jnp.sin(tor_angle),-jnp.cos(tor_angle),0],
                     [-jnp.cos(tor_angle)*jnp.sin(pol_angle),-jnp.sin(tor_angle)*jnp.sin(pol_angle),jnp.cos(pol_angle)]], float)

def create_arbitrary_windowpanes(ntoroidalcurves, npoloidalcurves, nfp, stellsym, sin_components, cos_components,
                                rotation_angle=0, winding_surface_function = None, fixed=True, normaltowinding=False,
                                order=5, numquadpoints=None):
    '''
    Create ``ntoroidalcurves * npoloidalcurves`` curves of type
    :obj:`~simsopt.geo.curvexyzfourier.CurveXYZFourier` of order
    ``order`` that will result in equally angularly spaced windowpane coils
    lying on a winding surface specified by the input function
    ''winding_surface_function'' or the default winding surface function
    ''circular_torus_winding'' after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries`.
    The shape of these coils is specified by ''sin_components'' and
    ''cos_components'' which are nested lists or numpy arrays in the
    format [[planar_vector_1_order_1_component, planar_vector_2_order_1_component],
    [pv1order2c,pv2order2c],...], where planar vector 1 is the first vector specified
    by the planar_vector_list function and planar vector 2 is the second
    vector specified by the planar_vector_list function. The desired shape is
    then rotated by an angle ''rotation_angle''.
    If ''fixed=True'' is specified, the center point of the coils will
    remain fixed in place and will not be dofs.
    If ''normaltowinding=True'' is specified, the coils will be initialized
    with their normal vectors pointing along the normal vector of the
    winding surface instead of towards the primary axis of the stellarator.
    Additionally, if ''normaltowinding=True'' is specified, the input winding 
    surface function must also output its normal vector in the form:
    return xyz_location, normal_vector
    
    
    There is a known bug with normaltowinding=True where the coil shapes
    are rotated for pi/2 < poloidal angle < 3pi/2.

    Usage example: create 4 rings of 12 base poloidal curves, 
    which are then rotated 3 times and
    flipped for stellarator symmetry:

    .. code-block::

        sin_components = [[0.068,0],[0,0.034]]
        cos_components = [[0,0.034]]
        base_curves = create_arbitrary_windowpanes(4, 12, 3, stellsym=True, sin_components, cos_components)
        base_currents = [Current(1e5) for c in base_curves]
        coils = coils_via_symmetries(base_curves, base_currents, 3, stellsym=True)
    '''
    if numquadpoints is None:
        numquadpoints = order * 15
    if winding_surface_function is None:
        winding_surface_function = circular_torus_winding
    if not normaltowinding:
        assert len(winding_surface_function(0,0)) == 3, 'Winding surface not specified as a 3-vector'
    else:
        assert len(winding_surface_function(0,0)[0]) == 3, 'Winding surface not specified as a 3-vector'
    # Clean up the component lists - make arrays, correct length, etc.
    sin_components = clean_components(sin_components, order)
    cos_components = clean_components(cos_components, order)
    # Rotate the windowpane coil shapes if there is a rotation angle
    if not rotation_angle==0:
        sin_components, cos_components = rotate_windowpane_shapes(sin_components, cos_components, rotation_angle, order)
    curves=[]
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i in range(ntoroidalcurves):
        tor_angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        for j in range(npoloidalcurves):
            pol_angle = j*(2*np.pi)/npoloidalcurves
            curve = CurveXYZFourier(numquadpoints, order)
            # Initialize the components
            components = np.array([[0.0]*3 for k in range(1+2*order)])
            if normaltowinding:
                comps, normal_vec = winding_surface_function(tor_angle, pol_angle)
                components[0] = list(comps)
                tor_winding = atan2(normal_vec[1], normal_vec[0])
                pol_winding = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
                '''
                # Attempts at solving a known bug
                if (np.pi/2) < pol_angle <= (3*np.pi/2):
                    pol_winding = np.pi - pol_winding
                    tor_winding = np.pi - tor_winding
                '''
                planar_vectors = planar_vector_list(tor_winding, pol_winding)
            else:
                components[0] = winding_surface_function(tor_angle, pol_angle)
                planar_vectors = planar_vector_list(tor_angle, pol_angle)
            # Set the sin components
            for s in range(order):
                components[2*s+1] = sin_components[s][0]*planar_vectors[0] + sin_components[s][1]*planar_vectors[1]
            # Set the cos components
            for c in range(order):
                components[2*c+2] = cos_components[c][0]*planar_vectors[0] + cos_components[c][1]*planar_vectors[1]
            curve.x = list(components.T.flatten())
            curve.x = curve.x
            if fixed:
                curve.fix('xc(0)')
                curve.fix('yc(0)')
                curve.fix('zc(0)')
            curves.append(curve)
    return curves

def create_multiple_arbitrary_windowpanes(ntoroidalcurves, npoloidalcurves, nfp, stellsym, sin_components_0,
                                          cos_components_0, unique_shapes = 1, shapes_matrix = None,
                                          rotation_angles=[0], winding_surface_function = None, fixed=True,
                                          normaltowinding=False,
                                          order=5, numquadpoints=None, **sin_cos_comps):
    '''
    Create ``ntoroidalcurves * npoloidalcurves`` curves of type
    :obj:`~simsopt.geo.curvexyzfourier.CurveXYZFourier` of order
    ``order`` that will result in equally angularly spaced windowpane coils
    lying on a winding surface specified by the input function
    ''winding_surface_function'' or the default winding surface function
    ''circular_torus_winding'' after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries`.
    The shapes of these coils are specified by ''sin_components_0'' and
    ''cos_components_0'' as well as any **kwargs, ''sin_components_1'', ''cos_components_1'',
    ''sin_components_2'', cos_components_2'', etc. which are nested lists or numpy arrays in the
    format [[planar_vector_1_order_1_component, planar_vector_2_order_1_component],
    [pv1order2c,pv2order2c],...], where planar vector 1 is the first vector specified
    by the planar_vector_list function and planar vector 2 is the second
    vector specified by the planar_vector_list function. The desired shape is
    then rotated by an angle ''rotation_angles[shape number]''.
    The number of unique shapes is given by ''unique_shapes''.
    If ''shapes_matrix'' is specified, it must be a matrix or list with shape
    ``ntoroidalcurves'' by ''npoloidalcurves`` whose entries are 0,1,..., ''unique_shapes'' - 1.
    This matrix specifies the locations of the shapes on the winding surface. If -1 is used for an
    entry, it will result in no coil being initialized in that location.
    If ''shapes_matrix'' is not specified, the shapes will be laid out as one coil of each shape
    and then repeat this pattern.
    If ''fixed=True'' is specified, the center point of the coils will
    remain fixed in place and will not be dofs.
    If ''normaltowinding=True'' is specified, the coils will be initialized
    with their normal vectors pointing along the normal vector of the
    winding surface instead of towards the primary axis of the stellarator.
    Additionally, if ''normaltowinding=True'' is specified, the input winding 
    surface function must also output its normal vector in the form:
    return xyz_location, normal_vector
    
    
    There is a known bug with normaltowinding=True where the coil shapes
    are rotated for pi/2 < poloidal angle < 3pi/2.

    Usage example: create 4 rings of 12 base poloidal curves, 
    which are then rotated 3 times and
    flipped for stellarator symmetry:

    .. code-block::

        sin_components_0 = [[0.068,0],[0,0.034]]
        cos_components_0 = [[0,0.034]]
        sin_components_1 = [[1,0]]
        cos_components_1 = [[0,0.5]]
        base_curves = create_multiple_arbitrary_windowpanes(4, 12, 3, stellsym=True, sin_components_0, cos_components_0
                                                   sin_components_1 = sin_components_1, cos_components_1 = cos_components_1)
        base_currents = [Current(1e5) for c in base_curves]
        coils = coils_via_symmetries(base_curves, base_currents, 3, stellsym=True)
    '''
    shapes_matrix = np.array(shapes_matrix)
    if shapes_matrix is None:
        tile = np.array(range(unique_shapes))
        shapes_matrix = np.resize(tile, (ntoroidalcurves, npoloidalcurves))
    else:
        assert shapes_matrix.shape == (ntoroidalcurves, npoloidalcurves)
    assert len(rotation_angles) == unique_shapes, 'Rotation angles must be a list of length unique_shapes'
    if numquadpoints is None:
        numquadpoints = order * 15
    if winding_surface_function is None:
        winding_surface_function = circular_torus_winding
    if not normaltowinding:
        assert len(winding_surface_function(0,0)) == 3, 'Winding surface not specified as a 3-vector'
    else:
        assert len(winding_surface_function(0,0)[0]) == 3, 'Winding surface not specified as a 3-vector'
    sin_components_all = [None]*unique_shapes
    cos_components_all = [None]*unique_shapes
    for i in range(unique_shapes):
        # Clean up all components
        if not i == 0:
            try:
                sin_components = clean_components(sin_cos_comps['sin_components_{}'.format(i)], order)
            except:
                sin_components = clean_components([[np.max(sin_components_0),0]], order)
            try:
                cos_components = clean_components(sin_cos_comps['cos_components_{}'.format(i)], order)
            except:
                cos_components = clean_components([[0,np.max(cos_components_0)]], order)
        else:
            sin_components = clean_components(sin_components_0, order)
            cos_components = clean_components(cos_components_0, order)
        # Rotate the windowpane coil shapes if there is a rotation angle
        if not rotation_angles[i]==0:
            sin_components, cos_components = rotate_windowpane_shapes(sin_components, cos_components, rotation_angles[i], order)
        sin_components_all[i] = sin_components
        cos_components_all[i] = cos_components
    curves=[]
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i in range(ntoroidalcurves):
        tor_angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        for j in range(npoloidalcurves):
            # Set the shape
            shape = shapes_matrix[i][j]
            if not shape == -1:
                pol_angle = j*(2*np.pi)/npoloidalcurves
                sin_components_this = sin_components_all[shape]
                cos_components_this = cos_components_all[shape]
                curve = CurveXYZFourier(numquadpoints, order)
                # Initialize the components
                components = np.array([[0.0]*3 for k in range(1+2*order)])
                if normaltowinding:
                    comps, normal_vec = winding_surface_function(tor_angle, pol_angle)
                    components[0] = list(comps)
                    tor_winding = atan2(normal_vec[1], normal_vec[0])
                    pol_winding = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
                    '''
                    # Attempts at solving a known bug
                    if (np.pi/2) < pol_angle <= (3*np.pi/2):
                        pol_winding = np.pi - pol_winding
                        tor_winding = np.pi - tor_winding
                    '''
                    planar_vectors = planar_vector_list(tor_winding, pol_winding)
                else:
                    components[0] = winding_surface_function(tor_angle, pol_angle)
                    planar_vectors = planar_vector_list(tor_angle, pol_angle)
                # Set the sin components
                for s in range(order):
                    components[2*s+1] = sin_components_this[s][0]*planar_vectors[0] + sin_components_this[s][1]*planar_vectors[1]
                # Set the cos components
                for c in range(order):
                    components[2*c+2] = cos_components_this[c][0]*planar_vectors[0] + cos_components_this[c][1]*planar_vectors[1]
                curve.x = list(components.T.flatten())
                curve.x = curve.x
                if fixed:
                    curve.fix('xc(0)')
                    curve.fix('yc(0)')
                    curve.fix('zc(0)')
                curves.append(curve)
    return curves
