import numpy as np
import jax.numpy as jnp
from math import sin, cos, sqrt, atan2
from Cyclone.axis_functions import import_axis
from Cyclone.surface_functions import import_surface
from Cyclone.helper_functions import maximum_coil_radius, clean_components, planar_vector_list, rotate_windowpane_shapes
from Cyclone.angle_generation import create_toroidal_angles, create_poloidal_angles, check_tor_pol_rotation_angles
from Cyclone.fixed_functions import fix_curve, return_unfixed_orders, all_planar_dofs, unfixed_planar_dofs, all_nonplanar_dofs, unfixed_nonplanar_dofs, unfixed_simsopt_dofs

def unpack_all_components(sin_cos_components, order, unique_shapes, coil_radius):
    if 'sin_cos_components' in sin_cos_components and len(sin_cos_components) == 1:
        sin_cos_components = sin_cos_components['sin_cos_components']
    if 'sin_cos_components' in sin_cos_components and len(sin_cos_components) != 1:
        raise ValueError('Dictionary input used along with secondary inputs to **kwargs is not a valid input.')
    sin_components_all = [None]*unique_shapes
    cos_components_all = [None]*unique_shapes
    nonplanar_sin_components_all = [None]*unique_shapes
    nonplanar_cos_components_all = [None]*unique_shapes
    for i in range(unique_shapes):
        # Clean up all components
        if 'sin_components_{}'.format(i) in sin_cos_components:
            sin_components_all[i] = clean_components(sin_cos_components['sin_components_{}'.format(i)], order)
        else:
            sin_components_all[i] = clean_components([[coil_radius,0]], order)
        if 'cos_components_{}'.format(i) in sin_cos_components:
            cos_components_all[i] = clean_components(sin_cos_components['cos_components_{}'.format(i)], order)
        else:
            cos_components_all[i] = clean_components([[0,coil_radius]], order)
        if 'nonplanar_sin_components_{}'.format(i) in sin_cos_components:
            nonplanar_sin_components_all[i] = np.array(sin_cos_components['nonplanar_sin_components_{}'.format(i)])
        else:
            nonplanar_sin_components_all[i] = np.zeros(order)
        if 'nonplanar_cos_components_{}'.format(i) in sin_cos_components:
            nonplanar_cos_components_all[i] = np.array(sin_cos_components['nonplanar_cos_components_{}'.format(i)])
        else:
            nonplanar_cos_components_all[i] = np.zeros(order)
    return sin_components_all, cos_components_all, nonplanar_sin_components_all, nonplanar_cos_components_all

def create_shapes_matrix(tile_as, unique_shapes, shapes_size):
    if type(shapes_size) == int:
        shapes_size = (shapes_size,)
    if len(shapes_size) == 1 or (len(shapes_size) == 2 and type(shapes_size[1]) == int):
        if type(tile_as) == str and tile_as.lower() == 'tile':
            tile = np.array(range(unique_shapes))
            shapes_matrix = np.resize(tile, shapes_size)
        elif type(tile_as) == str and tile_as.lower() == 'random':
            shapes = np.append(-1, np.arange(unique_shapes))
            shapes_matrix = np.random.choice(shapes, shapes_size)
        elif type(tile_as) == list:
            shapes_matrix = np.array(tile_as)
            assert shapes_matrix.shape == shapes_size, 'Length of tile_as list is not equal to number of coils.'
        elif type(tile_as) == np.ndarray:
            assert tile_as.shape == shapes_size, 'Length of tile_as array is not equal to number of coils.'
            shapes_matrix = tile_as
        else:
            raise TypeError('tile_as could not be interpreted as a valid option.')
        return shapes_matrix
    elif len(shapes_size) == 2:
        n_shapes = sum(shapes_size[1])
        shapes_size = (shapes_size[0], *[shapes_size[1][i] for i in range(len(shapes_size[1]))])
    else:
        n_shapes = sum(shapes_size[1:])
    n_rings = shapes_size[0]
    assert len(shapes_size[1:]) == n_rings, 'Number of curve rings specified is different from number expected from shapes_size[0].'
    shapes_size = shapes_size[1:]
    shapes_matrix = [[] for i in range(n_rings)]
    tally = 0
    if type(tile_as) == str and tile_as.lower() == 'tile':
        tile = np.array(range(unique_shapes))
        full_tile = np.resize(tile, (n_shapes,))
        for i, num in enumerate(shapes_size):
            shapes_matrix[i] = full_tile[tally:(tally+num)]
            tally += num
    elif type(tile_as) == str and tile_as.lower() == 'random':
        shapes = np.append(-1, np.arange(unique_shapes))
        full_tile = np.random.choice(shapes, (n_shapes,))
        for i, num in enumerate(shapes_size):
            shapes_matrix[i] = full_tile[tally:(tally+num)]
            tally += num
    elif type(tile_as) == list:
        shapes_matrix = tile_as
        for i in range(n_rings):
            assert len(shapes_matrix[i]) == shapes_size[i], 'Size of tile_as list is not equal to number of coils.'
    elif type(tile_as) == np.ndarray:
        shapes_matrix = list(tile_as)
        for i in range(n_rings):
            shapes_matrix[i] = list(shapes_matrix[i])
            assert len(shapes_matrix[i]) == shapes_size[i], 'Size of tile_as array is not equal to number of coils.'
    else:
        raise TypeError('tile_as could not be interpreted as a valid option.')
    return shapes_matrix

def init_simsopt_stellarator_coils(axis, ncurves, R1 = 0.5, order=1, numquadpoints = None, fixed='all', **dofs):
    nfp, stellsym, axis_function, _ = import_axis(axis)
    if 'dofs' in dofs and len(dofs) == 1:
        dofs = dofs['dofs']
    if 'dofs' in dofs and len(dofs) != 1:
        raise ValueError('Dictionary input used along with secondary inputs to **kwargs is not a valid input.')
    ncurves, toroidal_angles = create_toroidal_angles(ncurves, nfp, stellsym)
    if numquadpoints is None:
        numquadpoints = 15 * order
    curves = []
    centers = []
    center_tors = []
    simsopt_dofs_all = [None] * ncurves
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i, angle in enumerate(toroidal_angles):
        curve = CurveXYZFourier(numquadpoints, order)
        centers.append(axis_function(angle))
        center_tors.append(angle)
        curve.set("xc(0)", centers[-1][0])
        curve.set("yc(0)", centers[-1][1])
        curve.set("zc(0)", centers[-1][2])
        curve.x = curve.x
        curve.fix("xc(0)")
        curve.fix("yc(0)")
        curve.fix("zc(0)")
        curve.set("xc(1)", cos(angle)*R1)
        curve.set("yc(1)", sin(angle)*R1)
        curve.set("zs(1)", -R1)
        if 'dofs_{}'.format(i) in dofs:
            curve.x = dofs['dofs_{}'.format(i)]
        curve.x = curve.x
        simsopt_dofs_all[i] = curve.x
        curve.unfix_all()
        fix_curve(curve, order, fixed)
        curves.append(curve)
    ncurves = len(curves)
    unfixed_orders = return_unfixed_orders(order, fixed)
    simsopt_dofs = unfixed_simsopt_dofs(simsopt_dofs_all, unfixed_orders)
    return curves, ncurves, unfixed_orders, centers, center_tors, simsopt_dofs_all, simsopt_dofs, axis_function

def init_stellarator_coils(axis, ncurves, unique_shapes=None, tor_angles=None, pol_angles=None, rotation_vector=None, tile_as='tile', R1 = 0.5, order=1, numquadpoints=None, fixed='all', **sin_cos_components):
    ## Axis
    nfp, stellsym, axis_function, normal_vec_function = import_axis(axis)
    ## curve locations
    ncurves, toroidal_angles = create_toroidal_angles(ncurves, nfp, stellsym)
    ## unique shapes
    if (unique_shapes is None) or (unique_shapes == 0):
        unique_shapes = ncurves
    ## components
    sin_components_all, cos_components_all, nonplanar_sin_components_all, nonplanar_cos_components_all = unpack_all_components(sin_cos_components, order, unique_shapes, R1)
    ## verify angles are all specified correctly if specified
    tor_angles, pol_angles, rotation_vector = check_tor_pol_rotation_angles(tor_angles, pol_angles, rotation_vector, (ncurves,))
    ## tile as
    shapes_vector = create_shapes_matrix(tile_as, unique_shapes, (ncurves,))
    ## quad points
    if numquadpoints is None:
        numquadpoints = 15 * order
    curves = []
    rotation_angles = []
    curve_shapes = []
    centers = []
    center_tors = []
    normal_tors = []
    normal_pols = []
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i, angle in enumerate(toroidal_angles):
        shape = shapes_vector[i]
        if shape == -1:
            continue
        curve_shapes.append(shape)
        rotation_angle = rotation_vector[i]
        curve = CurveXYZFourier(numquadpoints, order=order)
        components = np.array([[0.0]*3 for k in range(1+2*order)])
        ## c0 components
        components[0] = axis_function(angle)
        centers.append(components[0])
        center_tors.append(angle)
        ## Get angles
        if tor_angles is None:
            normal_vec = normal_vec_function(angle)
            tor_axis = atan2(normal_vec[1], normal_vec[0])
            pol_axis = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
        else:
            tor_axis = tor_angles[i]
            pol_axis = pol_angles[i]
        normal_tors.append(tor_axis)
        normal_pols.append(pol_axis)
        ## Set rest of components
        planar_vectors = planar_vector_list(tor_axis, pol_axis)
        ## I could make this a function? Not sure if it's worth it or not.
        normal_vector = jnp.array([jnp.cos(tor_axis)*jnp.cos(pol_axis), jnp.sin(tor_axis)*jnp.cos(pol_axis), jnp.sin(pol_axis)])
        sin_components_this_unrot = sin_components_all[shape]
        cos_components_this_unrot = cos_components_all[shape]
        sin_components_this, cos_components_this = rotate_windowpane_shapes(sin_components_this_unrot, cos_components_this_unrot, rotation_angle, order)
        nonplanar_sin_components_this = nonplanar_sin_components_all[shape]
        nonplanar_cos_components_this = nonplanar_cos_components_all[shape]
        for s in range(order):
            components[2*s+1] = sin_components_this[s][0]*planar_vectors[0] + sin_components_this[s][1]*planar_vectors[1] + nonplanar_sin_components_this[s]*normal_vector
            components[2*s+2] = cos_components_this[s][0]*planar_vectors[0] + cos_components_this[s][1]*planar_vectors[1] + nonplanar_cos_components_this[s]*normal_vector
        curve.x = list(components.T.flatten())
        curve.x = curve.x  # need to do this to transfer data to C++
        fix_curve(curve, order, fixed)
        curves.append(curve)
        rotation_angles.append(rotation_angle)
    ncurves = len(curves)
    unfixed_orders = return_unfixed_orders(order, fixed)
    full_planar_dofs = all_planar_dofs(sin_components_all, cos_components_all)
    planar_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    full_nonplanar_dofs = all_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all)
    nonplanar_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    return curves, ncurves, unfixed_orders, planar_dofs, full_planar_dofs, nonplanar_dofs, full_nonplanar_dofs, normal_tors, normal_pols, rotation_angles, centers, center_tors, unique_shapes, shapes_vector, curve_shapes, axis_function

def init_simsopt_windowpane_coils(surface, ntoroidalcurves, npoloidalcurves, R0=1., R1=0.5, coil_radius = None, order=5, numquadpoints=None, fixed='all', normal_to_winding=False, surface_extension=0., **dofs):
    nfp, stellsym, surface_function, _ = import_surface(surface, surface_extension, normal_to_winding)
    ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves, nfp, stellsym)
    ## curve poloidal locations
    npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves, ntoroidalcurves)
    if coil_radius is None:
        coil_radius = maximum_coil_radius(ntoroidalcurves, max(npoloidalcurves), nfp, stellsym, R0=R0, R1=R1)
    if numquadpoints is None:
        numquadpoints = 15 * order
    count = 0
    curves = []
    centers = []
    center_tors = []
    center_pols = []
    simsopt_dofs_all = [None] * sum(npoloidalcurves)
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i, tor_angle in enumerate(toroidal_angles):
        for j, pol_angle in enumerate(poloidal_angles[i]):
            curve = CurveXYZFourier(numquadpoints, order=order)
            centers.append(surface_function(tor_angle, pol_angle))
            center_tors.append(tor_angle)
            center_pols.append(pol_angle)
            curve.set("xc(0)", centers[-1][0])
            curve.set("yc(0)", centers[-1][1])
            curve.set("zc(0)", centers[-1][2])
            curve.x = curve.x
            curve.fix("xc(0)")
            curve.fix("yc(0)")
            curve.fix("zc(0)")
            curve.set("xs(1)", coil_radius * sin(tor_angle))
            curve.set("ys(1)", -coil_radius * cos(tor_angle))
            curve.set("xc(1)", -coil_radius * cos(tor_angle) * sin(pol_angle))
            curve.set("yc(1)", -coil_radius * sin(tor_angle) * sin(pol_angle))
            curve.set("zc(1)", coil_radius * cos(pol_angle))
            if 'dofs_{}_{}'.format(i,j) in dofs:
                curve.x = dofs['dofs_{}_{}'.format(i,j)]
            curve.x = curve.x
            simsopt_dofs_all[count] = curve.x
            count += 1
            curve.unfix_all()
            fix_curve(curve, order, fixed)
            curves.append(curve)
    unfixed_orders = return_unfixed_orders(order, fixed)
    print(simsopt_dofs_all)
    simsopt_dofs = unfixed_simsopt_dofs(simsopt_dofs_all, unfixed_orders)
    return curves, ntoroidalcurves, npoloidalcurves, unfixed_orders, centers, center_tors, center_pols, simsopt_dofs_all, simsopt_dofs, surface_function

def init_windowpane_coils(surface, ntoroidalcurves, npoloidalcurves, unique_shapes=None, tor_angles=None, pol_angles=None, rotation_matrix=None, tile_as='tile', R0=1., R1=0.5, coil_radius = None, order=5, numquadpoints=None, fixed='all', normal_to_winding=False, surface_extension=0., **sin_cos_components):
    ## Surface
    nfp, stellsym, surface_function, normal_vec_function = import_surface(surface, surface_extension, normal_to_winding)
    ## curve toroidal locations
    ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves, nfp, stellsym)
    ## curve poloidal locations
    npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves, ntoroidalcurves)
    ## unique shapes
    if (unique_shapes is None) or (unique_shapes == 0):
        unique_shapes = sum(npoloidalcurves)
    ## components
    if coil_radius is None:
        coil_radius = maximum_coil_radius(ntoroidalcurves, max(npoloidalcurves), nfp, stellsym, R0=R0, R1=R1)
    sin_components_all, cos_components_all, nonplanar_sin_components_all, nonplanar_cos_components_all = unpack_all_components(sin_cos_components, order, unique_shapes, coil_radius)
    ## verify angles are all specified correctly if specified
    tor_angles, pol_angles, rotation_matrix = check_tor_pol_rotation_angles(tor_angles, pol_angles, rotation_matrix, (ntoroidalcurves,npoloidalcurves))
    ## tile as
    shapes_matrix = create_shapes_matrix(tile_as, unique_shapes, (ntoroidalcurves,npoloidalcurves))
    ## quad points
    if numquadpoints is None:
        numquadpoints = 15 * order
    curves = []
    rotation_angles = []
    curve_shapes = []
    centers = []
    center_tors = []
    center_pols = []
    normal_tors = []
    normal_pols = []
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i, tor_angle in enumerate(toroidal_angles):
        for j, pol_angle in enumerate(poloidal_angles[i]):
            shape = shapes_matrix[i][j]
            if shape == -1:
                continue
            curve_shapes.append(shape)
            rotation_angle = rotation_matrix[i][j]
            curve = CurveXYZFourier(numquadpoints, order=order)
            components = np.array([[0.0]*3 for k in range(1+2*order)])
            ## c0 components
            components[0] = surface_function(tor_angle, pol_angle)
            centers.append(components[0])
            center_tors.append(tor_angle)
            center_pols.append(pol_angle)
            ## Get angles
            if tor_angles is None:
                normal_vec = normal_vec_function(tor_angle, pol_angle)
                tor_surf = atan2(normal_vec[1], normal_vec[0])
                pol_surf = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
            else:
                tor_surf = tor_angles[i][j]
                pol_surf = pol_angles[i][j]
            normal_tors.append(tor_surf)
            normal_pols.append(pol_surf)
            ## Set rest of components
            planar_vectors = planar_vector_list(tor_surf, pol_surf)
            ## I could make this a function? Not sure if it's worth it or not.
            normal_vector = jnp.array([jnp.cos(tor_surf)*jnp.cos(pol_surf), jnp.sin(tor_surf)*jnp.cos(pol_surf), jnp.sin(pol_surf)])
            sin_components_this_unrot = sin_components_all[shape]
            cos_components_this_unrot = cos_components_all[shape]
            sin_components_this, cos_components_this = rotate_windowpane_shapes(sin_components_this_unrot, cos_components_this_unrot, rotation_angle, order)
            nonplanar_sin_components_this = nonplanar_sin_components_all[shape]
            nonplanar_cos_components_this = nonplanar_cos_components_all[shape]
            for s in range(order):
                components[2*s+1] = sin_components_this[s][0]*planar_vectors[0] + sin_components_this[s][1]*planar_vectors[1] + nonplanar_sin_components_this[s]*normal_vector
                components[2*s+2] = cos_components_this[s][0]*planar_vectors[0] + cos_components_this[s][1]*planar_vectors[1] + nonplanar_cos_components_this[s]*normal_vector
            curve.x = list(components.T.flatten())
            curve.x = curve.x  # need to do this to transfer data to C++
            fix_curve(curve, order, fixed)
            curves.append(curve)
            rotation_angles.append(rotation_angle)
    unfixed_orders = return_unfixed_orders(order, fixed)
    full_planar_dofs = all_planar_dofs(sin_components_all, cos_components_all)
    planar_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    full_nonplanar_dofs = all_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all)
    nonplanar_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    return curves, ntoroidalcurves, npoloidalcurves, unfixed_orders, planar_dofs, full_planar_dofs, nonplanar_dofs, full_nonplanar_dofs, normal_tors, normal_pols, rotation_angles, centers, center_tors, center_pols, unique_shapes, shapes_matrix, curve_shapes, surface_function