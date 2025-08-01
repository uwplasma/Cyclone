import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from helper_functions import planar_vector_list, rotate_windowpane_shapes
from fixed_functions import update_all_planar_dofs_from_unfixed_planar_dofs, update_all_nonplanar_dofs_from_unfixed_nonplanar_dofs, update_all_simsopt_dofs_from_unfixed_simsopt_dofs

def unpack_curve_centers(dofs, ncurves, center_opt_flag, center_opt_type_flag):
    if center_opt_flag:
        if center_opt_type_flag == 'direct':
            ncenter_dofs = 3*ncurves
            curve_centers = dofs[:ncenter_dofs].reshape((ncurves, 3))
        elif center_opt_type_flag == 'on_axis':
            ncenter_dofs = ncurves
            curve_centers = dofs[:ncenter_dofs].reshape(ncurves, 1)
        elif center_opt_type_flag == 'on_surface':
            ncenter_dofs = 2*ncurves
            curve_centers = dofs[:ncenter_dofs].reshape((ncurves,2), order='F')
        else:
            raise ValueError('center_opt_type_flag could not be interpreted as a valid input.')
    else:
        ncenter_dofs = 0
        curve_centers = None
    return ncenter_dofs, curve_centers

@partial(jax.jit, static_argnames=['rotation_flag', 'normal_flag', 'planar_flag', 'planar_opt_flag', 'nonplanar_opt_flag', 'center_opt_flag', 'center_opt_type_flag', 'unique_shapes', 'ncurves', 'unfixed_orders', 'curve_shapes', 'axis_function'])
def generate_simsopt_stellarator_dofs(cyclone_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag='direct', axis_function=None):
    sin_components_all = [None] * unique_shapes
    cos_components_all = [None] * unique_shapes
    if not planar_flag:
        nonplanar_sin_components_all = [None] * unique_shapes
        nonplanar_cos_components_all = [None] * unique_shapes
    num_orders = len(unfixed_orders)
    if 0 in unfixed_orders:
        num_orders -= 1
    ncenter_dofs, curve_centers = unpack_curve_centers(cyclone_dofs, ncurves, center_opt_flag, center_opt_type_flag)
    nplanar_dofs = 0
    nnonplanar_dofs = 0
    for i in range(unique_shapes):
        if planar_opt_flag:
            nplanar_dofs = 2*num_orders*2*unique_shapes
            sin_components_all[i] = cyclone_dofs[(ncenter_dofs + 2*num_orders*(2*i)):(ncenter_dofs + 2*num_orders*(2*i+1))].reshape(num_orders, 2)
            cos_components_all[i] = cyclone_dofs[(ncenter_dofs + 2*num_orders*(2*i+1)):(ncenter_dofs + 2*num_orders*(2*i+2))].reshape(num_orders, 2)
        else:
            sin_components_all[i] = jnp.array(planar_dofs[(2*num_orders*(2*i)):(2*num_orders*(2*i+1))]).reshape(num_orders, 2)
            cos_components_all[i] = jnp.array(planar_dofs[(2*num_orders*(2*i+1)):(2*num_orders*(2*i+2))]).reshape(num_orders, 2)
        if not planar_flag and nonplanar_opt_flag:
            nnonplanar_dofs = 2*num_orders*unique_shapes
            nonplanar_sin_components_all[i] = cyclone_dofs[(ncenter_dofs + nplanar_dofs + num_orders*(2*i)):(ncenter_dofs + nplanar_dofs + num_orders*(2*i+1))].reshape(num_orders, 1)
            nonplanar_cos_components_all[i] = cyclone_dofs[(ncenter_dofs + nplanar_dofs + num_orders*(2*i+1)):(ncenter_dofs + nplanar_dofs + num_orders*(2*i+2))].reshape(num_orders, 1)
        elif not planar_flag:
            nonplanar_sin_components_all[i] = jnp.array(nonplanar_dofs[(num_orders*(2*i)):(num_orders*(2*i+1))]).reshape(num_orders, 1)
            nonplanar_cos_components_all[i] = jnp.array(nonplanar_dofs[(num_orders*(2*i+1)):(num_orders*(2*i+2))]).reshape(num_orders, 1)
    if rotation_flag:
        nrotation_dofs = ncurves
        rotation_angles = cyclone_dofs[(ncenter_dofs + nplanar_dofs + nnonplanar_dofs):(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs)]
    else:
        assert len(rotation_angles) == ncurves, 'Number of specified rotation angles is not equal to the number of curves.'
        nrotation_dofs = 0
    if normal_flag:
        nnormal_dofs = 2*ncurves
        normal_tors = cyclone_dofs[(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs):(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs + int(nnormal_dofs/2))]
        normal_pols = cyclone_dofs[(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs + int(nnormal_dofs/2)):(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs + nnormal_dofs)]
    else:
        assert len(normal_tors) == ncurves, 'Number of specified normal toroidal angles is not equal to the number of curves.'
        assert len(normal_pols) == ncurves, 'Number of specified normal poloidal angles is not equal to the number of curves.'
        nnormal_dofs = 0
    simsopt_dofs = jnp.array([],float)
    for i in range(ncurves):
        shape = curve_shapes[i]
        rotation_angle = rotation_angles[i]
        normal_tor = normal_tors[i]
        normal_pol = normal_pols[i]
        sin_components_this_unrot = sin_components_all[shape]
        cos_components_this_unrot = cos_components_all[shape]
        sin_components_this, cos_components_this = rotate_windowpane_shapes(sin_components_this_unrot, cos_components_this_unrot, rotation_angle, num_orders)
        if not planar_flag:
            nonplanar_sin_components_this = nonplanar_sin_components_all[shape]
            nonplanar_cos_components_this = nonplanar_cos_components_all[shape]
        else:
            nonplanar_sin_components_this = [0] * num_orders
            nonplanar_cos_components_this = [0] * num_orders
        components = jnp.array([],float)
        if center_opt_flag:
            if center_opt_type_flag == 'direct':
                components = jnp.append(components, curve_centers[i])
            elif center_opt_type_flag == 'on_axis':
                components = jnp.append(components, axis_function(curve_centers[i][0]))
            else:
                raise ValueError('Should never get here.')
        planar_vectors = planar_vector_list(normal_tor, normal_pol)
        normal_vector = jnp.array([jnp.cos(normal_tor)*jnp.cos(normal_pol), jnp.sin(normal_tor)*jnp.cos(normal_pol), jnp.sin(normal_pol)])
        for ord in range(num_orders):
            scomponent = sin_components_this[ord][0]*planar_vectors[0] + sin_components_this[ord][1]*planar_vectors[1] + nonplanar_sin_components_this[ord]*normal_vector
            ccomponent = cos_components_this[ord][0]*planar_vectors[0] + cos_components_this[ord][1]*planar_vectors[1] + nonplanar_cos_components_this[ord]*normal_vector
            components = jnp.append(components, scomponent)
            components = jnp.append(components, ccomponent)
        components = components.reshape((int(center_opt_flag) + 2*num_orders),3).T.flatten()
        simsopt_dofs = jnp.append(simsopt_dofs, components)
    return simsopt_dofs

cyclone_stellarator_jacobian = jax.jacfwd(generate_simsopt_stellarator_dofs)
def calculate_cyclone_stellarator_jacobian(cyclone_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag='direct', axis_function=None):
    return cyclone_stellarator_jacobian(cyclone_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag=center_opt_type_flag, axis_function=axis_function)

@partial(jax.jit, static_argnames=['rotation_flag', 'normal_flag', 'planar_flag', 'planar_opt_flag', 'nonplanar_opt_flag', 'center_opt_flag', 'center_opt_type_flag', 'surface_function', 'unique_shapes', 'unfixed_orders', 'curve_shapes', 'ncurves'])
def generate_simsopt_windowpane_dofs(cyclone_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag='direct', surface_function=None):
    sin_components_all = [None] * unique_shapes
    cos_components_all = [None] * unique_shapes
    if not planar_flag:
        nonplanar_sin_components_all = [None] * unique_shapes
        nonplanar_cos_components_all = [None] * unique_shapes
    num_orders = len(unfixed_orders)
    if 0 in unfixed_orders:
        num_orders -= 1
    ncenter_dofs, curve_centers = unpack_curve_centers(cyclone_dofs, ncurves, center_opt_flag, center_opt_type_flag)
    nplanar_dofs = 0
    nnonplanar_dofs = 0
    for i in range(unique_shapes):
        if planar_opt_flag:
            nplanar_dofs = 2*num_orders*2*unique_shapes
            sin_components_all[i] = cyclone_dofs[(ncenter_dofs + 2*num_orders*(2*i)):(ncenter_dofs + 2*num_orders*(2*i+1))].reshape(num_orders, 2)
            cos_components_all[i] = cyclone_dofs[(ncenter_dofs + 2*num_orders*(2*i+1)):(ncenter_dofs + 2*num_orders*(2*i+2))].reshape(num_orders, 2)
        else:
            sin_components_all[i] = jnp.array(planar_dofs[(2*num_orders*(2*i)):(2*num_orders*(2*i+1))]).reshape(num_orders, 2)
            cos_components_all[i] = jnp.array(planar_dofs[(2*num_orders*(2*i+1)):(2*num_orders*(2*i+2))]).reshape(num_orders, 2)
        if not planar_flag and nonplanar_opt_flag:
            nnonplanar_dofs = 2*num_orders*unique_shapes
            nonplanar_sin_components_all[i] = cyclone_dofs[(ncenter_dofs + nplanar_dofs + num_orders*(2*i)):(ncenter_dofs + nplanar_dofs + num_orders*(2*i+1))].reshape(num_orders, 1)
            nonplanar_cos_components_all[i] = cyclone_dofs[(ncenter_dofs + nplanar_dofs + num_orders*(2*i+1)):(ncenter_dofs + nplanar_dofs + num_orders*(2*i+2))].reshape(num_orders, 1)
        elif not planar_flag:
            nonplanar_sin_components_all[i] = jnp.array(nonplanar_dofs[(num_orders*(2*i)):(num_orders*(2*i+1))]).reshape(num_orders, 1)
            nonplanar_cos_components_all[i] = jnp.array(nonplanar_dofs[(num_orders*(2*i+1)):(num_orders*(2*i+2))]).reshape(num_orders, 1)
    if rotation_flag:
        nrotation_dofs = ncurves
        rotation_angles = cyclone_dofs[(ncenter_dofs + nplanar_dofs + nnonplanar_dofs):(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs)]
    else:
        assert len(rotation_angles) == ncurves, 'Number of specified rotation angles is not equal to the number of curves.'
        nrotation_dofs = 0
    if normal_flag:
        nnormal_dofs = 2*ncurves
        normal_tors = cyclone_dofs[(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs):(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs + int(nnormal_dofs/2))]
        normal_pols = cyclone_dofs[(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs + int(nnormal_dofs/2)):(ncenter_dofs + nplanar_dofs + nnonplanar_dofs + nrotation_dofs + nnormal_dofs)]
    else:
        assert len(normal_tors) == ncurves, 'Number of specified normal toroidal angles is not equal to the number of curves.'
        assert len(normal_pols) == ncurves, 'Number of specified normal poloidal angles is not equal to the number of curves.'
        nnormal_dofs = 0
    simsopt_dofs = jnp.array([],float)
    for i in range(ncurves):
        shape = curve_shapes[i]
        rotation_angle = rotation_angles[i]
        normal_tor = normal_tors[i]
        normal_pol = normal_pols[i]
        sin_components_this_unrot = sin_components_all[shape]
        cos_components_this_unrot = cos_components_all[shape]
        sin_components_this, cos_components_this = rotate_windowpane_shapes(sin_components_this_unrot, cos_components_this_unrot, rotation_angle, num_orders)
        if not planar_flag:
            nonplanar_sin_components_this = nonplanar_sin_components_all[shape]
            nonplanar_cos_components_this = nonplanar_cos_components_all[shape]
        else:
            nonplanar_sin_components_this = [0] * num_orders
            nonplanar_cos_components_this = [0] * num_orders
        components = jnp.array([],float)
        if center_opt_flag:
            if center_opt_type_flag == 'direct':
                components = jnp.append(components, curve_centers[i])
            #elif center_opt_type_flag == 'on_axis':
                #components = jnp.append(components, axis_function(curve_centers[i]))
            elif center_opt_type_flag == 'on_surface':
                components = jnp.append(components, surface_function(curve_centers[i][0], curve_centers[i][1]))
            else:
                raise ValueError('Should never get here.')
        planar_vectors = planar_vector_list(normal_tor, normal_pol)
        normal_vector = jnp.array([jnp.cos(normal_tor)*jnp.cos(normal_pol), jnp.sin(normal_tor)*jnp.cos(normal_pol), jnp.sin(normal_pol)])
        for ord in range(num_orders):
            scomponent = sin_components_this[ord][0]*planar_vectors[0] + sin_components_this[ord][1]*planar_vectors[1] + nonplanar_sin_components_this[ord]*normal_vector
            ccomponent = cos_components_this[ord][0]*planar_vectors[0] + cos_components_this[ord][1]*planar_vectors[1] + nonplanar_cos_components_this[ord]*normal_vector
            components = jnp.append(components, scomponent)
            components = jnp.append(components, ccomponent)
        components = components.reshape((int(center_opt_flag) + 2*num_orders),3).T.flatten()
        simsopt_dofs = jnp.append(simsopt_dofs, components)
    return simsopt_dofs

cyclone_windowpane_jacobian = jax.jacfwd(generate_simsopt_windowpane_dofs)
def calculate_cyclone_windowpane_jacobian(cyclone_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag='direct', surface_function=None):
    return cyclone_windowpane_jacobian(cyclone_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag=center_opt_type_flag, surface_function=surface_function)

@partial(jax.jit, static_argnames=['center_opt_flag', 'center_opt_type_flag', 'axis_function', 'surface_function', 'ncurves', 'unfixed_orders', 'shape_opt_flag'])
def generate_simsopt_simsopt_dofs(simsopt_dofs, ncurves, center_opt_flag, center_opt_type_flag, dofs, shape_opt_flag, unfixed_orders, axis_function, surface_function):
    if center_opt_flag == False and shape_opt_flag:
        return simsopt_dofs
    _, curve_centers = unpack_curve_centers(simsopt_dofs, ncurves, center_opt_flag, center_opt_type_flag)
    components = jnp.array([],float)
    num_orders = len(unfixed_orders)
    if 0 in unfixed_orders:
        num_orders -= 1
    offset = 0
    for i in range(ncurves):
        if center_opt_flag:
            if center_opt_type_flag == 'direct':
                offset = 3
                components = jnp.append(components, curve_centers[i])
            elif center_opt_type_flag == 'on_axis':
                offset = 1
                components = jnp.append(components, axis_function(curve_centers[i]))
            elif center_opt_type_flag == 'on_surface':
                offset = 2
                components = jnp.append(components, surface_function(*curve_centers[i]))
            else:
                raise ValueError('Should never get here.')
        if shape_opt_flag:
            components = jnp.append(components, simsopt_dofs[((offset+6*num_orders)*i+offset):((offset+6*num_orders)*(i+1))])
        else:
            components = jnp.append(components, jnp.array(dofs[((6*num_orders)*i):((6*num_orders)*(i+1))]))
    return components

simsopt_jacobian = jax.jacfwd(generate_simsopt_simsopt_dofs)
def calculate_simsopt_jacobian(simsopt_dofs, ncurves, center_opt_flag, center_opt_type_flag, dofs, shape_opt_flag, unfixed_orders, axis_function, surface_function):
    return simsopt_jacobian(simsopt_dofs, ncurves, center_opt_flag, center_opt_type_flag, dofs, shape_opt_flag, unfixed_orders, axis_function, surface_function)

def sanity_check_number_of_dofs(dictionary, num_dofs):
    coil_type = dictionary['type']
    num_orders = len(dictionary['unfixed_orders'])
    num_coils = len(dictionary['curves'])
    if 0 in dictionary['unfixed_orders']:
        num_orders -= 1
    if dictionary['optimizables']['center_opt_flag'][0]:
        if dictionary['optimizables']['center_opt_type_flag'][0] == 'direct':
            num_dofs_calculated = 3 * num_coils
        elif dictionary['optimizables']['center_opt_type_flag'][0] == 'on_axis':
            num_dofs_calculated = num_coils
        elif dictionary['optimizables']['center_opt_type_flag'][0] == 'on_surface':
            num_dofs_calculated = 2 * num_coils
        else:
            raise ValueError("Should never get here.")
    else:
        num_dofs_calculated = 0
    if coil_type in ['simsopt_stellarator', 'simsopt_windowpane']:
        if coil_type == 'simsopt_stellarator':
            ncurves = dictionary['ncurves']
        else:
            ncurves = sum(dictionary['npoloidalcurves'])
        if dictionary['optimizables']['shape_opt_flag'][0]:
            num_dofs_calculated = num_dofs_calculated + 2*3*num_orders*ncurves
    elif coil_type in ['cyclone_stellarator', 'cyclone_windowpane']:
        unique_shapes = dictionary['unique_shapes']
        if dictionary['optimizables']['planar_opt_flag'][0]:
            num_dofs_calculated = num_dofs_calculated + 2*2*num_orders*unique_shapes
        if dictionary['optimizables']['nonplanar_opt_flag'][0]:
            num_dofs_calculated = num_dofs_calculated + 2*num_orders*unique_shapes
        if dictionary['optimizables']['rotation_opt_flag'][0]:
            num_dofs_calculated = num_dofs_calculated + num_coils
        if dictionary['optimizables']['normal_opt_flag'][0]:
            num_dofs_calculated = num_dofs_calculated + 2*num_coils
    elif coil_type in ['cws_stellarator', 'cws_windowpane']:
        raise NotImplemented('CWS curves not yet implemented.')
    else:
        raise ValueError('Not a valid option for coil type.')
    assert num_dofs_calculated == num_dofs, 'Number of dofs does not pass the sanity check compared to number of expected dofs.'

def generate_dofs_from_dictionary(full_dictionary):
    #all currents, then dofs by coil set
    ##centers, planars, nonplanars (both previous by unique shape), rotations, normals
    full_dofs = [current.current_to_scale.current for current in full_dictionary['all_currents']]
    for coil_set in full_dictionary['coil_sets']:
        #find dofs for individual coil sets
        this_dict = full_dictionary[coil_set]
        this_dofs = []
        if this_dict['optimizables']['center_opt_flag'][0]:
            if this_dict['optimizables']['center_opt_type_flag'][0] == 'direct':
                for center in this_dict['centers']:
                    this_dofs.extend(center)
            elif this_dict['optimizables']['center_opt_type_flag'][0] == 'on_axis':
                this_dofs.extend(this_dict['center_tors'])
            elif this_dict['optimizables']['center_opt_type_flag'][0] == 'on_surface':
                this_dofs.extend(this_dict['center_tors'])
                this_dofs.extend(this_dict['center_pols'])
        if this_dict['type'] in ['simsopt_stellarator', 'simsopt_windowpane']:
            if this_dict['optimizables']['shape_opt_flag'][0]:
                this_dofs.extend(this_dict['dofs'])
        elif this_dict['type'] in ['cyclone_stellarator', 'cyclone_windowpane']:
            if this_dict['optimizables']['planar_opt_flag'][0]:
                this_dofs.extend(this_dict['planar_dofs'])
            if this_dict['optimizables']['nonplanar_opt_flag'][0]:
                this_dofs.extend(this_dict['nonplanar_dofs'])
            if this_dict['optimizables']['rotation_opt_flag'][0]:
                this_dofs.extend(this_dict['rotation_angles'])
            if this_dict['optimizables']['normal_opt_flag'][0]:
                this_dofs.extend(this_dict['normal_tors'])
                this_dofs.extend(this_dict['normal_pols'])
        elif this_dict['type'] in ['cws_stellarator', 'cws_windowpane']:
            raise NotImplemented('Not yet implemented')
        else:
            raise ValueError('Not a valid coil type.')
        num_dofs = len(this_dofs)
        sanity_check_number_of_dofs(this_dict, num_dofs)
        this_dict['num_dofs'] = num_dofs
        this_dict['opt_dofs'] = this_dofs
        full_dofs.extend(this_dofs)
    full_dictionary['opt_dofs'] = full_dofs
    return full_dofs

def get_coil_set_parameters(dofs_full, full_dictionary, label, current_tally):
    coil_dictionary = full_dictionary[label]
    num_coil_set_dofs = coil_dictionary['num_dofs']
    coil_set_dofs = dofs_full[current_tally:(current_tally + num_coil_set_dofs)]
    new_tally = current_tally + num_coil_set_dofs
    coil_type = coil_dictionary['type']
    curves = coil_dictionary['curves']
    ncurves = len(curves)
    unfixed_orders = tuple(coil_dictionary['unfixed_orders'])
    center_opt_flag = coil_dictionary['optimizables']['center_opt_flag'][0]
    center_opt_type_flag = coil_dictionary['optimizables']['center_opt_type_flag'][0]
    if center_opt_flag:
        if center_opt_type_flag == 'on_axis':
            axis_function = coil_dictionary['axis_function']
            surface_function = None
        elif center_opt_type_flag == 'on_surface':
            surface_function = coil_dictionary['surface_function']
            axis_function = None
        else:
            axis_function = None
            surface_function = None
    else:
        axis_function = None
        surface_function = None
    if 'cyclone' in coil_type:
        unique_shapes = coil_dictionary['unique_shapes']
        curve_shapes = tuple(coil_dictionary['curve_shapes'])
        normal_flag = coil_dictionary['optimizables']['normal_opt_flag'][0]
        planar_flag = coil_dictionary['planar_flag']
        planar_opt_flag = coil_dictionary['optimizables']['planar_opt_flag'][0]
        nonplanar_opt_flag = coil_dictionary['optimizables']['nonplanar_opt_flag'][0]
        shape_opt_flag = None
        dofs = None
        rotation_flag = coil_dictionary['optimizables']['rotation_opt_flag'][0]
        if not planar_opt_flag:
            planar_dofs = coil_dictionary['planar_dofs']
        else:
            planar_dofs = None
        if not nonplanar_opt_flag:
            nonplanar_dofs = coil_dictionary['nonplanar_dofs']
        else:
            nonplanar_dofs = None
        if not rotation_flag:
            rotation_angles = coil_dictionary['rotation_angles']
        else:
            rotation_angles = None
        if not normal_flag:
            normal_tors = coil_dictionary['normal_tors']
            normal_pols = coil_dictionary['normal_pols']
        else:
            normal_tors = None
            normal_pols = None
    else:
        shape_opt_flag = coil_dictionary['optimizables']['shape_opt_flag'][0]
        if not shape_opt_flag:
            dofs = coil_dictionary['dofs']
        else:
            dofs = None
        unique_shapes = None
        curve_shapes = None
        normal_flag = None
        planar_flag = None
        planar_dofs = None
        planar_opt_flag = None
        nonplanar_dofs = None
        nonplanar_opt_flag = None
        rotation_flag = None
        rotation_angles = None
        normal_tors = None
        normal_pols = None
    return coil_set_dofs, coil_type, curves, ncurves, unfixed_orders, center_opt_flag, center_opt_type_flag, axis_function, surface_function, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, dofs, shape_opt_flag, unique_shapes, curve_shapes, normal_flag, normal_tors, normal_pols, planar_flag, rotation_flag, rotation_angles, new_tally

def cyclone_optimization_function_scipy(dofs_full, *args):
    full_dictionary = args[0]
    objective = args[1]
    num_current_dofs = full_dictionary['num_current_dofs']
    simsopt_dofs_full = dofs_full[:num_current_dofs]
    tally = num_current_dofs
    num_orders_per_coil_set = []
    coil_sets = full_dictionary['coil_sets']
    for label in coil_sets:
        coil_set_dofs, coil_type, curves, ncurves, unfixed_orders, center_opt_flag, \
            center_opt_type_flag, axis_function, surface_function, planar_dofs, planar_opt_flag, \
            nonplanar_dofs, nonplanar_opt_flag, dofs, shape_opt_flag, unique_shapes, \
            curve_shapes, normal_flag, normal_tors, normal_pols, planar_flag, \
            rotation_flag, rotation_angles, tally = get_coil_set_parameters(dofs_full, full_dictionary, label, tally)
        num_orders_per_coil_set.append(len(unfixed_orders))
        if 0 in unfixed_orders:
            num_orders_per_coil_set[-1] -= 0.5 # For the purposes of counting dofs, 0th order dofs only count for half (3 dofs instead of 6)
        if coil_type == 'simsopt_stellarator':
            simsopt_dofs = generate_simsopt_simsopt_dofs(coil_set_dofs, ncurves, center_opt_flag, center_opt_type_flag, dofs, shape_opt_flag, unfixed_orders, axis_function, surface_function)
        elif coil_type == 'cyclone_stellarator':
            simsopt_dofs = generate_simsopt_stellarator_dofs(coil_set_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag=center_opt_type_flag, axis_function=axis_function)
        elif coil_type == 'simsopt_windowpane':
            simsopt_dofs = generate_simsopt_simsopt_dofs(coil_set_dofs, ncurves, center_opt_flag, center_opt_type_flag, dofs, shape_opt_flag, unfixed_orders, axis_function, surface_function)
        elif coil_type == 'cyclone_windowpane':
            simsopt_dofs = generate_simsopt_windowpane_dofs(coil_set_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag=center_opt_type_flag, surface_function=surface_function)
        simsopt_dofs_full = np.append(simsopt_dofs_full, simsopt_dofs)
        # coil_dictionary['currents'] = [] # add new currents to dictionary
        # change all the dictionary entries that aren't tracked by the dictionary (i.e. they are in the dofs)
    # reorder simsopt_dofs_full to account for the fact that Cyclone sorts coils numerically by labal and simsopt does so by string ordering
    numerically_ordered_simsopt_dofs_full = list(simsopt_dofs_full)
    first_coil_label_number = full_dictionary['first_coil_label_number']
    string_sorted_mapping = np.array(sorted([f"{i+first_coil_label_number}" for i in range(num_current_dofs)]))
    string_ordered_simsopt_dofs_full = []
    for i in range(num_current_dofs):
        string_ordered_simsopt_dofs_full.append(numerically_ordered_simsopt_dofs_full[int(string_sorted_mapping[i])-first_coil_label_number])
    num_curves_per_coil_set = full_dictionary['num_currents']
    num_dofs_per_curve_per_coil_set = [6 * num_orders for num_orders in num_orders_per_coil_set]
    num_dofs_per_curve = np.repeat(num_dofs_per_curve_per_coil_set, num_curves_per_coil_set)
    num_dofs_per_curve = num_dofs_per_curve.astype(int)
    assert len(num_dofs_per_curve) == num_current_dofs
    for i in range(num_current_dofs):
        num_curve_dofs_before_this_curve = sum(num_dofs_per_curve[:int(string_sorted_mapping[i])-first_coil_label_number])
        string_ordered_simsopt_dofs_full.extend(numerically_ordered_simsopt_dofs_full[(num_current_dofs+num_curve_dofs_before_this_curve):(num_current_dofs+num_curve_dofs_before_this_curve+num_dofs_per_curve[int(string_sorted_mapping[i])-first_coil_label_number])])
    objective.x = string_ordered_simsopt_dofs_full
    objective_eval = objective.J()
    string_ordered_objective_simsopt_grad = objective.dJ()
    objective_simsopt_grad = []
    string_sorted_num_dofs_per_curve = [num_dofs_per_curve[int(string_sorted_mapping[i])-first_coil_label_number] for i in range(num_current_dofs)]
    # need to send simsopt string sorted order to Cyclone numerically sorted order
    for i in range(num_current_dofs):
        objective_simsopt_grad.extend([string_ordered_objective_simsopt_grad[int(np.argwhere(string_sorted_mapping == f"{i+first_coil_label_number}")[0][0])]])
    for i in range(num_current_dofs):
        num_curve_dofs_before_this_curve = sum(string_sorted_num_dofs_per_curve[:np.argwhere(string_sorted_mapping == f"{i+first_coil_label_number}")[0][0]])
        objective_simsopt_grad.extend(string_ordered_objective_simsopt_grad[(num_current_dofs+num_curve_dofs_before_this_curve):(num_current_dofs+num_curve_dofs_before_this_curve+string_sorted_num_dofs_per_curve[np.argwhere(string_sorted_mapping == f"{i+first_coil_label_number}")[0][0]])])
    cyclone_tally = num_current_dofs
    simsopt_tally = num_current_dofs
    objective_grad = objective_simsopt_grad[:num_current_dofs]
    objective_simsopt_grad = np.array(objective_simsopt_grad)
    for label in coil_sets:
        coil_set_dofs, coil_type, curves, ncurves, unfixed_orders, center_opt_flag, \
            center_opt_type_flag, axis_function, surface_function, planar_dofs, planar_opt_flag, \
            nonplanar_dofs, nonplanar_opt_flag, dofs, shape_opt_flag, unique_shapes, \
            curve_shapes, normal_flag, normal_tors, normal_pols, planar_flag, \
            rotation_flag, rotation_angles, cyclone_tally = get_coil_set_parameters(dofs_full, full_dictionary, label, cyclone_tally)
        if coil_type == 'simsopt_stellarator':
            this_jacobian = calculate_simsopt_jacobian(coil_set_dofs, ncurves, center_opt_flag, center_opt_type_flag, dofs, shape_opt_flag, unfixed_orders, axis_function, surface_function)
        elif coil_type == 'cyclone_stellarator':
            this_jacobian = calculate_cyclone_stellarator_jacobian(coil_set_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag=center_opt_type_flag, axis_function=axis_function)
        elif coil_type == 'simsopt_windowpane':
            this_jacobian = calculate_simsopt_jacobian(coil_set_dofs, ncurves, center_opt_flag, center_opt_type_flag, dofs, shape_opt_flag, unfixed_orders, axis_function, surface_function)
        elif coil_type == 'cyclone_windowpane':
            this_jacobian = calculate_cyclone_windowpane_jacobian(coil_set_dofs, ncurves, unique_shapes, curve_shapes, rotation_angles, rotation_flag, normal_tors, normal_pols, normal_flag, unfixed_orders, planar_flag, planar_dofs, planar_opt_flag, nonplanar_dofs, nonplanar_opt_flag, center_opt_flag, center_opt_type_flag=center_opt_type_flag, surface_function=surface_function)
        this_num_simsopt_dofs = len(this_jacobian)
        this_simsopt_grad = objective_simsopt_grad[simsopt_tally:(simsopt_tally + this_num_simsopt_dofs)]
        simsopt_tally += this_num_simsopt_dofs
        this_objective_grad = jnp.matmul(this_simsopt_grad.reshape((1,-1)), this_jacobian)
        if this_objective_grad.tolist()[0] == []:
            continue
        objective_grad.extend(this_objective_grad.tolist()[0])
    print(objective_eval)
    return objective_eval, objective_grad

def update_dictionary_after_optimization(dofs_post, full_dictionary):
    full_dictionary['opt_dofs'] = dofs_post
    # somehow unpack dofs to dictionary format (backwards of generate_dofs_from_dictionary)
    # First I need to count up currents
    current_scales = []
    num_currents_per_coil_set = full_dictionary['num_currents']
    # Update all_current_values
    for i, current_object in enumerate(full_dictionary['all_currents']):
        current_scales.append(current_object.scale)
        full_dictionary['all_current_values'][i] = current_scales[i] * dofs_post[i]
    tally = 0
    for i, coil_set in enumerate(full_dictionary['coil_sets']):
        this_dict = full_dictionary[coil_set]
        # Update coil set 'current_list'
        for j in range(num_currents_per_coil_set[i]):
            this_dict['currents']['current_list'][j] = current_scales[tally+j] * dofs_post[tally+j]
        # Update coil set 'currents'
        for j, current_label in enumerate(this_dict['currents']):
            if current_label == 'current_list':
                continue
            this_dict['currents'][current_label] = this_dict['currents']['current_list'][j]
        tally += num_currents_per_coil_set[i]
    # Next up is to count out how many dofs I need
    # That's the whole point of opt_dofs
    coil_sets = full_dictionary['coil_sets']
    for label in coil_sets:
        this_dict = full_dictionary[label]
        this_dofs = []
        coil_set_dofs, coil_type, _, ncurves, unfixed_orders, center_opt_flag, \
            center_opt_type_flag, axis_function, surface_function, _, planar_opt_flag, \
            _, nonplanar_opt_flag, _, shape_opt_flag, unique_shapes, \
            _, normal_flag, _, _, planar_flag, \
            rotation_flag, _, tally = get_coil_set_parameters(dofs_post, full_dictionary, label, tally)
        ncenter_dofs, curve_centers = unpack_curve_centers(coil_set_dofs, ncurves, center_opt_flag, center_opt_type_flag)
        num_orders = len(unfixed_orders)
        if 0 in unfixed_orders:
            num_orders = num_orders - 1
        # Update centers
        if ncenter_dofs == 3*ncurves:
            for i in range(ncurves):
                this_dict['centers'][i] = curve_centers[i]
            if 'center_tors' in this_dict:
                this_dict['center_tors'] = 'Untracked after optimizing centers directly.'
            if 'center_pols' in this_dict:
                this_dict['center_pols'] = 'Untracked after optimizing centers directly.'
            this_dofs.extend(this_dict['centers'])
        elif ncenter_dofs == ncurves:
            for i in range(ncurves):
                this_dict['center_tors'][i] = curve_centers[i][0]
                this_dict['centers'][i] = axis_function(curve_centers[i][0])
            this_dofs.extend(this_dict['center_tors'])
        elif ncenter_dofs == 2*ncurves:
            for i in range(ncurves):
                this_dict['center_tors'][i] = curve_centers[i][0]
                this_dict['center_pols'][i] = curve_centers[i][1]
                this_dict['centers'][i] = surface_function(curve_centers[i][0], curve_centers[i][1])
            this_dofs.extend(this_dict['center_tors'])
            this_dofs.extend(this_dict['center_pols'])
        # after centers is...
        # simsopt shape - 2*3*order*num_coils       OR
        # cyclone shape - planar, nonplanar, rotation, normal
        if coil_type in ['simsopt_stellarator', 'simsopt_windowpane']:
            if shape_opt_flag:
                simsopt_dofs = coil_set_dofs[ncenter_dofs:]
                full_simsopt_dofs = this_dict['full_dofs']
                full_simsopt_dofs = update_all_simsopt_dofs_from_unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders, full_simsopt_dofs)
                this_dict['dofs'] = simsopt_dofs
                this_dict['full_dofs'] = full_simsopt_dofs
                this_dofs.extend(this_dict['dofs'])
        elif coil_type in ['cyclone_stellarator', 'cyclone_windowpane']:
            if planar_opt_flag:
                nplanar_dofs = 2*num_orders*2*unique_shapes
                planar_dofs = coil_set_dofs[ncenter_dofs:ncenter_dofs+nplanar_dofs]
                full_planar_dofs = this_dict['full_planar_dofs']
                full_planar_dofs = update_all_planar_dofs_from_unfixed_planar_dofs(planar_dofs, unfixed_orders, full_planar_dofs, unique_shapes)
                this_dict['planar_dofs'] = planar_dofs
                this_dict['full_planar_dofs'] = full_planar_dofs
                this_dofs.extend(this_dict['planar_dofs'])
            else:
                nplanar_dofs = 0
            if not planar_flag and nonplanar_opt_flag:
                nnonplanar_dofs = 2*num_orders*unique_shapes
                nonplanar_dofs = coil_set_dofs[(ncenter_dofs+nplanar_dofs):(ncenter_dofs+nplanar_dofs+nnonplanar_dofs)]
                full_nonplanar_dofs = this_dict['full_nonplanar_dofs']
                full_nonplanar_dofs = update_all_nonplanar_dofs_from_unfixed_nonplanar_dofs(nonplanar_dofs, unfixed_orders, full_nonplanar_dofs, unique_shapes)
                this_dict['nonplanar_dofs'] = nonplanar_dofs
                this_dict['full_nonplanar_dofs'] = full_nonplanar_dofs
                this_dofs.extend(this_dict['nonplanar_dofs'])
            else:
                nnonplanar_dofs = 0
            if rotation_flag:
                nrotation_dofs = ncurves
                rotation_dofs = coil_set_dofs[(ncenter_dofs+nplanar_dofs+nnonplanar_dofs):(ncenter_dofs+nplanar_dofs+nnonplanar_dofs+nrotation_dofs)]
                this_dict['rotation_angles'] = rotation_dofs
                this_dofs.extend(this_dict['rotation_angles'])
            else:
                nrotation_dofs = 0
            if normal_flag:
                nnormal_dofs = 2*ncurves
                normal_dofs = coil_set_dofs[(ncenter_dofs+nplanar_dofs+nnonplanar_dofs+nrotation_dofs):(ncenter_dofs+nplanar_dofs+nnonplanar_dofs+nrotation_dofs+nnormal_dofs)]
                this_dict['normal_tors'] = normal_dofs[:int(nnormal_dofs/2)]
                this_dict['normal_pols'] = normal_dofs[int(nnormal_dofs/2):]
                this_dofs.extend(this_dict['normal_tors'])
                this_dofs.extend(this_dict['normal_pols'])
        this_dict['opt_dofs'] = this_dofs
    return full_dictionary

def run_minimize(dofs_full, full_dictionary, objective):
    optimization_dict = full_dictionary['optimization_parameters']
    if 'library' in optimization_dict:
        library = optimization_dict['library'][0]
    else:
        library = 'scipy'
    if 'jac' in optimization_dict or 'jacobian' in optimization_dict:
        if 'jac' in optimization_dict:
            jac = optimization_dict['jac'][0]
        else:
            jac = optimization_dict['jacobian'][0]
    else:
        jac = True
    if 'method' in optimization_dict:
        method = optimization_dict['method'][0]
    else:
        method = None
    if 'tol' in optimization_dict or 'tolerance' in optimization_dict:
        if 'tol' in optimization_dict:
            tol = optimization_dict['tol'][0]
        else:
            tol = optimization_dict['tolerance'][0]
    else:
        tol = 1e-4
    if 'maxiter' in optimization_dict or 'MAXITER' in optimization_dict:
        if 'maxiter' in optimization_dict:
            maxiter = optimization_dict['maxiter'][0]
        else:
            maxiter = optimization_dict['MAXITER'][0]
    else:
        maxiter = None
    if 'options' in optimization_dict:
        options = optimization_dict['options'][0]
    else:
        options = {}
    if maxiter is not None:
        options.update({'maxiter': maxiter})
    if library == 'scipy':
        from scipy.optimize import minimize
        if method is None:
            method = 'L-BFGS-B'
        if options is None:
            options = {}
        res = minimize(cyclone_optimization_function_scipy, dofs_full, args=(full_dictionary, objective), jac=jac, method=method, tol=tol, options=options)
    else:
        raise NotImplemented('Only scipy minimize is supported at this time')
    full_dictionary = update_dictionary_after_optimization(res.x, full_dictionary)
    return res.x

def determine_number_of_optimizations(full_dictionary):
    max_opts = 1
    for coil_set in full_dictionary['coil_sets']:
        this_dict = full_dictionary[coil_set]
        for optimizable in this_dict['optimizables']:
            if len(this_dict['optimizables'][optimizable]) > max_opts:
                max_opts = len(this_dict['optimizables'][optimizable])
    #something to look at the stuff in the full_dictionary['optimization'] entries as well
    #full_dictionary[''] = max_opts
    return max_opts

def cycle_dictionary_optimizables(full_dictionary):
    for coil_set in full_dictionary['coil_sets']:
        this_dict = full_dictionary[coil_set]
        for optimizable in this_dict['optimizables']:
            if len(this_dict['optimizables'][optimizable]) > 1:
                this_dict['optimizables'][optimizable] = this_dict['optimizables'][optimizable][1:]
    #something to look at the stuff in the full_dictionary['optimization'] entries as well

'''  #### This function still needs work before it is usable, along with creating functions to allow it to work
def run_minimize_multiple_parameters(dofs_full, full_dictionary, objective, minimize_dictionary):
    num_optimizations = determine_number_of_optimizations(full_dictionary)
    for iteration in range(num_optimizations):
        #probably not how I'm going to do this going forward
        library, jac, method, tol, options = unpack_minimize_dictionary(minimize_dictionary, iteration)
        objective = unpack_objective(objective, iteration)
        dofs_full = run_minimize(dofs_full, full_dictionary, objective, library=library, jac=jac, method=method, tol=tol, options=options)
        cycle_dictionary_optimizables(full_dictionary, iteration)
        dofs_full = recompute_dofs(dofs_full, full_dictionary)
        # I don't think this one is done yet.
'''