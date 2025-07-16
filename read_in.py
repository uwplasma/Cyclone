import numpy as np
import tomli
from Cyclone.coil_initialization import create_shapes_matrix, init_simsopt_stellarator_coils, init_simsopt_windowpane_coils, init_stellarator_coils, init_windowpane_coils
from simsopt.field import coils_via_symmetries, Current, BiotSavart
from simsopt.geo import SurfaceRZFourier

def import_magnetic_surface(surface_representation):
    magnetic_surface = SurfaceRZFourier.from_vmec_input(surface_representation, range="half period")
    return magnetic_surface

def clean_current_dictionary(dictionary, ncurves, default_current, shapes_matrix = None):
    if shapes_matrix is None:
        shapes_matrix = create_shapes_matrix('tile', 1, ncurves)
    current_list = []
    if type(ncurves) == int:
        for i in range(ncurves):
            if shapes_matrix[i] == -1:
                continue
            if 'current_{}'.format(i) in dictionary:
                current_list.append(dictionary['current_{}'.format(i)])
            else:
                dictionary['current_{}'.format(i)] = default_current
                current_list.append(dictionary['current_{}'.format(i)])
    elif type(ncurves) == tuple:
        if type(ncurves[1]) == tuple:
            ncurves = ncurves[1]
        else:
            ncurves = ncurves[1:]
        for i, npoloidalcurves in enumerate(ncurves):
            for j in range(npoloidalcurves):
                if shapes_matrix[i][j] == -1:
                    continue
                if 'current_{}_{}'.format(i,j) in dictionary:
                    current_list.append(dictionary['current_{}_{}'.format(i,j)])
                else:
                    dictionary['current_{}_{}'.format(i,j)] = default_current
                    current_list.append(dictionary['current_{}_{}'.format(i,j)])
    dictionary['current_list'] = current_list
    dictionary['num_currents'] = len(current_list)
    return dictionary

def fixed_to_list(fixed, order):
    if (type(fixed) == str and fixed == 'all') or \
    (type(fixed) == int and fixed>=order) or \
    ((type(fixed) == list or type(fixed) == np.ndarray) and \
            sum([order_n in fixed for order_n in list(range(order+1))])/(order+1) == 1):
        fixed_list = list(range(order+1))
    elif type(fixed) == int:
        fixed_list = list(range(fixed+1))
    elif type(fixed) == list or type(fixed) == np.ndarray or type(fixed) == tuple:
        fixed_list = sorted([ord for ind, ord in enumerate(fixed) if ((ord <= order) and not ord in fixed[ind+1:])])
    elif fixed is None:
        fixed_list = [None]
    else:
        raise TypeError('The input you entered for the \'fixed\' variable is not supported.')
    return fixed_list

def clean_simsopt_dofs_dictionary(dictionary, order):
    for ind_dofs in dictionary:
        this_dofs = dictionary[ind_dofs]
        list_dofs = []
        for i in range(3):
            if len(this_dofs[i]) > 2*order:
                raise ValueError('dofs for shape {} are overspecified in {} component.'.format(ind_dofs, i))
            elif len(this_dofs[i]) < 2*order:
                this_dofs[i].extend(np.zeros(2*order - len(this_dofs[i])))
                list_dofs.extend(this_dofs[i])
            else:
                list_dofs.extend(this_dofs[i])
        dictionary[ind_dofs] = this_dofs
    return dictionary

def send_to_list(item):
    if type(item) != list:
        if isinstance(item, (tuple, np.ndarray)):
            item = list(item)
        else:
            item = [item]
    return item

def clean_optimizables_dictionary(dictionary, coil_set_type, fixed):
    new_dictionary = {}
    new_dictionary['center_opt_flag'] = send_to_list(dictionary.pop('center_opt_flag', [False]))
    if fixed == [None] and any(new_dictionary['center_opt_flag']) == False:
        print('Unspecified \'fixed\' variable becomes [0] due to \'center_opt_flag\' being False.')
        fixed = [0]
    elif fixed == [None]:
        print('Unspecified \'fixed\' variable becomes [] due to \'center_opt_flag\' being True.')
        fixed = []
    if 0 in fixed and any(new_dictionary['center_opt_flag']) == True:
        print('\'fixed\' variable containing 0 overrides \'center_opt_flag\' variable to False.')
        new_dictionary['center_opt_flag'] = [False] * len(new_dictionary['center_opt_flag'])
    elif not 0 in fixed and any(new_dictionary['center_opt_flag']) == False:
        print('\'fixed\' variable not containing 0 overrides \'center_opt_flag\' variable to True.')
        new_dictionary['center_opt_flag'] = [True] * len(new_dictionary['center_opt_flag'])
    new_dictionary['center_opt_type_flag'] = send_to_list(dictionary.pop('center_opt_type_flag', ['direct']))
    if coil_set_type in ['simsopt_stellarator', 'simsopt_windowpane']:
        new_dictionary['shape_opt_flag'] = send_to_list(dictionary.pop('shape_opt_flag', [True]))
    if coil_set_type in ['cyclone_stellarator', 'cyclone_windowpane']:
        new_dictionary['planar_opt_flag'] = send_to_list(dictionary.pop('planar_opt_flag', [True]))
        new_dictionary['nonplanar_opt_flag'] = send_to_list(dictionary.pop('nonplanar_opt_flag', [False]))
        planar_flag = dictionary.pop('planar_flag')
        if planar_flag:
            new_dictionary['nonplanar_opt_flag'][0] = False
        new_dictionary['rotation_opt_flag'] = send_to_list(dictionary.pop('rotation_opt_flag', [False]))
        new_dictionary['normal_opt_flag'] = send_to_list(dictionary.pop('normal_opt_flag', [False]))
    return new_dictionary, fixed

def clean_optimization_dictionary(dictionary):
    new_dictionary = {}
    new_dictionary['library'] = send_to_list(dictionary.pop('library', ['scipy']))
    if 'jacobian' in dictionary:
        new_dictionary['jac'] = send_to_list(dictionary['jacobian'])
    else:
        new_dictionary['jac'] = send_to_list(dictionary.pop('jac', [True]))
    new_dictionary['method'] = send_to_list(dictionary.pop('method', [None]))
    new_dictionary['tolerance'] = send_to_list(dictionary.pop('tolerance', [1e-4]))
    if 'MAXITER' in dictionary:
        new_dictionary['maxiter'] = send_to_list(dictionary['MAXITER'])
    else:
        new_dictionary['maxiter'] = send_to_list(dictionary.pop('maxiter', [None]))
    new_dictionary['options'] = send_to_list(dictionary.pop('options', [{}]))
    return new_dictionary

def clean_objective_dictionary(dictionary):
    return None

def import_oneoverR_field(dictionary):
    from simsopt.field import ToroidalField
    new_dictionary = {}
    new_dictionary['R0'] = dictionary.pop('R0', 1.)
    new_dictionary['B0'] = dictionary.pop('B0', 1.)
    new_dictionary['field'] = ToroidalField(new_dictionary['R0'], new_dictionary['B0'])
    return new_dictionary

def import_simsopt_stellarator_coils(dictionary):
    assert dictionary['type'] == 'simsopt_stellarator', 'Type of coils should be simsopt_stellarator, but another type was passed.'
    default_current = dictionary.pop('default_current', 1e5)
    axis_representation = dictionary.pop('axis_representation', 'default')
    ncurves = dictionary['ncurves']
    R1 = dictionary.pop('R1', 0.5)
    order = dictionary.pop('order', 1)
    numquadpoints = dictionary.pop('numquadpoints', None)
    if numquadpoints == 'None':
        numquadpoints = None
    fixed = dictionary.pop('fixed', None)
    fixed = fixed_to_list(fixed, order)
    dofs_dictionary = dictionary.pop('dofs', {})
    dofs_dictionary = clean_simsopt_dofs_dictionary(dofs_dictionary, order)
    optimizables_dictionary = dictionary.pop('optimizables', {})
    optimizables_dictionary, fixed = clean_optimizables_dictionary(optimizables_dictionary, 'simsopt_stellarator', fixed)
    curves, ncurves, unfixed_orders, centers, center_tors, full_dofs, simsopt_dofs, axis_function = init_simsopt_stellarator_coils(axis_representation, ncurves, R1 = R1, order = order, numquadpoints = numquadpoints, fixed=fixed, dofs = dofs_dictionary)
    current_dictionary = dictionary.pop('currents', {})
    current_dictionary = clean_current_dictionary(current_dictionary, ncurves, default_current)
    num_currents = current_dictionary.pop('num_currents')
    new_dictionary = {
        'type' : 'simsopt_stellarator',
        'num_currents' : num_currents,
        'axis_function' : axis_function,
        'ncurves' : ncurves,
        'order' : order,
        'fixed' : fixed,
        'unfixed_orders' : unfixed_orders,
        'curves' : curves,
        'full_dofs' : full_dofs,
        'dofs' : simsopt_dofs,
        'centers' : centers,
        'center_tors' : center_tors,
        'currents' : current_dictionary,
        'optimizables' : optimizables_dictionary,
    }
    return new_dictionary

def import_cyclone_stellarator_coils(dictionary):
    assert dictionary['type'] == 'cyclone_stellarator', 'Type of coils should be cyclone_stellarator, but another type was passed.'
    default_current = dictionary.pop('default_current', 1e5)
    planar_flag = dictionary['planar_flag'] # I could have this pop with a default if I want too?
    axis_representation = dictionary.pop('axis_representation', None)
    ncurves = dictionary['ncurves']
    unique_shapes = dictionary.pop('unique_shapes', None)
    normal_tors = dictionary.pop('normal_tors', None)
    normal_pols = dictionary.pop('normal_pols', None)
    rotation_vector = dictionary.pop('rotation_vector', None)
    tile_as = dictionary.pop('tile_as', 'tile')
    R1 = dictionary.pop('R1', 0.5)
    order = dictionary.pop('order', 1)
    numquadpoints = dictionary.pop('numquadpoints', None)
    if numquadpoints == 'None':
        numquadpoints = None
    fixed = dictionary.pop('fixed', None)
    fixed = fixed_to_list(fixed, order)
    sin_cos_components_dictionary = dictionary.pop('sin_cos_components', {})
    optimizables_dictionary = dictionary.pop('optimizables', {})
    optimizables_dictionary['planar_flag'] = planar_flag
    optimizables_dictionary, fixed = clean_optimizables_dictionary(optimizables_dictionary, 'cyclone_stellarator', fixed)
    curves, ncurves, unfixed_orders, planar_dofs, full_planar_dofs, nonplanar_dofs, full_nonplanar_dofs, normal_tors, normal_pols, rotation_angles, centers, center_tors, unique_shapes, shapes_vector, curve_shapes, axis_function = init_stellarator_coils(axis_representation, ncurves, unique_shapes=unique_shapes, tor_angles=normal_tors, pol_angles=normal_pols, rotation_vector=rotation_vector, tile_as=tile_as, R1 = R1, order=order, numquadpoints=numquadpoints, fixed=fixed, sin_cos_components = sin_cos_components_dictionary)
    if (unique_shapes / ncurves) > (1 - 0.5 / order):
        print('Given the number of unique shapes, the number of curves, and the order of the curves, less dofs will be created using simsopt curves than cyclone curves. You may want to consider augmenting your config file to use simsopt coils instead depending on your use case.')
    current_dictionary = dictionary.pop('currents', {})
    current_dictionary = clean_current_dictionary(current_dictionary, ncurves, default_current, shapes_matrix = shapes_vector)
    num_currents = current_dictionary.pop('num_currents')
    new_dictionary = {
        'type' : 'cyclone_stellarator',
        'num_currents' : num_currents,
        'planar_flag' : planar_flag,
        'axis_function' : axis_function,
        'ncurves' : ncurves,
        'order' : order,
        'fixed' : fixed,
        'unfixed_orders' : unfixed_orders,
        'unique_shapes' : unique_shapes,
        'shapes_vector' : shapes_vector,
        'curves' : curves,
        'curve_shapes' : curve_shapes,
        'full_planar_dofs' : full_planar_dofs,
        'planar_dofs' : planar_dofs,
        'full_nonplanar_dofs' : full_nonplanar_dofs,
        'nonplanar_dofs' : nonplanar_dofs,
        'centers' : centers,
        'center_tors' : center_tors,
        'normal_tors' : normal_tors,
        'normal_pols' : normal_pols,
        'rotation_angles' : rotation_angles,
        'currents' : current_dictionary,
        'optimizables' : optimizables_dictionary
    }
    return new_dictionary

def import_simsopt_windowpane_coils(dictionary):
    assert dictionary['type'] == 'simsopt_windowpane', 'Type of coils should be simsopt_windowpane, but another type was passed.'
    default_current = dictionary.pop('default_current', 1e5)
    surface_representation = dictionary.pop('surface_representation', None)
    surface_extension = dictionary.pop('surface_extension', 0.)
    normal_to_winding = dictionary.pop('normal_to_winding', False)
    ntoroidalcurves = dictionary['ntoroidalcurves']
    npoloidalcurves = dictionary['npoloidalcurves']
    R0 = dictionary.pop('R0', 1.)
    R1 = dictionary.pop('R1', 0.5)
    coil_radius = dictionary.pop('coil_radius', None)
    order = dictionary.pop('order', 5)
    numquadpoints = dictionary.pop('numquadpoints', None)
    if numquadpoints == 'None':
        numquadpoints = None
    fixed = dictionary.pop('fixed', None)
    fixed = fixed_to_list(fixed, order)
    dofs_dictionary = dictionary.pop('dofs', {})
    dofs_dictionary = clean_simsopt_dofs_dictionary(dofs_dictionary, order)
    optimizables_dictionary = dictionary.pop('optimizables', {})
    optimizables_dictionary, fixed = clean_optimizables_dictionary(optimizables_dictionary, 'simsopt_windowpane', fixed)
    curves, ntoroidalcurves, npoloidalcurves, unfixed_orders, centers, center_tors, center_pols, full_dofs, simsopt_dofs, surface_function = init_simsopt_windowpane_coils(surface_representation, ntoroidalcurves, npoloidalcurves, R0=R0, R1=R1, coil_radius = coil_radius, order=order, numquadpoints=numquadpoints, fixed=fixed, normal_to_winding=normal_to_winding, surface_extension=surface_extension, dofs = dofs_dictionary)
    current_dictionary = dictionary.pop('currents', {})
    current_dictionary = clean_current_dictionary(current_dictionary, (ntoroidalcurves, npoloidalcurves), default_current)
    num_currents = current_dictionary.pop('num_currents')
    new_dictionary = {
        'type' : 'simsopt_windowpane',
        'num_currents' : num_currents,
        'surface_function' : surface_function,
        'normal_to_winding' : normal_to_winding,
        'ntoroidalcurves' : ntoroidalcurves,
        'npoloidalcurves' : npoloidalcurves,
        'order' : order,
        'fixed' : fixed,
        'unfixed_orders' : unfixed_orders,
        'curves' : curves,
        'full_dofs' : full_dofs,
        'dofs' : simsopt_dofs,
        'centers' : centers,
        'center_tors' : center_tors,
        'center_pols' : center_pols,
        'currents' : current_dictionary,
        'optimizables' : optimizables_dictionary
    }
    return new_dictionary

def import_cyclone_windowpane_coils(dictionary):
    assert dictionary['type'] == 'cyclone_windowpane', 'Type of coils should be cyclone_windowpane, but another type was passed.'
    default_current = dictionary.pop('default_current', 1e5)
    planar_flag = dictionary['planar_flag']
    surface_representation = dictionary.pop('surface_representation', None)
    surface_extension = dictionary.pop('surface_extension', 0.)
    normal_to_winding = dictionary.pop('normal_to_winding', False)
    ntoroidalcurves = dictionary['ntoroidalcurves']
    npoloidalcurves = dictionary['npoloidalcurves']
    unique_shapes = dictionary.pop('unique_shapes', None)
    normal_tors = dictionary.pop('normal_tors', None)
    normal_pols = dictionary.pop('normal_pols', None)
    rotation_matrix = dictionary.pop('rotation_matrix', None)
    tile_as = dictionary.pop('tile_as', 'tile')
    R0 = dictionary.pop('R0', 1.)
    R1 = dictionary.pop('R1', 0.5)
    coil_radius = dictionary.pop('coil_radius', None)
    order = dictionary.pop('order', 5)
    numquadpoints = dictionary.pop('numquadpoints', None)
    if numquadpoints == 'None':
        numquadpoints = None
    fixed = dictionary.pop('fixed', None)
    fixed = fixed_to_list(fixed, order)
    sin_cos_components_dictionary = dictionary.pop('sin_cos_components', {})
    optimizables_dictionary = dictionary.pop('optimizables', {})
    optimizables_dictionary['planar_flag'] = planar_flag
    optimizables_dictionary, fixed = clean_optimizables_dictionary(optimizables_dictionary, 'cyclone_windowpane', fixed)
    curves, ntoroidalcurves, npoloidalcurves, unfixed_orders, planar_dofs, full_planar_dofs, nonplanar_dofs, full_nonplanar_dofs, normal_tors, normal_pols, rotation_angles, centers, center_tors, center_pols, unique_shapes, shapes_matrix, curve_shapes, surface_function = init_windowpane_coils(surface_representation, ntoroidalcurves, npoloidalcurves, unique_shapes=unique_shapes, tor_angles=normal_tors, pol_angles=normal_pols, rotation_matrix=rotation_matrix, tile_as=tile_as, R0=R0, R1=R1, coil_radius = coil_radius, order=order, numquadpoints=numquadpoints, fixed=fixed, normal_to_winding=normal_to_winding, surface_extension=surface_extension, sin_cos_components = sin_cos_components_dictionary)
    if (unique_shapes / sum(npoloidalcurves)) > (1 - 0.5 / order):
        print('Given the number of unique shapes, the number of curves, and the order of the curves, less dofs will be created using simsopt curves than cyclone curves. You may want to consider augmenting your config file to use simsopt coils instead depending on your use case.')
    current_dictionary = dictionary.pop('currents', {})
    current_dictionary = clean_current_dictionary(current_dictionary, (ntoroidalcurves, npoloidalcurves), default_current, shapes_matrix = shapes_matrix)
    num_currents = current_dictionary.pop('num_currents')
    new_dictionary = {
        'type' : 'cyclone_windowpane',
        'num_currents' : num_currents,
        'planar_flag' : planar_flag,
        'surface_function' : surface_function,
        'normal_to_winding' : normal_to_winding,
        'ntoroidalcurves' : ntoroidalcurves,
        'npoloidalcurves' : npoloidalcurves,
        'order' : order,
        'fixed' : fixed,
        'unfixed_orders' : unfixed_orders,
        'unique_shapes' : unique_shapes,
        'shapes_matrix' : shapes_matrix,
        'curves' : curves,
        'curve_shapes' : curve_shapes,
        'full_planar_dofs' : full_planar_dofs,
        'planar_dofs' : planar_dofs,
        'full_nonplanar_dofs' : full_nonplanar_dofs,
        'nonplanar_dofs' : nonplanar_dofs,
        'centers' : centers,
        'center_tors' : center_tors,
        'center_pols' : center_pols,
        'normal_tors' : normal_tors,
        'normal_pols' : normal_pols,
        'rotation_angles' : rotation_angles,
        'currents' : current_dictionary,
        'optimizables' : optimizables_dictionary
    }
    return new_dictionary

def read_in_toml_config(toml_file):
    full_dictionary = {'magnetic_surface':[], 'coil_sets':[], 'num_currents':[], 'all_current_values':[], 'all_curves':[]}
    with open(toml_file, mode="rb") as fp:
        config = tomli.load(fp)
    optimization_dict = config['optimization']
    magnetic_surface = import_magnetic_surface(optimization_dict['magnetic_surface'])
    full_dictionary['magnetic_surface'] = magnetic_surface
    nfp = magnetic_surface.nfp
    stellsym = magnetic_surface.stellsym
    if 'simsopt_existing_config' in config:
        existing_dict = config['simsopt_existing_config']
        if 'ncsx' in existing_dict and existing_dict['ncsx'] == True:
            from simsopt.configs import get_ncsx_data
            ncsx_curves, currents, ma = get_ncsx_data()
            #something that turns this into a full dictionary entry of its own
        elif 'ncsx' in existing_dict:
            pass
        else:
            raise ValueError('Not yet implemented')
        if len(existing_dict.keys()) > 1:
            raise ValueError('Not yet implemented')
    if 'oneoverR' in config:
        oneoverR_dict = config['oneoverR']
        if oneoverR_dict['oneoverR'] == True:
            full_dictionary['oneoverR'] = import_oneoverR_field(oneoverR_dict)
    for key in config.keys():
        if not 'type' in config[key]:
            continue
        label = key
        full_dictionary['coil_sets'].append(label)
        this_dict = config[label]
        if this_dict['type'] == 'simsopt_stellarator':
            full_dictionary[label] = import_simsopt_stellarator_coils(this_dict)
        elif this_dict['type'] == 'cyclone_stellarator':
            full_dictionary[label] = import_cyclone_stellarator_coils(this_dict)
        elif this_dict['type'] == 'simsopt_windowpane':
            full_dictionary[label] = import_simsopt_windowpane_coils(this_dict)
        elif this_dict['type'] == 'cyclone_windowpane':
            full_dictionary[label] = import_cyclone_windowpane_coils(this_dict)
        elif this_dict['type'] in ['cws_stellarator', 'cws_windowpane']:
            raise ValueError('CWS coils are not yet implemented')
        else:
            raise ValueError('Not a valid type for {}'.format(label))
        full_dictionary['num_currents'].append(full_dictionary[label]['num_currents'])
        #temporary fix
        full_dictionary['num_current_dofs'] = sum(full_dictionary['num_currents'])
        full_dictionary['all_current_values'].extend(full_dictionary[label]['currents']['current_list'])
        full_dictionary['all_curves'].extend(full_dictionary[label]['curves'])
    simsopt_current_value = optimization_dict.pop('simsopt_current_value', 1)
    full_dictionary['optimization_parameters'] = clean_optimization_dictionary(optimization_dict)
    full_dictionary['all_currents'] = [(current/simsopt_current_value) * Current(simsopt_current_value) for current in full_dictionary['all_current_values']]
    full_dictionary['all_coils'] = coils_via_symmetries(full_dictionary['all_curves'], full_dictionary['all_currents'], nfp, stellsym)
    biot_savart_field = BiotSavart(full_dictionary['all_coils'])
    full_dictionary['first_coil_label_number'] = int(full_dictionary['all_coils'][0].current.dof_names[0].replace(':x0','').replace('Current',''))
    if 'oneoverR' in full_dictionary:
        biot_savart_field = biot_savart_field + full_dictionary['oneoverR']['field']
    full_dictionary['biot_savart_field'] = biot_savart_field.set_points(magnetic_surface.gamma().reshape((-1,3)))
    return full_dictionary