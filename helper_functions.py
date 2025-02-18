import numpy as np
from math import sin, cos
import jax.numpy as jnp
from qsc.qsc import Qsc
import os

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

def rotate_windowpane_shapes(sin_components, cos_components, rotation_angle, order):
    '''
    This function rotates the input windowpane coil shape around it's normal vector.
    Input the sine and cosine components (in the format of the components
    for the create_arbitrary_windowpanes function), a rotation angle, and
    the order of the windowpane coils.
    Output the sine and cosine components with a rotation applied of
    rotation angle.
    '''
    sin_components_out = jnp.array([], float)
    cos_components_out = jnp.array([], float)
    for i in range(order):
        rot_mat = jnp.append(sin_components[i],cos_components[i]).reshape(2,2)
        rot_mat = rot_mat * jnp.cos(rotation_angle) + jnp.fliplr(rot_mat)*jnp.array([[1,-1],[1,-1]])*jnp.sin(rotation_angle)
        sin_components_out = jnp.append(sin_components_out, rot_mat[0])
        cos_components_out = jnp.append(cos_components_out, rot_mat[1])
    sin_components_out = jnp.reshape(sin_components_out, (order,2))
    cos_components_out = jnp.reshape(cos_components_out, (order,2))
    return sin_components_out, cos_components_out

def scaled_stel(stel_like, alpha):
    if not 0 < abs(alpha):
        raise ValueError('Alpha must not be 0')
    rc = [stel_like.rc[0]] + list(alpha * stel_like.rc[1:])
    zs = [stel_like.zs[0]] + list(alpha * stel_like.zs[1:])
    rs = [stel_like.rs[0]] + list(alpha * stel_like.rs[1:])
    zc = [stel_like.zc[0]] + list(alpha * stel_like.zc[1:])
    nfp = stel_like.nfp
    etabar = stel_like.etabar
    sigma0 = stel_like.sigma0
    B0 = stel_like.B0
    I2 = stel_like.I2
    sG = stel_like.sG
    spsi = stel_like.spsi
    nphi = stel_like.nphi
    B2s = stel_like.B2s
    B2c = stel_like.B2c
    p2 = stel_like.p2
    order = stel_like.order
    new_stel = Qsc(rc=rc, zs=zs, rs=rs, zc=zc, nfp=nfp, etabar=etabar, sigma0=sigma0, B0=B0,
                   I2=I2, sG=sG, spsi=spsi, nphi=nphi, B2s=B2s, B2c=B2c, p2=p2, order=order)
    return new_stel

def save_scaled_iota(name_or_stel, out_dir='', filename = None, r=0.1, first_alpha = 0.1, last_alpha = 1, number_alpha = 13):
    if not out_dir == '':
        os.makedirs(out_dir, exist_ok=True)
    alpha_space = np.linspace(first_alpha, last_alpha, number_alpha)
    if type(name_or_stel) is str:
        name = name_or_stel
        stel = Qsc.from_paper(name_or_stel)
    else:
        name = 'custom'
        stel = name_or_stel
    if filename is not None:
        name = filename
    stel_list = [scaled_stel(stel, alpha) for alpha in alpha_space]
    for stel in stel_list:
        iota = round(stel.iota, 3)
        if iota == 0:
            iota = '<0.001'
        stel.to_vmec(out_dir+'input.'+name.replace(' ','_')+'_iota:{}'.format(iota), r=r)

def generate_permutations(*args):
    import itertools
    all_list = []
    for set_ in args:
        if set_[1] == float:
            temp_list = np.arange(set_[2], set_[3], set_[4])
            all_list = all_list + [['{}:{}'.format(set_[0], value) for value in temp_list]]
        elif set_[1] == int:
            temp_list = np.arange(int(set_[2]), set_[3], int(set_[4]))
            all_list = all_list + [['{}:{}'.format(set_[0], value) for value in temp_list]]
        elif set_[1] == bool:
            if set_[2] is None:
                set_[2] = [True]
            if not isinstance(set_[2], (list, np.ndarray, tuple)):
                set_[2] = [set_[2]]
            assert all([type(inp) == bool for inp in set_[2]]), "Entry into boolean variable is not a boolean."
            # Also going to want to write a check to ensure there is at max only True and False in the list
            all_list = all_list + [['{}:{}'.format(set_[0], value) for value in set_[2]]]
        elif set_[1] == list:
            if set_[2] is None:
                set_[2] = ['default']
            if not isinstance(set_[2], (list, np.ndarray, tuple)):
                set_[2] = [set_[2]]
            all_list = all_list + [['{}:{}'.format(set_[0], value) for value in set_[2]]]
        elif set_[1] == 'file':
            if not isinstance(set_[2], (list, np.ndarray, tuple)):
                set_[2] = [set_[2]]
            all_list = all_list + [['{}:{}'.format(set_[0], value) for value in set_[2]]]
        else:
            raise ValueError('Type of variable could not be successfully parsed.')
    permutations = list(itertools.product(*all_list))
    return(permutations)

file_parameters = ['magnetic_surface', 'stellarator_axis_representation', 'windowpane_surface_representation']
interger_parameters = ['MAXITER', 'stellarator_ncurves', 'stellarator_unique_shapes', 'stellarator_order', 'windowpane_ntoroidalcurves', 'windowpane_npoloidalcurves', 'windowpane_unique_shapes', 'windowpane_order']
string_parameters = ['magnetic_surface', 'stellarator_axis_representation', 'stellarator_tile_as', 'stellarator_center_opt_type_flag', 'windowpane_surface_representation', 'windowpane_tile_as', 'windowpane_center_opt_type_flag']
boolean_parameters = ['stellarator_planar_flag', 'stellarator_rotation_opt_flag', 'stellarator_normal_opt_flag', 'stellarator_center_opt_flag', 'stellarator_planar_opt_flag', 'stellarator_nonplanar_opt_flag', 'windowpane_planar_flag', 'windowpane_normal_to_winding', 'windowpane_rotation_opt_flag', 'windowpane_normal_opt_flag', 'windowpane_center_opt_flag', 'windowpane_planar_opt_flag', 'windowpane_nonplanar_opt_flag']

def create_config_from_input_line(base_config, in_line, out_dir='', filename=None):
    if filename is None:
        filename = 'config_'
    i = 0
    while filename+f'{i}.toml' in os.listdir(out_dir):
        i += 1
    filename = filename + f'{i}'
    parameters = [specification.split(':')[0] for specification in in_line]
    values = [specification.split(':')[1] for specification in in_line]
    with open(base_config, 'r') as f:
        with open('{}/{}.toml'.format(out_dir, filename), 'w') as new_config:
            for line in f.readlines():
                if not 'replace' in line:
                    new_config.write(line)
                    continue
                replacing_parameter = line[line.index("replace")+8:].replace('\n', '')
                if not replacing_parameter in parameters:
                    print(f"{replacing_parameter} not in parameters, so excluding line from config.") # Might not be an issue to just not include the line I guess? I usually do a good job with the default values (not for everything, though, so this is pretty dangerous)
                    continue
                replacing_value = values[parameters.index(replacing_parameter)]
                if replacing_parameter in file_parameters and replacing_value == 'None':
                    replacing_value = values[parameters.index('magnetic_surface')]
                if replacing_value == 'default':
                    #replacing_value = find_default_value(replacing_parameter)
                    print(f"{replacing_parameter} set to default, so excluding line from config.")
                if replacing_parameter in boolean_parameters:
                    replacing_value = replacing_value.lower()
                if replacing_parameter in interger_parameters:
                    replacing_value = f"{int(float(replacing_value))}"
                new_line = line[:line.index("replace")]
                if replacing_parameter in string_parameters:
                    new_line = new_line + f"'{replacing_value}'" + "\n"
                else:
                    new_line = new_line + f"{replacing_value}\n"
                new_config.write(new_line)
    return None