import numpy as np

def create_toroidal_angles(ntoroidalcurves, nfp, stellsym):
    if type(ntoroidalcurves) == int and ntoroidalcurves != 0:
        toroidal_angles = [(i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves) for i in range(ntoroidalcurves)]
    else:
        if type(ntoroidalcurves) == float:
            ntoroidalcurves = np.array([ntoroidalcurves])
        if type(ntoroidalcurves) == list:
            ntoroidalcurves = np.array(ntoroidalcurves)
        if (0 in ntoroidalcurves or 1 in ntoroidalcurves) and stellsym == True:
            raise ValueError('You cannot put a coil on the stellarator symmetry boundary (0 or 1)')
        toroidal_angles = ntoroidalcurves * (2*np.pi)/((1+int(stellsym))*nfp)
        ntoroidalcurves = len(ntoroidalcurves)
    return ntoroidalcurves, toroidal_angles

def count_nested_lists(nested_list):
    list_lengths = []
    for list_ in nested_list:
        list_lengths.append(len(list_))
    return tuple(list_lengths)

def create_poloidal_angles(npoloidalcurves, ntoroidalcurves):
    if npoloidalcurves == 0:
        poloidal_angles = [0]
        poloidal_angles = [poloidal_angles for i in range(ntoroidalcurves)]
        npoloidalcurves = tuple([1]*ntoroidalcurves)
        return npoloidalcurves, poloidal_angles
    elif type(npoloidalcurves) == int:
        poloidal_angles = [i*(2*np.pi)/(npoloidalcurves) for i in range(npoloidalcurves)]
        poloidal_angles = [poloidal_angles for i in range(ntoroidalcurves)]
        npoloidalcurves = tuple([npoloidalcurves]*ntoroidalcurves)
        return npoloidalcurves, poloidal_angles
    elif type(npoloidalcurves) == np.ndarray or type(npoloidalcurves) == tuple:
        poloidal_angles = list(npoloidalcurves)
    elif type(npoloidalcurves) == list:
        poloidal_angles = npoloidalcurves
    elif not 0<=npoloidalcurves<=1:
        raise ValueError('npoloidalcurves interpreted as float, but floats cannot be set outside of 0<=x<=1.')
    else:
        poloidal_angles = [npoloidalcurves*2*np.pi]
        poloidal_angles = [poloidal_angles for i in range(ntoroidalcurves)]
        npoloidalcurves = tuple([1]*ntoroidalcurves)
        return npoloidalcurves, poloidal_angles
    ## anything past here is a list
    if all(isinstance(x, (int, float)) for x in poloidal_angles):
        if any(not 0<=x<=1 for x in poloidal_angles):
            pass
        else:
            poloidal_angles = list(np.array(npoloidalcurves)*2*np.pi)
            npoloidalcurves = tuple([len(poloidal_angles)]*ntoroidalcurves)
            poloidal_angles = [poloidal_angles for i in range(ntoroidalcurves)]
            return npoloidalcurves, poloidal_angles
    ## not a list that's meant to be stacked
    assert len(poloidal_angles) == ntoroidalcurves, 'List must be of length ntoroidalcurves'
    for i, entry in enumerate(poloidal_angles):
        if entry == 0:
            poloidal_angles[i] = [0]
        elif type(entry) == int:
            poloidal_angles[i] = [i*(2*np.pi)/(entry) for i in range(entry)]
        elif type(entry) == np.ndarray or type(entry) == tuple or type(entry) == list:
            if any(not 0<=x<=1 for x in entry):
                raise ValueError('floats expected, but floats cannot be set outside of 0<=x<=1.')
            poloidal_angles[i] = list(np.array(entry)*2*np.pi)
        elif not 0<=entry<=1:
            raise ValueError('floats expected, but floats cannot be set outside of 0<=x<=1.')
        else:
            poloidal_angles[i] = [entry*2*np.pi]
    npoloidalcurves = count_nested_lists(poloidal_angles)
    return npoloidalcurves, poloidal_angles

def check_tor_pol_rotation_angles(tor_angles, pol_angles, rotation_matrix, angles_size):
    if type(angles_size) == int:
        angles_size = (angles_size,)
    if len(angles_size) == 1 or (len(angles_size) == 2 and type(angles_size[1]) == int):
        if ((tor_angles is not None) or (pol_angles is not None)):
            if type(tor_angles) == list:
                tor_angles = np.array(tor_angles)
            if type(pol_angles) == list:
                pol_angles = np.array(pol_angles)
            if not ((tor_angles is not None) and (pol_angles is not None)):
                raise ValueError('If one of tor_angles and pol_angles is specified, they must both be specified')
            else:
                assert tor_angles.shape == pol_angles.shape == angles_size, 'Wrong number or shape of toroidal or poloidal angles specified'
        if rotation_matrix is not None:
            if type(rotation_matrix) == list:
                rotation_matrix = np.array(rotation_matrix)
            assert rotation_matrix.shape == angles_size, 'Shape of rotation matrix must equal (number of toroidal curves, number of poloidal curves)'
        else:
            rotation_matrix = np.zeros(angles_size)
        return tor_angles, pol_angles, rotation_matrix
    elif len(angles_size) == 2:
        n_angles = sum(angles_size[1])
        angles_size = (angles_size[0], *[angles_size[1][i] for i in range(len(angles_size[1]))])
    else:
        n_angles = sum(angles_size[1:])
    n_rings = angles_size[0]
    assert len(angles_size[1:]) == n_rings, 'Number of curve rings specified is different from number expected from shapes_size[0].'
    angles_size = angles_size[1:]
    if ((tor_angles is not None) or (pol_angles is not None)):
        if type(tor_angles) == np.ndarray:
            tor_angles = list(tor_angles)
            for i in range(n_rings):
                tor_angles[i] = list(tor_angles[i])
        if type(pol_angles) == np.ndarray:
            pol_angles = list(pol_angles)
            for i in range(n_rings):
                pol_angles[i] = list(pol_angles[i])
        if not ((tor_angles is not None) and (pol_angles is not None)):
            raise ValueError('If one of tor_angles and pol_angles is specified, they must both be specified')
        else:
            for i in range(n_rings):
                assert len(tor_angles[i]) == len(pol_angles[i]) == angles_size[i], 'Wrong number or shape of toroidal or poloidal angles specified'
    if rotation_matrix is not None:
        if type(rotation_matrix) == np.ndarray:
            rotation_matrix = list(rotation_matrix)
            for i in range(n_rings):
                rotation_matrix[i] = list(rotation_matrix[i])
        for i in range(n_rings):
            assert len(rotation_matrix[i]) == angles_size[i], 'Shape of rotation matrix must equal (number of toroidal curves, number of poloidal curves)'
    else:
        tally = 0
        full_tile = np.zeros((n_angles,))
        rotation_matrix = [[] for i in range(n_rings)]
        for i, num in enumerate(angles_size):
            rotation_matrix[i] = full_tile[tally:(tally+num)]
            tally += num
    return tor_angles, pol_angles, rotation_matrix