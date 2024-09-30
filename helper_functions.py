import numpy as np
from math import sin, cos, sqrt
from qsc.qsc import Qsc
import os

def remove_coils_on_curvature(coils, curvature_threshold, nfp, stellsym):
    """
    Take a list of Coil objects (with field periodicity ''nfp'' and
    obeying the symmetry imposed by ''stellsym'') and return
    the list with Coils whose maximum curvatures
    exceeded ''curvature_threshold'' removed from the list.
    """
    from simsopt.field import coils_via_symmetries
    numindependentcoils = len(coils)/((1+int(stellsym))*nfp)
    assert numindependentcoils==int(numindependentcoils), 'Number of field periods and/or stellsym improperly specified'
    coils = coils[:int(numindependentcoils)]
    curves = [coil.curve for coil in coils]
    currents = [coil.current for coil in coils]
    curvature_list = np.array([np.max(curve.kappa()) for curve in curves])
    curves = list(np.array(curves)[curvature_list < 150])
    currents = list(np.array(currents)[curvature_list < 150])
    coils = coils_via_symmetries(curves, currents, nfp, stellsym = True)
    return coils

def remove_coils_on_current(coils, current_threshold, nfp, stellsym):
    """
    Take a list of Coil objects (with field periodicity ''nfp'' and
    obeying the symmetry imposed by ''stellsym'') and return
    the list with Coils whose currents are below
    ''current_threshold'' removed from the list.
    """
    from simsopt.field import coils_via_symmetries
    numindependentcoils = len(coils)/((1+int(stellsym))*nfp)
    assert numindependentcoils==int(numindependentcoils), 'Number of field periods and/or stellsym improperly specified'
    coils = coils[:int(numindependentcoils)]
    curves = [coil.curve for coil in coils]
    currents = [coil.current for coil in coils]
    current_list = np.abs(np.array([current.get_value() for current in currents]))
    curves = list(np.array(curves)[current_list > current_threshold])
    currents = list(np.array(currents)[current_list > current_threshold])
    coils = coils_via_symmetries(curves, currents, nfp, stellsym = True)
    return coils

def rz_surf_to_rz(surface, tor_angle, pol_angle):
    '''
    This function outputs the r,z coordinates of a
    Simsopt surface at given toroidal and poloidal
    angles.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and a
    poloidal angle.
    Output the r and z coordinates of the surface at
    the specified toroidal and poloidal angles.
    '''
    num_m_modes = len(surface.rc)
    num_n_modes = len(surface.rc[0])
    n_min = -(num_n_modes - 1)/2
    r=0
    z=0
    nfp = surface.nfp
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            sinterm = sin(m*pol_angle - (n+n_min)*nfp*tor_angle)
            costerm = cos(m*pol_angle - (n+n_min)*nfp*tor_angle)
            r += surface.rc[m][n]*costerm+surface.rs[m][n]*sinterm
            z += surface.zc[m][n]*costerm+surface.zs[m][n]*sinterm
    return [r, z]

def rz_to_xyz(r,z,tor_angle):
    '''
    This function performs a coordinate transform
    from r,z coordinates to x,y,z coordinates.
    Input r and z coordinates and a toroidal angle.
    Output the x, y, and z coordinates of that point.
    '''
    return [r*cos(tor_angle), r*sin(tor_angle), z]

def rz_surf_to_xyz(surface, tor_angle, pol_angle):
    '''
    This function outputs the x,y,z coordinates of a
    Simsopt surface at given toroidal and poloidal
    angles. This function is meant to act as a way to specify a
    winding function. An example code block is shown below.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the x, y, and z coordinates of the surface
    at the specified toroidal and poloidal angles.
    
    Usage example: create a winding surface function for use
    with create_arbitrary_windowpanes()
    
    .. code-block::

        surface = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        def winding_surface_function(tor_angle, pol_angle):
            xyz = rz_surf_to_xyz(surface, tor_angle, pol_angle)
            return xyz
    '''
    rz=rz_surf_to_rz(surface, tor_angle, pol_angle)
    return rz_to_xyz(rz[0],rz[1],tor_angle)

def rz_gammadash_phi_to_rz(surface, tor_angle, pol_angle):
    '''
    This function takes a derivative with respect to
    the toroidal angle in r,z coordinates.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the derivatives of r and z with respect to
    the toroidal angle at the given toroidal and
    poloidal angles.
    '''
    num_m_modes = len(surface.rc)
    num_n_modes = len(surface.rc[0])
    n_min = -(num_n_modes - 1)/2
    dr=0
    dz=0
    nfp = surface.nfp
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            sinterm = sin(m*pol_angle-(n+n_min)*nfp*tor_angle)
            costerm = cos(m*pol_angle-(n+n_min)*nfp*tor_angle)
            dr += surface.rc[m][n]*(n+n_min)*nfp*sinterm-surface.rs[m][n]*(n+n_min)*nfp*costerm
            dz += surface.zc[m][n]*(n+n_min)*nfp*sinterm-surface.zs[m][n]*(n+n_min)*nfp*costerm
    return [dr, dz]

def rz_gammadash_phi_to_xyz(surface, tor_angle, pol_angle):
    '''
    This function takes a derivative with respect to
    the toroidal angle in x,y,z coordinates.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the derivatives of x, y, and z with respect to
    the toroidal angle at the given toroidal and
    poloidal angles.
    '''
    drz = rz_gammadash_phi_to_rz(surface, tor_angle, pol_angle)
    rz = rz_surf_to_rz(surface, tor_angle, pol_angle)
    dx = drz[0]*cos(tor_angle)-rz[0]*sin(tor_angle)
    dy = drz[0]*sin(tor_angle)+rz[0]*cos(tor_angle)
    return [dx,dy,drz[1]]

def rz_gammadash_theta_to_rz(surface, tor_angle, pol_angle):
    '''
    This function takes a derivative with respect to
    the poloidal angle in r,z coordinates.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the derivatives of r and z with respect to
    the poloidal angle at the given toroidal and
    poloidal angles.
    '''
    num_m_modes = len(surface.rc)
    num_n_modes = len(surface.rc[0])
    n_min = -(num_n_modes - 1)/2
    dr=0
    dz=0
    nfp = surface.nfp
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            sinterm = sin(m*pol_angle-(n+n_min)*nfp*tor_angle)
            costerm = cos(m*pol_angle-(n+n_min)*nfp*tor_angle)
            dr += -surface.rc[m][n]*m*sinterm+surface.rs[m][n]*m*costerm
            dz += -surface.zc[m][n]*m*sinterm+surface.zs[m][n]*m*costerm
    return [dr, dz]

def rz_gammadash_theta_to_xyz(surface, tor_angle, pol_angle):
    '''
    This function takes a derivative with respect to
    the poloidal angle in x,y,z coordinates.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the derivatives of x, y, and z with respect to
    the poloidal angle at the given toroidal and
    poloidal angles.
    '''
    drz = rz_gammadash_theta_to_rz(surface, tor_angle, pol_angle)
    return rz_to_xyz(drz[0],drz[1],tor_angle)

def rz_normal_to_rz(surface, tor_angle, pol_angle):
    '''
    This function finds the normal vector of the surface at the
    specified toroidal and poloidal angles in r,z coordinates.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the normal vector to the surface at the given toroidal
    and poloidal angles in r,z coordinates.
    '''
    drz_phi = rz_gammadash_phi_to_rz(surface, tor_angle, pol_angle)
    drz_theta = rz_gammadash_theta_to_rz(surface, tor_angle, pol_angle)
    return [drz_phi[1]*drz_theta[2]-drz_phi[2]*drz_theta[1],
            drz_phi[2]*drz_theta[0]-drz_phi[0]*drz_theta[2],
            drz_phi[0]*drz_theta[1]-drz_phi[1]*drz_theta[2]]

def rz_unitnormal_to_rz(surface, tor_angle, pol_angle):
    '''
    This function finds the unit normal vector of the surface at the
    specified toroidal and poloidal angles in r,z coordinates.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the unit normal vector to the surface at the given toroidal
    and poloidal angles in r,z coordinates.
    '''
    normal = np.array(rz_normal_to_rz(surface, tor_angle, pol_angle))
    return list(normal / np.sqrt(normal.dot(normal)))

def rz_normal_to_xyz(surface, tor_angle, pol_angle):
    '''
    This function finds the normal vector of the surface at the
    specified toroidal and poloidal angles in x,y,z coordinates.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the normal vector to the surface at the given toroidal
    and poloidal angles in x,y,z coordinates.
    '''
    dxyz_phi = rz_gammadash_phi_to_xyz(surface, tor_angle, pol_angle)
    dxyz_theta = rz_gammadash_theta_to_xyz(surface, tor_angle, pol_angle)
    return [dxyz_phi[1]*dxyz_theta[2]-dxyz_phi[2]*dxyz_theta[1],
            dxyz_phi[2]*dxyz_theta[0]-dxyz_phi[0]*dxyz_theta[2],
            dxyz_phi[0]*dxyz_theta[1]-dxyz_phi[1]*dxyz_theta[0]]

def rz_unitnormal_to_xyz(surface, tor_angle, pol_angle):
    '''
    This function finds the unit normal vector of the surface at the
    specified toroidal and poloidal angles in x,y,z coordinates.
    This function is meant to act as a way to specify the normal
    vector for a winding function. An example code block is shown below.
    Input a Simsopt SurfaceRZFourier object, a toroidal angle, and
    a poloidal angle.
    Output the unit normal vector to the surface at the given toroidal
    and poloidal angles in x,y,z coordinates.
    
    Usage example: create a winding surface function for use
    with create_arbitrary_windowpanes() with normaltowinding=True
    
    .. code-block::

        surface = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        def winding_surface_function(tor_angle, pol_angle):
            xyz = rz_surf_to_xyz(surface, tor_angle, pol_angle)
            normal_vec = rz_unitnormal_to_xyz(surface, tor_angle, pol_angle)
            return xyz, normal_vec
    '''
    normal = np.array(rz_normal_to_xyz(surface, tor_angle, pol_angle))
    return list(normal / np.sqrt(normal.dot(normal)))

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
    from math import ceil
    all_list = []
    for set_ in args:
        temp_list = np.arange(set_[1], set_[2], set_[3])
        all_list = all_list + [['{}:{}'.format(set_[0], value) for value in temp_list]]
    permutations = list(itertools.product(*all_list))
    return(permutations)
