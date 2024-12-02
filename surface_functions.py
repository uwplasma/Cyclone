import numpy as np
import jax.numpy as jnp
from simsopt.geo import SurfaceRZFourier

def r_surface(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    num_m_modes = len(rsurfacecc)
    num_n_modes = len(rsurfacecc[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    r = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            r += rsurfacecc[m][n]*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if rsurfacecs is not None:
        assert len(rsurfacecc) == len(rsurfacecs), 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                r += rsurfacecs[m][n]*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return r

def z_surface(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    num_m_modes = len(zsurfacecs)
    num_n_modes = len(zsurfacecs[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    z = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            z += zsurfacecs[m][n]*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if zsurfacecc is not None:
        assert len(zsurfacecs) == len(zsurfacecc), 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                z += zsurfacecc[m][n]*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return z

def r_surface_prime_phi(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    num_m_modes = len(rsurfacecc)
    num_n_modes = len(rsurfacecc[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    r_phi = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            r_phi += rsurfacecc[m][n]*(n+n_min)*nfp*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if rsurfacecs is not None:
        assert len(rsurfacecc) == len(rsurfacecs), 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                r_phi += -rsurfacecs[m][n]*(n+n_min)*nfp*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return r_phi

def z_surface_prime_phi(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    num_m_modes = len(zsurfacecs)
    num_n_modes = len(zsurfacecs[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    z_phi = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            z_phi += -zsurfacecs[m][n]*(n+n_min)*nfp*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if zsurfacecc is not None:
        assert len(zsurfacecs) == len(zsurfacecc), 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                z_phi += zsurfacecc[m][n]*(n+n_min)*nfp*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return z_phi

def surface_del_phi(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    r = r_surface(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    r_phi = r_surface_prime_phi(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    z_phi = z_surface_prime_phi(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc=zsurfacecc, n_min=n_min, m_min=m_min)
    costerm = jnp.cos(tor_angle)
    sinterm = jnp.sin(pol_angle)
    return [r_phi*costerm - r*sinterm, r_phi*sinterm + r*costerm, z_phi]

def r_surface_prime_theta(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    num_m_modes = len(rsurfacecc)
    num_n_modes = len(rsurfacecc[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    r_theta = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            r_theta += -rsurfacecc[m][n]*(m+m_min)*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if rsurfacecs is not None:
        assert len(rsurfacecc) == len(rsurfacecs), 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                r_theta += rsurfacecs[m][n]*(m+m_min)*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return r_theta

def z_surface_prime_theta(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    num_m_modes = len(zsurfacecs)
    num_n_modes = len(zsurfacecs[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    z_theta = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            z_theta += zsurfacecs[m][n]*(m+m_min)*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if zsurfacecc is not None:
        assert len(zsurfacecs) == len(zsurfacecc), 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                z_theta += -zsurfacecc[m][n]*(m+m_min)*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return z_theta

def surface_del_theta(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    r_theta = r_surface_prime_theta(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    z_theta = z_surface_prime_theta(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc=zsurfacecc, n_min=n_min, m_min=m_min)
    return [r_theta*jnp.cos(tor_angle), r_theta*jnp.sin(pol_angle), z_theta]

def surface_normal(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    del_phi = surface_del_phi(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = rsurfacecs, zsurfacecc = zsurfacecc, n_min = n_min, m_min = m_min)
    del_theta = surface_del_theta(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = rsurfacecs, zsurfacecc = zsurfacecc, n_min = n_min, m_min = m_min)
    return [del_phi[1]*del_theta[2]-del_phi[2]*del_theta[1],
            del_phi[2]*del_theta[0]-del_phi[0]*del_theta[2],
            del_phi[0]*del_theta[1]-del_phi[1]*del_theta[0]]

def surface_unitnormal(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    normal = np.array(surface_normal(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = rsurfacecs, zsurfacecc = zsurfacecc, n_min = n_min, m_min = m_min))
    return list(normal / np.sqrt(normal.dot(normal)))

def surface_from_vmec_input(file, surface_extension):
    this_surface = SurfaceRZFourier.from_vmec_input(file)
    nfp = this_surface.nfp
    stellsym = this_surface.stellsym
    this_surface.extend_via_normal(surface_extension)
    rsurfacecc = this_surface.rc
    zsurfacecs = this_surface.zs
    rsurfacecs = None
    zsurfacecc = None
    if not stellsym:
        rsurfacecs = this_surface.rs
        zsurfacecc = this_surface.zc
    return nfp, stellsym, rsurfacecc, zsurfacecs, rsurfacecs, zsurfacecc

def surface_from_wout(file, surface_extension):
    this_surface = SurfaceRZFourier.from_wout(file)
    nfp = this_surface.nfp
    stellsym = this_surface.stellsym
    this_surface.extend_via_normal(surface_extension)
    rsurfacecc = this_surface.rc
    zsurfacecs = this_surface.zs
    rsurfacecs = None
    zsurfacecc = None
    if not stellsym:
        rsurfacecs = this_surface.rs
        zsurfacecc = this_surface.zc
    return nfp, stellsym, rsurfacecc, zsurfacecs, rsurfacecs, zsurfacecc

def surface_from_list():
    return None

def surface_from_dict():
    return None

#  np.array([cos(tor_angle)*cos(pol_angle), sin(tor_angle)*cos(pol_angle), sin(pol_angle)])

def import_surface(surface, surface_extension, normal_to_winding):
    if 'input.' in surface:
        n_min = None
        m_min = 0
        nfp, stellsym, rsurfacecc, zsurfacecs, rsurfacecs, zsurfacecc = surface_from_vmec_input(surface, surface_extension)
    elif 'wout' in surface and '.nc' in surface:
        n_min = None
        m_min = 0
        nfp, stellsym, rsurfacecc, zsurfacecs, rsurfacecs, zsurfacecc = surface_from_wout(surface, surface_extension)
    elif surface is None or surface == 'default':
        n_min = None
        m_min = 0
        raise TypeError('Default not implemented yet.')
    elif type(surface) == list or type(surface) == dict:
        n_min = None
        m_min = 0
        raise TypeError('User input lists or dictionaries for surface parameters are not presently supported.')
    else:
        raise TypeError('Axis could not be interpreted as VMEC input. file or VMEC wout.nc file.')
    def surface_function(tor_angle, pol_angle):
        r = r_surface(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
        return jnp.array([r*jnp.cos(tor_angle), r*jnp.sin(tor_angle), z_surface(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc=zsurfacecc, n_min=n_min, m_min=m_min)])
    if normal_to_winding:
        def normal_vec_function(tor_angle, pol_angle):
            return surface_normal(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, zsurfacecc=zsurfacecc, n_min=n_min, m_min=m_min)
    else:
        def normal_vec_function(tor_angle, pol_angle):
            return jnp.array([jnp.cos(tor_angle)*jnp.cos(pol_angle), jnp.sin(tor_angle)*jnp.cos(pol_angle), jnp.sin(pol_angle)])
    return nfp, stellsym, surface_function, normal_vec_function
