import numpy as np
from math import cos, sin, atan2, sqrt
from create_windowpanes import planar_vector_list

def r_axis(raxiscc, nfp, tor_angle, raxiscs = None):
    """Given the cosine (and optionally sine) Fourier components
    of the r coordinate of the magnetic axis, returns the r coordinate
    at the given toroidal angle.

    Args:
        raxiscc (list(float)): Cosine Fourier components of the r coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the r coordinate of the magnetic axis.
        raxiscs (list(float), optional): Sine Fourier components of the r coordinate. Defaults to None.

    Returns:
        float: The r coordinate of the magnetic axis at the given toroidal angle.
    """
    r = 0
    for m in range(len(raxiscc)):
        r += raxiscc[m] * cos(nfp * m * tor_angle)
    if raxiscs is not None:
        for m in range(len(raxiscs)):
            r += raxiscs[m] * sin(nfp * m * tor_angle)
    return r

def z_axis(zaxiscs, nfp, tor_angle, zaxiscc = None):
    """Given the sine (and optionally cosine) Fourier components
    of the z coordinate of the magnetic axis, returns the z coordinate
    at the given toroidal angle.

    Args:
        zaxiscs (list(float)): Sine Fourier components of the z coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the r coordinate of the magnetic axis.
        zaxiscc (list(float), optional): Cosine Fourier components of the r coordinate. Defaults to None. Defaults to None.

    Returns:
        float: The z coordinate of the magnetic axis at the given toroidal angle.
    """
    z = 0
    for m in range(len(zaxiscs)):
        z += zaxiscs[m] * sin(nfp * m * tor_angle)
    if zaxiscc is not None:
        for m in range(len(zaxiscc)):
            z += zaxiscc[m] * cos(nfp * m * tor_angle)
    return z

def r_axis_prime(raxiscc, nfp, tor_angle, raxiscs = None):
    """Given the cosine (and optionally sine) Fourier components
    of the r coordinate of the magnetic axis, returns the derivative
    of the r coordinate at the given toroidal angle.

    Args:
        raxiscc (list(float)): Cosine Fourier components of the r coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the r coordinate of the magnetic axis.
        raxiscs (list(float), optional): Sine Fourier components of the r coordinate. Defaults to None.

    Returns:
        float: The derivative of the r component of the magnetic axis with respect to the toroidal angle at the given toroidal angle.
    """
    r_ = 0
    for m in range(len(raxiscc)):
        r_ += -nfp * m * raxiscc[m] * sin(nfp * m * tor_angle)
    if raxiscs is not None:
        for m in range(len(raxiscs)):
            r_ += nfp * m * raxiscs[m] * cos(nfp * m * tor_angle)
    return r_

def z_axis_prime(zaxiscs, nfp, tor_angle, zaxiscc = None):
    """Given the sine (and optionally cosine) Fourier components
    of the z coordinate of the magnetic axis, returns the derivative
    of the z coordinate at the given toroidal angle.

    Args:
        zaxiscs (list(float)): Sine Fourier components of the z coordinate.
        tor_angle (float): Toroidal angle at which to evaluate the z coordinate of the magnetic axis.
        zaxiscc (list(float), optional): Cosine Fourier components of the z coordinate. Defaults to None.

    Returns:
        float: The derivative of the z component of the magnetic axis with respect to the toroidal angle at the given toroidal angle.
    """
    z_ = 0
    for m in range(len(zaxiscs)):
        z_ += nfp * m * zaxiscs[m] * cos(nfp * m * tor_angle)
    if zaxiscc is not None:
        for m in range(len(zaxiscc)):
            z_ += -nfp * m * zaxiscc[m] * sin(nfp * m * tor_angle)
    return z_

def del_phi(raxiscc, zaxiscs, nfp, tor_angle, raxiscs = None, zaxiscc = None):
    """Given the cosine (and optionally sine) Fourier components
    of the r coordinate and the sine (and optionally cosine) Fourier
    components of the z coordinate of the magnetic axis, returns the
    tangent vector to the magnetic axis at the given toroidal angle.

    Args:
        raxiscc (list(float)): Cosine Fourier components of the r coordinate.
        zaxiscs (list(float)): Sine Fourier components of the z coordinate.
        tor_angle (_type_): Toroidal angle at which to evaluate the derivatives of the magnetic axis.
        raxiscs (list(float), optional): Sine Fourier components of the r coordinate. Defaults to None.
        zaxiscc (list(float), optional): Cosine Fourier components of the z coordinate. Defaults to None.

    Returns:
        3_list(float): The tangent vector of the magnetic axis at the given toroidal angle.
    """
    rax = r_axis(raxiscc, nfp, tor_angle, raxiscs = raxiscs)
    rax_ = r_axis_prime(raxiscc, nfp, tor_angle, raxiscs = raxiscs)
    zax_ = z_axis_prime(zaxiscs, nfp, tor_angle, zaxiscc = zaxiscc)
    sinterm = sin(tor_angle)
    costerm = cos(tor_angle)
    return [rax_*costerm - rax*sinterm, rax_*sinterm + rax*costerm, zax_]

def create_TF_coils_on_magnetic_axis(file, ncurves, R1 = 0.5, order=1, numquadpoints=None, fixed='all'):
    """Given a VMEC input. or wout.nc file and the desired number of curves per half-field-period, returns
    a list of circular simsopt CurveXYZFourier objects with radius ''R1'' which are centered on the magnetic axis and whose area
    normals point along the tangent of the magnetic axis equally spaced in toroidal angle.


    Args:
        file (VMEC input. or wout.nc file): The VMEC file from which to read the Fourier components of the magnetic axis.
        ncurves (int): The number of simsopt curves to generate (per half-field-period).
        R1 (float, optional): The radius of the circular curves. Defaults to 0.5.
        order (int, optional): The maximum order of Fourier components to include in the CurveXYZFourier objects. Defaults to 1.
        numquadpoints (int, optional): The number of quadrature points to evalute the curves at. Defaults to None.
        fixed ((str, int, list, or np.ndarray), optional): A description of which - if any - of the orders of the curves to set to fixed. Defaults to 'all'.

    Raises:
        TypeError: The input file from which to determine the magnetic axis Fourier components was not able to be interpreted as a VMEC input. or wout.nc file. Support for user input lists and dictionaries may be added at a future date.
        TypeError: The style of 'fixed' variable input by the user is not currently supported or was not able to be correctly interpreted.

    Returns:
        ncurves_list(simsopt CurveXYZFourier): An ncurves long list containing simsopt CurveXYZFourier objects at locations along the magnetic axis with their area normals pointing along the tangent of the magnetic axis.
    """
    if 'input.' in file:
        import f90nml
        nml = f90nml.read(file)['indata']
        raxiscc = nml['raxis_cc']
        zaxiscs = -np.array(nml['zaxis_cs'])
        nfp = nml['nfp']
        stellsym = not nml['lasym']
        raxiscs = None
        zaxiscc = None
        if not stellsym:
            raxiscs = -np.array(nml['raxis_cs'])
            zaxiscc = nml['zaxis_cc']
    elif 'wout' in file and '.nc' in file:
        from scipy.io import netcdf_file
        f = netcdf_file(file, mmap=False)
        raxiscc = f.variables['raxis_cc'][()]
        zaxiscs = -f.variables['zaxis_cs'][()]
        nfp = f.variables['nfp'][()]
        stellsym = not bool(f.variables['lasym__logical__'][()])
        raxiscs = None
        zaxiscc = None
        if not stellsym:
            raxiscs = -f.variables['raxis_cs'][()]
            zaxiscc = f.variables['zaxis_cc'][()]
    elif type(file) == list or type(file) == dict:
        raise TypeError('User input lists or dictionaries for axis parameters are not presently supported.')
    else:
        raise TypeError('File could not be interpreted as VMEC input. file or VMEC wout.nc file.')
    if numquadpoints is None:
        numquadpoints = 15 * (order + 1)
    curves = []
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i in range(ncurves):
        curve = CurveXYZFourier(numquadpoints, order=order)
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ncurves)
        r = r_axis(raxiscc, nfp, angle, raxiscs = raxiscs)
        normal_vec = del_phi(raxiscc, zaxiscs, nfp, angle, raxiscs = raxiscs, zaxiscc = zaxiscc)
        tor_axis = atan2(normal_vec[1], normal_vec[0])
        pol_axis = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
        planar_vectors = planar_vector_list(tor_axis, pol_axis)
        curve.set("xc(0)", r * cos(angle))
        curve.set("xc(1)", R1 * planar_vectors[0][0])
        curve.set("xs(1)", R1 * planar_vectors[1][0])
        curve.set("yc(0)", r * sin(angle))
        curve.set("yc(1)", R1 * planar_vectors[0][1])
        curve.set("ys(1)", R1 * planar_vectors[1][1])
        curve.set("zc(0)", z_axis(zaxiscs, nfp, angle, zaxiscc = zaxiscc))
        # planar_vectors[0][2] is 0 by definition so there is
        # no need to set the zc(1) component
        curve.set("zs(1)", R1 * planar_vectors[1][2])
        curve.x = curve.x  # need to do this to transfer data to C++
        if (fixed == 'all') or \
           (type(fixed) == int and fixed>=order) or \
           ((type(fixed) == list or type(fixed) == np.ndarray) and \
                sum([order_n in fixed for order_n in list(range(order+1))])/(order+1) == 1):
            curve.fix_all()
        elif type(fixed) == int:
            for i in range(fixed):
                if not i == 0:
                    curve.fix("xc({})".format(i))
                    curve.fix("xs({})".format(i))
                    curve.fix("yc({})".format(i))
                    curve.fix("ys({})".format(i))
                    curve.fix("zc({})".format(i))
                    curve.fix("zs({})".format(i))
                else:
                    curve.fix("xc(0)")
                    curve.fix("yc(0)")
                    curve.fix("zc(0)")
        elif type(fixed) == list or type(fixed) == np.ndarray:
            for i in fixed:
                if not i == 0:
                    curve.fix("xc({})".format(i))
                    curve.fix("xs({})".format(i))
                    curve.fix("yc({})".format(i))
                    curve.fix("ys({})".format(i))
                    curve.fix("zc({})".format(i))
                    curve.fix("zs({})".format(i))
                else:
                    curve.fix("xc(0)")
                    curve.fix("yc(0)")
                    curve.fix("zc(0)")
        elif fixed is None:
            pass
        else:
            TypeError('The input you entered for the \'fixed\' variable is not supported.')
        curves.append(curve)
    return curves
