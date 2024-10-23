import numpy as np
from math import cos, sin, atan2, sqrt
from create_windowpanes import planar_vector_list

def r_axis(raxiscc, tor_angle, raxiscs = None):
    r = 0
    for m in range(len(raxiscc)):
        r += raxiscc[m] * cos(m * tor_angle)
    if raxiscs is not None:
        for m in range(len(raxiscs)):
            r += raxiscs[m] * sin(m * tor_angle)
    return r

def z_axis(zaxiscs, tor_angle, zaxiscc = None):
    z = 0
    for m in range(len(zaxiscs)):
        z += zaxiscs[m] * sin(m * tor_angle)
    if zaxiscc is not None:
        for m in range(len(zaxiscc)):
            z += zaxiscc[m] * cos(m * tor_angle)
    return z

def r_axis_prime(raxiscc, tor_angle, raxiscs = None):
    r_ = 0
    for m in range(len(raxiscc)):
        r_ += -m * raxiscc[m] * sin(m * tor_angle)
    if raxiscs is not None:
        for m in range(len(raxiscs)):
            r_ += m * raxiscs[m] * cos(m * tor_angle)
    return r_

def z_axis_prime(zaxiscs, tor_angle, zaxiscc = None):
    z_ = 0
    for m in range(len(zaxiscs)):
        z_ += m * zaxiscs[m] * cos(m * tor_angle)
    if zaxiscc is not None:
        for m in range(len(zaxiscc)):
            z_ += -m * zaxiscc[m] * sin(m * tor_angle)
    return z_

def del_phi(raxiscc, zaxiscs, tor_angle, raxiscs = None, zaxiscc = None):
    rax = r_axis(raxiscc, tor_angle, raxiscs = raxiscs)
    rax_ = r_axis_prime(raxiscc, tor_angle, raxiscs = raxiscs)
    zax_ = z_axis_prime(zaxiscs, tor_angle, zaxiscc = zaxiscc)
    sinterm = sin(tor_angle)
    costerm = cos(tor_angle)
    return [rax_*costerm - rax*sinterm, rax_*sinterm + rax*costerm, zax_]

def create_TF_coils_on_magnetic_axis(file, ncurves, R1 = 0.5, order=1, numquadpoints=None, fixed='all'):
    if 'input.' in file:
        import f90nml
        nml = f90nml.read('Cyclone/input.QI_NFP1_r1_test')['indata']
        raxiscc = nml['raxis_cc']
        zaxiscs = nml['zaxis_cs']
        nfp = nml['nfp']
        stellsym = not nml['lasym']
        raxiscs = None
        zaxiscc = None
        if not stellsym:
            raxiscs = nml['raxis_cs']
            zaxiscc = nml['zaxis_cc']
    elif 'wout' in file and '.nc' in file:
        from scipy.io import netcdf_file
        f = netcdf_file('Cyclone/wout_QI_NFP1_r1_test.nc', mmap=False)
        raxiscc = f.variables['raxis_cc'][()]
        zaxiscs = f.variables['zaxis_cs'][()]
        nfp = f.variables['nfp'][()]
        stellsym = not bool(f.variables['lasym__logical__'][()])
        raxiscs = None
        zaxiscc = None
        if not stellsym:
            raxiscs = f.variables['raxis_cs'][()]
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
        r = r_axis(raxiscc, angle, raxiscs = raxiscs)
        normal_vec = del_phi(raxiscc, zaxiscs, angle, raxiscs = raxiscs, zaxiscc = zaxiscc)
        tor_axis = atan2(normal_vec[1], normal_vec[0])
        pol_axis = atan2(normal_vec[2], sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2))
        planar_vectors = planar_vector_list(tor_axis, pol_axis)
        curve.set("xc(0)", r * cos(angle))
        curve.set("xc(1)", R1 * planar_vectors[0][0])
        curve.set("xs(1)", R1 * planar_vectors[1][0])
        curve.set("yc(0)", r * sin(angle))
        curve.set("yc(1)", R1 * planar_vectors[0][1])
        curve.set("ys(1)", R1 * planar_vectors[1][1])
        curve.set("zc(0)", z_axis(zaxiscs, angle, zaxiscc = zaxiscc))
        # planar_vectors[1][2] is 0 by definition so there is
        # no need to set the zc(1) component
        curve.set("zs(1)", R1 * planar_vectors[1][2])
        curve.x = curve.x  # need to do this to transfer data to C++
        if (fixed == 'all') or \
           (type(fixed) == int and fixed>order) or \
           ((type(fixed) == list or type(fixed) == np.ndarray) and \
                sum([orm in fixed for orm in list(range(ordd+1))])/(ordd+1) == 1):
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
