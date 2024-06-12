#!/usr/bin/env python
import numpy as np
from math import sin, cos

def create_windowpane_curves(ntoroidalcurves, npoloidalcurves, nfp, stellsym, R0=1.0, R1=0.5, max_radius_prop=0.8, order=1, numquadpoints=None, fixed=True):
    """
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
    poloidal_max_radius = R1*(1-cos((2*np.pi)/npoloidalcurves))
    toroidal_max_radius = (R0-R1)*(1-cos((2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)))
    coil_radius = max_radius_prop*min(poloidal_max_radius, toroidal_max_radius)
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
    poloidal_max_radius = R1*(1-cos((2*np.pi)/npoloidalcurves))
    toroidal_max_radius = (R0-R1)*(1-cos((2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)))
    max_radius = min(poloidal_max_radius, toroidal_max_radius)
    modular_curves = []
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    for i in range(ntoroidalcurves):
        curve = CurveXYZFourier(numquadpoints, order=1)
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ntoroidalcurves)
        curve.set("xc(0)", cos(angle)*R0)
        curve.set("xc(1)", cos(angle)*(R1+max_radius))
        curve.set("yc(0)", sin(angle)*R0)
        curve.set("yc(1)", sin(angle)*(R1+max_radius))
        # The the next line, the minus sign is for consistency with
        # Vmec.external_current(), so the coils create a toroidal field of the
        # proper sign and free-boundary equilibrium works following stage-2 optimization.
        curve.set("zs(1)", -(R1+max_radius))
        curve.x = curve.x  # need to do this to transfer data to C++
        curve.fix_all()
        modular_curves.append(curve)
    return windowpane_curves, modular_curves

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
