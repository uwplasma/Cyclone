import numpy as np

def fix_order(curve, order):
    if not order == 0:
        curve.fix("xc({})".format(order))
        curve.fix("xs({})".format(order))
        curve.fix("yc({})".format(order))
        curve.fix("ys({})".format(order))
        curve.fix("zc({})".format(order))
        curve.fix("zs({})".format(order))
    else:
        curve.fix("xc(0)")
        curve.fix("yc(0)")
        curve.fix("zc(0)")

def fix_curve(curve, order, fixed):
    if (type(fixed) == str and fixed == 'all') or \
    (type(fixed) == int and fixed>=order) or \
    ((type(fixed) == list or type(fixed) == np.ndarray) and \
            sum([order_n in fixed for order_n in list(range(order+1))])/(order+1) == 1):
        curve.fix_all()
    elif type(fixed) == int:
        for ind in range(order+1):
            if ind in range(fixed+1):
                fix_order(curve, ind)
    elif type(fixed) == list or type(fixed) == np.ndarray:
        for ind in range(order+1):
            if ind in fixed:
                fix_order(curve, ind)
    elif fixed is None:
        pass
    else:
        raise TypeError('The input you entered for the \'fixed\' variable is not supported.')

def return_unfixed_orders(order, fixed):
    unfixed = []
    if (type(fixed) == str and fixed == 'all') or \
    (type(fixed) == int and fixed>=order) or \
    ((type(fixed) == list or type(fixed) == np.ndarray) and \
            sum([order_n in fixed for order_n in list(range(order+1))])/(order+1) == 1):
        pass
    elif type(fixed) == int:
        for orde in range(order+1):
            if not orde in range(fixed+1):
                unfixed.append(orde)
    elif type(fixed) == list or type(fixed) == np.ndarray:
        for orde in range(order+1):
            if not orde in fixed:
                unfixed.append(orde)
    elif fixed is None:
        for orde in range(order+1):
            unfixed.append(orde)
    else:
        TypeError('The input you entered for the \'fixed\' variable is not supported.')
    return unfixed

def all_planar_dofs(sin_components_all, cos_components_all):
    unique_shapes = len(sin_components_all)
    dofs = []
    for i in range(unique_shapes):
        sindofs = []
        cosdofs = []
        for sub_list in sin_components_all[i]:
            sindofs.extend(sub_list)
        for sub_list in cos_components_all[i]:
            cosdofs.extend(sub_list)
        dofs.extend(sindofs)
        dofs.extend(cosdofs)
    return dofs

def unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders):
    unique_shapes = len(sin_components_all)
    unfixed_orders = sorted(unfixed_orders)
    dofs = []
    for i in range(unique_shapes):
        sindofs = []
        cosdofs = []
        for order in unfixed_orders:
            if order == 0:
                continue
            sindofs.extend(sin_components_all[i][order-1])
            cosdofs.extend(cos_components_all[i][order-1])
        dofs.extend(sindofs)
        dofs.extend(cosdofs)
    return dofs

def all_nonplanar_dofs(sin_components_all, cos_components_all):
    unique_shapes = len(sin_components_all)
    dofs = []
    for i in range(unique_shapes):
        sindofs = []
        cosdofs = []
        for entry in sin_components_all[i]:
            sindofs.append(entry)
        for entry in cos_components_all[i]:
            cosdofs.append(entry)
        dofs.extend(sindofs)
        dofs.extend(cosdofs)
    return dofs

def unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders):
    unique_shapes = len(nonplanar_sin_components_all)
    unfixed_orders = sorted(unfixed_orders)
    dofs = []
    for i in range(unique_shapes):
        sindofs = []
        cosdofs = []
        for order in unfixed_orders:
            if order == 0:
                continue
            sindofs.append(nonplanar_sin_components_all[i][order-1])
            cosdofs.append(nonplanar_cos_components_all[i][order-1])
        dofs.extend(sindofs)
        dofs.extend(cosdofs)
    return dofs

def unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders):
    ncurves = len(simsopt_dofs)
    unfixed_orders = sorted(unfixed_orders)
    dofs = []
    for i in range(ncurves):
        for order in unfixed_orders:
            if order == 0:
                continue
            dofs.extend(simsopt_dofs[i][6*(order-1):6*order])
    return dofs