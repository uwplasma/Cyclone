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

def update_all_planar_dofs_from_unfixed_planar_dofs(planar_dofs, unfixed_orders, full_planar_dofs, unique_shapes):
    unfixed_orders = sorted(unfixed_orders)
    num_unfixed_orders = len(unfixed_orders)
    if 0 in unfixed_orders:
        num_unfixed_orders = num_unfixed_orders - 1
    highest_order = int(len(full_planar_dofs) / (2*2*unique_shapes))
    updated_full_planar_dofs = []
    tally=0
    for i in range(unique_shapes):
        this_updated_full_planar_dofs = full_planar_dofs[2*2*highest_order*i:2*2*highest_order*(i+1)]
        for j in range(highest_order):
            if j+1 in unfixed_orders:
                this_updated_full_planar_dofs[4*j:4*(j+1)] = planar_dofs[tally:tally+4]
                tally = tally + 4
        updated_full_planar_dofs.extend(this_updated_full_planar_dofs)
    return updated_full_planar_dofs

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

def update_all_nonplanar_dofs_from_unfixed_nonplanar_dofs(nonplanar_dofs, unfixed_orders, full_nonplanar_dofs, unique_shapes):
    unfixed_orders = sorted(unfixed_orders)
    num_unfixed_orders = len(unfixed_orders)
    if 0 in unfixed_orders:
        num_unfixed_orders = num_unfixed_orders - 1
    highest_order = int(len(full_nonplanar_dofs) / (2*2*unique_shapes))
    updated_full_nonplanar_dofs = []
    tally=0
    for i in range(unique_shapes):
        this_updated_full_nonplanar_dofs = full_nonplanar_dofs[2*highest_order*i:2*highest_order*(i+1)]
        for j in range(highest_order):
            if j+1 in unfixed_orders:
                this_updated_full_nonplanar_dofs[2*j:2*(j+1)] = nonplanar_dofs[tally:tally+2]
                tally = tally + 2
        updated_full_nonplanar_dofs.extend(this_updated_full_nonplanar_dofs)
    return updated_full_nonplanar_dofs

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

def update_all_simsopt_dofs_from_unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders, full_simsopt_dofs, ncurves):
    unfixed_orders = sorted(unfixed_orders)
    num_unfixed_orders = len(unfixed_orders)
    if 0 in unfixed_orders:
        num_unfixed_orders = num_unfixed_orders - 1
    highest_order = int(len(full_simsopt_dofs) / 6)
    updated_full_simsopt_dofs = []
    tally=0
    for i in range(ncurves):
        this_updated_full_simsopt_dofs = full_simsopt_dofs[6*highest_order*i:6*highest_order*(i+1)]
        for j in range(highest_order):
            if j+1 in unfixed_orders:
                this_updated_full_simsopt_dofs[6*j:6*(j+1)] = simsopt_dofs[tally:tally+6]
                tally = tally + 6
        updated_full_simsopt_dofs.extend(this_updated_full_simsopt_dofs)
    return updated_full_simsopt_dofs