import numpy as np
from Cyclone.coil_initialization import fix_order, fix_curve, all_planar_dofs, return_unfixed_orders, unfixed_planar_dofs, all_nonplanar_dofs, unfixed_nonplanar_dofs, unfixed_simsopt_dofs
from Cyclone.read_in import fixed_to_list
from simsopt.geo import CurveXYZFourier

def test_fix_order():
    # ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve = CurveXYZFourier(5, 5)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)'], 'Should never see this error message'
    order = 5
    fix_order(curve, order)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)']
    order = 3
    fix_order(curve, order)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(4)', 'xc(4)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(4)', 'yc(4)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(4)', 'zc(4)']
    order = 0
    fix_order(curve, order)
    assert curve.local_dof_names == ['xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(4)', 'xc(4)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(4)', 'yc(4)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(4)', 'zc(4)']
    order = 5
    fix_order(curve, order)
    assert curve.local_dof_names == ['xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(4)', 'xc(4)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(4)', 'yc(4)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(4)', 'zc(4)']

def test_fix_curve():
    order = 5
    curve = CurveXYZFourier(5, order)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)'], 'Should never see this error message'
    # Testing all fixed by str
    fixed = 'all'
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    curve.unfix_all()
    # Testing all fixed by int
    fixed = order + 1
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    curve.unfix_all()
    fixed = order + 6
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    curve.unfix_all()
    fixed = order
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    # Testing all fixed by list
    curve.unfix_all()
    fixed = list(range(order+1))
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    curve.unfix_all()
    fixed = list(range(order+6))
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    # Testing all fixed by np.ndarray
    curve.unfix_all()
    fixed = np.array(range(order+1))
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    curve.unfix_all()
    fixed = np.array(range(order+7))
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == []
    # Testing case where no orders are fixed
    curve.unfix_all()
    fixed = None
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = []
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    # Testing cases where some orders are fixed with int
    curve.unfix_all()
    fixed = 0
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = 1
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = 4
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(5)', 'xc(5)', 'ys(5)', 'yc(5)', 'zs(5)', 'zc(5)']
    # Testing cases where some orders are fixed with list
    curve.unfix_all()
    fixed = [0]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = [0, 1]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = [0, 1, 2]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = [0, 1, 2, 3, 4]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(5)', 'xc(5)', 'ys(5)', 'yc(5)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = [3, 5]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(4)', 'xc(4)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(4)', 'yc(4)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(4)', 'zc(4)']
    curve.unfix_all()
    fixed = [1, 6]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = [6, 8, 12]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = [5, 4, 0]
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)']
    # Testing cases where some orders are fixed with np.ndarray
    curve.unfix_all()
    fixed = np.array([0])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = np.array([0, 1])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = np.array([0, 1, 2])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = np.array([0, 1, 2, 3, 4])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(5)', 'xc(5)', 'ys(5)', 'yc(5)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = np.array([3, 5])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(4)', 'xc(4)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(4)', 'yc(4)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(4)', 'zc(4)']
    curve.unfix_all()
    fixed = np.array([1, 6])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = np.array([6, 8, 12])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xc(0)', 'xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'xs(4)', 'xc(4)', 'xs(5)', 'xc(5)', 'yc(0)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'ys(4)', 'yc(4)', 'ys(5)', 'yc(5)', 'zc(0)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)', 'zs(4)', 'zc(4)', 'zs(5)', 'zc(5)']
    curve.unfix_all()
    fixed = np.array([5, 4, 0])
    fix_curve(curve, order, fixed)
    assert curve.local_dof_names == ['xs(1)', 'xc(1)', 'xs(2)', 'xc(2)', 'xs(3)', 'xc(3)', 'ys(1)', 'yc(1)', 'ys(2)', 'yc(2)', 'ys(3)', 'yc(3)', 'zs(1)', 'zc(1)', 'zs(2)', 'zc(2)', 'zs(3)', 'zc(3)']

def test_return_unfixed_orders():
    order = 5
    # Testing all fixed by str
    fixed = 'all'
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    # Testing all fixed by int
    fixed = order + 1
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    fixed = order + 6
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    fixed = order
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    # Testing all fixed by list
    fixed = list(range(order+1))
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    fixed = list(range(order+6))
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    # Testing all fixed by np.ndarray
    fixed = np.array(range(order+1))
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    fixed = np.array(range(order+7))
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == []
    # Testing case where no orders are fixed
    fixed = None
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 1, 2, 3, 4, 5]
    fixed = []
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 1, 2, 3, 4, 5]
    # Testing cases where some orders are fixed with int
    fixed = 0
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [1, 2, 3, 4, 5]
    fixed = 1
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [2, 3, 4, 5]
    fixed = 4
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [5]
    # Testing cases where some orders are fixed with list
    fixed = [0]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [1, 2, 3, 4, 5]
    fixed = [0, 1]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [2, 3, 4, 5]
    fixed = [0, 1, 2]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [3, 4, 5]
    fixed = [0, 1, 2, 3, 4]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [5]
    fixed = [3, 5]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 1, 2, 4]
    fixed = [1, 6]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 2, 3, 4, 5]
    fixed = [6, 8, 12]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 1, 2, 3, 4, 5]
    fixed = [5, 4, 0]
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [1, 2, 3]
    # Testing cases where some orders are fixed with np.ndarray
    fixed = np.array([0])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [1, 2, 3, 4, 5]
    fixed = np.array([0, 1])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [2, 3, 4, 5]
    fixed = np.array([0, 1, 2])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [3, 4, 5]
    fixed = np.array([0, 1, 2, 3, 4])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [5]
    fixed = np.array([3, 5])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 1, 2, 4]
    fixed = np.array([1, 6])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 2, 3, 4, 5]
    fixed = np.array([6, 8, 12])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [0, 1, 2, 3, 4, 5]
    fixed = np.array([5, 4, 0])
    unfixed = return_unfixed_orders(order, fixed)
    assert unfixed == [1, 2, 3]

def test_all_planar_dofs():
    # 3 shapes, order = 5
    sin_components_all = [[[1,2], [3,4], [5,6], [7,8], [9,10]], [[21,22], [23,24], [25,26], [27,28], [29,30]], [[41,42], [43,44], [45,46], [47,48], [49,50]]]
    cos_components_all = [[[11,12], [13,14], [15,16], [17,18], [19,20]], [[31,32], [33,34], [35,36], [37,38], [39,40]], [[51,52], [53,54], [55,56], [57,58], [59,60]]]
    dofs = all_planar_dofs(sin_components_all, cos_components_all)
    assert dofs == list(range(1,61))

def test_unfixed_planar_dofs():
    # 3 shapes, order = 5
    sin_components_all = [[[1,2], [3,4], [5,6], [7,8], [9,10]], [[21,22], [23,24], [25,26], [27,28], [29,30]], [[41,42], [43,44], [45,46], [47,48], [49,50]]]
    cos_components_all = [[[11,12], [13,14], [15,16], [17,18], [19,20]], [[31,32], [33,34], [35,36], [37,38], [39,40]], [[51,52], [53,54], [55,56], [57,58], [59,60]]]
    # Testing configurations where all are unfixed
    unfixed_orders = [0, 1, 2, 3, 4, 5]
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    assert unfixed_dofs == list(range(1,61))
    unfixed_orders = [1, 2, 3, 4, 5]
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    assert unfixed_dofs == list(range(1,61))
    # Testing configurations where all are fixed
    unfixed_orders = []
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    assert unfixed_dofs == []
    unfixed_orders = [0]
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    # Testing configurations where some are unfixed
    assert unfixed_dofs == []
    unfixed_orders = [1]
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    assert unfixed_dofs == [1,2,11,12,21,22,31,32,41,42,51,52]
    unfixed_orders = [3]
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    assert unfixed_dofs == [5,6,15,16,25,26,35,36,45,46,55,56]
    unfixed_orders = [1, 4, 5]
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    assert unfixed_dofs == [1,2,7,8,9,10,11,12,17,18,19,20,21,22,27,28,29,30,31,32,37,38,39,40,41,42,47,48,49,50,51,52,57,58,59,60]
    unfixed_orders = [5, 3, 1]
    unfixed_dofs = unfixed_planar_dofs(sin_components_all, cos_components_all, unfixed_orders)
    assert unfixed_dofs == [1,2,5,6,9,10,11,12,15,16,19,20,21,22,25,26,29,30,31,32,35,36,39,40,41,42,45,46,49,50,51,52,55,56,59,60]

def test_all_nonplanar_dofs():
    # 3 shapes, order = 5
    nonplanar_sin_components_all = [[1,2,3,4,5], [11,12,13,14,15], [21,22,23,24,25]]
    nonplanar_cos_components_all = [[6,7,8,9,10], [16,17,18,19,20], [26,27,28,29,30]]
    dofs = all_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all)
    assert dofs == list(range(1,31))

def test_unfixed_nonplanar_dofs():
    # 3 shapes, order = 5
    nonplanar_sin_components_all = [[1,2,3,4,5], [11,12,13,14,15], [21,22,23,24,25]]
    nonplanar_cos_components_all = [[6,7,8,9,10], [16,17,18,19,20], [26,27,28,29,30]]
    # Testing configurations where all are unfixed
    unfixed_orders = [0, 1, 2, 3, 4, 5]
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    assert unfixed_dofs == list(range(1,31))
    unfixed_orders = [1, 2, 3, 4, 5]
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    assert unfixed_dofs == list(range(1,31))
    # Testing configurations where all are fixed
    unfixed_orders = []
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    assert unfixed_dofs == []
    unfixed_orders = [0]
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    # Testing configurations where some are unfixed
    assert unfixed_dofs == []
    unfixed_orders = [1]
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    assert unfixed_dofs == [1,6,11,16,21,26]
    unfixed_orders = [3]
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    assert unfixed_dofs == [3,8,13,18,23,28]
    unfixed_orders = [1, 4, 5]
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    assert unfixed_dofs == [1,4,5,6,9,10,11,14,15,16,19,20,21,24,25,26,29,30]
    unfixed_orders = [5, 3, 1]
    unfixed_dofs = unfixed_nonplanar_dofs(nonplanar_sin_components_all, nonplanar_cos_components_all, unfixed_orders)
    assert unfixed_dofs == [1,3,5,6,8,10,11,13,15,16,18,20,21,23,25,26,28,30]

def test_unfixed_simsopt_dofs():
    # 3 curves, order = 5
    simsopt_dofs = [list(range(1,31)), list(range(31,61)), list(range(61,91))]
    # Testing configurations where all are unfixed
    unfixed_orders = [0, 1, 2, 3, 4, 5]
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    assert unfixed_dofs == list(range(1,91))
    unfixed_orders = [1, 2, 3, 4, 5]
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    assert unfixed_dofs == list(range(1,91))
    # Testing configurations where all are fixed
    unfixed_orders = []
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    assert unfixed_dofs == []
    unfixed_orders = [0]
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    # Testing configurations where some are unfixed
    assert unfixed_dofs == []
    unfixed_orders = [1]
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    assert unfixed_dofs == [1,2,3,4,5,6,31,32,33,34,35,36,61,62,63,64,65,66]
    unfixed_orders = [3]
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    assert unfixed_dofs == [13,14,15,16,17,18,43,44,45,46,47,48,73,74,75,76,77,78]
    unfixed_orders = [1, 4, 5]
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    assert unfixed_dofs == [1,2,3,4,5,6,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,79,80,81,82,83,84,85,86,87,88,89,90]
    unfixed_orders = [5, 3, 1]
    unfixed_dofs = unfixed_simsopt_dofs(simsopt_dofs, unfixed_orders)
    assert unfixed_dofs == [1,2,3,4,5,6,13,14,15,16,17,18,25,26,27,28,29,30,31,32,33,34,35,36,43,44,45,46,47,48,55,56,57,58,59,60,61,62,63,64,65,66,73,74,75,76,77,78,85,86,87,88,89,90]

def test_fixed_to_list():
    order = 5
    # Testing all fixed by str
    fixed = 'all'
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    # Testing all fixed by int
    fixed = order + 1
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    fixed = order + 6
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    fixed = order
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    # Testing all fixed by list
    fixed = list(range(order+1))
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    fixed = list(range(order+6))
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    # Testing all fixed by np.ndarray
    fixed = np.array(range(order+1))
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    fixed = np.array(range(order+7))
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4, 5]
    # Testing case where no orders are fixed
    fixed = None
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == []
    fixed = []
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == []
    # Testing cases where some orders are fixed with int
    fixed = 0
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0]
    fixed = 1
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1]
    fixed = 4
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4]
    # Testing cases where some orders are fixed with list
    fixed = [0]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0]
    fixed = [0, 1]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1]
    fixed = [0, 1, 2]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2]
    fixed = [0, 1, 2, 3, 4]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4]
    fixed = [3, 5]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [3, 5]
    fixed = [1, 6]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [1]
    fixed = [6, 8, 12]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == []
    fixed = [5, 4, 0]
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 4, 5]
    # Testing cases where some orders are fixed with np.ndarray
    fixed = np.array([0])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0]
    fixed = np.array([0, 1])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1]
    fixed = np.array([0, 1, 2])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2]
    fixed = np.array([0, 1, 2, 3, 4])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 1, 2, 3, 4]
    fixed = np.array([3, 5])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [3, 5]
    fixed = np.array([1, 6])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [1]
    fixed = np.array([6, 8, 12])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == []
    fixed = np.array([5, 4, 0])
    fixed_list = fixed_to_list(fixed, order)
    assert fixed_list == [0, 4, 5]