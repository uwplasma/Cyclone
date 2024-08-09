import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp

def set_JF(JF_):
    '''
    This function imports the objective function object used
    in the optimization loop as a global variable for use
    with the set of find_error functions in this module to
    allow for a finite difference approach.
    Input: JF (objective function object)
    Output: None
    '''
    global JF
    JF = JF_
    return None

def set_coil_parameters(ntoroidalcoils_, npoloidalcoils_, nfp_, stellsym_, unique_shapes_, winding_surface_function_, order_, curves_, num_currents_):
    '''
    This function imports the parameters used in the
    create_multiple_arbitrary_windowpanes function from
    create_windowpanes.py into this module as global
    variables for use with the set of find_error functions
    in this module. Additionally, it sends these parameters
    to the optimization_functions.py module for use there.
    Input: ntoroidalcoils, npoloidalcoils, nfp, stellsym, unique_shapes,
           winding_surface_function, order, curves, num_currents
    Output: None
    '''
    from optimization_functions import set_opt_coil_parameters
    global ntoroidalcoils, npoloidalcoils, nfp, stellsym, unique_shapes, winding_surface_function, order, curves, num_currents
    ntoroidalcoils = ntoroidalcoils_
    npoloidalcoils = npoloidalcoils_
    nfp = nfp_
    stellsym = stellsym_
    unique_shapes = unique_shapes_
    winding_surface_function = winding_surface_function_
    order = order_
    curves = curves_
    num_currents = num_currents_
    set_opt_coil_parameters(ntoroidalcoils_, npoloidalcoils_, nfp_, stellsym_, unique_shapes_, winding_surface_function_, order_, curves_, num_currents_)
    return None

def print_if(printout, statement):
    '''
    This function prints a statement if the input
    printout statement evaluates as true.
    Input: printout (binary statement), statement (string)
    Output: None
    Prints: Statement if printout
    '''
    if printout:
        print(statement)
    return None

def plot_if(graph, plot_obj):
    '''
    This function plots a pyplot graph if the 
    input graph statement evaluates as true.
    Input: graph (binary statement), plot_obj (pyplot object)
    Output: None
    Plots: plot_obj if graph
    '''
    if graph:
        plot_obj.show()
    return None

def save_plot(filename = 'plot', form = '.pdf', out_dir = './plot_output/'):
    '''
    This function saves the current working pyplot graph
    to the input output directory with the input file name
    and the input file extension.
    Input: filename (string), form (extension type),
           out_dir (desired directory)
    Output: None
    Saves: current working pyplot graph
    '''
    os.makedirs(out_dir, exist_ok=True)
    save_as = out_dir + filename + form
    count = 0
    while os.path.exists(save_as):
        save_as = out_dir + filename + '{}'.format(count) + form
        count = count + 1
    plt.savefig(save_as)
    return None

def save_plot_if(save, filename = 'plot', form = '.pdf', out_dir = './plot_output/'):
    '''
    This function saves a pyplot graph if the
    input save statement evaluates as true.
    It calls the save_plot function to accomplish this.
    Input: save (binary statement), filename (string),
           form (extension type), out_dir (desired directory)
    Output: None
    Saves: current working pyplot graph if save
    '''
    if save:
        save_plot(filename = filename, form = form, out_dir = out_dir)
    return None

def set_scale_epsilon(init_indexed, printout):
    '''
    This function creates a finite difference amount
    scaled to the desired number, input by user. It
    prints this finite difference if printout evaluates true.
    Input: init_indexed (float), printout (binary statement)
    Output first_epsilon (init_indexed / 10)
    '''
    first_epsilon = abs(init_indexed) / 10
    print_if(printout, "Initial epsilon set to {}".format(first_epsilon))
    if first_epsilon == 0:
        first_epsilon = 0.1
        print_if(printout, "No value to scale (entry = 0), switched to initial epsilon of 0.1")
    return first_epsilon

def find_error_mult_epsilon(index, init, first_epsilon = 0.1, num_epsilon = 5, factor = 10, printout = True, graph = True, save = False, dj_simsopt = None):
    '''
    This function evaluates the Simsopt given gradient of JF (objective function)
    at initial parameters ''init'' if ''dj_simsopt'' is None and otherwise takes
    ''dj_simsopt'' to be the Simsopt given gradient of JF.
    It then evaluates the finite difference gradient of a single index of ''init''
    (the ''index'' index of the array) for finite differences beginning at
    ''first_epsilon'' (or ''init[index] / 10'' if first_epsilon = 'Scaled' (0.1 if 
    ''init[index]'' = 0)) and for a total number of finite differences
    ''num_epsilon'' where each subsequent finite difference is a factor of
    ''factor'' smaller than the previous. If ''printout'' is True, this function will
    print out information about the ongoing validation. If ''graph'' is True, this
    function will plot the determined data in a number of formats. If save is true,
    this function will save the aforementioned plots.
    Lastly, the function returns a vector of the relative errors at each finite difference.
    Input: index (int), init (array), first_epsilon (float or 'Scale'),
           num_epsilon (int), factor (float), printout (binary),
           graph (binary), save (binary), dj_simsopt (array or None)
    Output: error_list (array), grad_list (array)
    '''
    if first_epsilon == "Scale":
        first_epsilon = set_scale_epsilon(init[index], printout)
    error_list = jnp.array([])
    grad_list = jnp.array([])
    JF.x = init
    J_init = JF.J()
    if dj_simsopt is None:
        dj_simsopt = JF.dJ()
    dj = dj_simsopt[index]
    new = list(init)
    print_if(printout, "J initially is:\n{}\nAnd Simsopt derivative is:\n{}".format(J_init, dj))
    for i in range(num_epsilon):
        epsilon = first_epsilon / (factor ** i)
        new[index] = init[index] + epsilon
        JF.x = new
        J = JF.J()
        approx_derv = (J - J_init) / epsilon
        error = abs((dj - approx_derv) / dj) * 100
        error_list = jnp.append(error_list, error)
        grad_list = jnp.append(grad_list, approx_derv)
        print_if((printout) and (not i == 0) and (abs(error_list[i]) > abs(error_list[i-1])), 
                "\n########\n########\n########\nTaylor test fails\n########\n########\n########")
        print_if(printout, "\n########\nFor epsilon of {}:\nJ is:\n{}\nApproximated derivative is:\n{}\nAnd the error \
                 (assuming Simsopt as truth) is:\n{} %\n########\n########".format(epsilon, J, approx_derv, error))
    if graph or save:
        f = plt.figure(1)
        plt.plot(range(num_epsilon), grad_list)
        plt.axhline(y=dj, color='r', linestyle='--')
        plt.text(0,np.mean(grad_list),'Simsopt Derivative', color='r')
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('Approximated derivative')
        plt.title('(DIRECT) Convergence of approximated derivatives for DOF {}'.format(index))
        plot_if(graph, f)
        save_plot_if(save, filename = 'Direct_grad')

        g = plt.figure(2)
        plt.plot(range(num_epsilon), grad_list)
        plt.axhline(y=dj, color='r', linestyle='--')
        plt.text(0,np.mean(grad_list),'Simsopt Derivative', color='r')
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('LOG Approximated derivative')
        plt.title('(DIRECT) Convergence of approximated derivatives for DOF {}'.format(index))
        plt.yscale('log')
        plot_if(graph, g)
        save_plot_if(save, filename = 'Direct_grad_log')

        epsilon_list = [first_epsilon / (factor ** i) for i in range(num_epsilon)]

        h = plt.figure(3)
        plt.plot(epsilon_list, error_list, marker='o')
        plt.xlabel('Epsilon'.format(first_epsilon, factor))
        plt.ylabel('Relative error (%)')
        plt.title('(DIRECT) Convergence of relative error for DOF {}'.format(index))
        plt.text(0.8*np.max(epsilon_list), 0.9*np.max(error_list), 'Minimum error is {} %'.format(np.min(error_list)))
        plt.gca().invert_xaxis()
        plot_if(graph, h)
        save_plot_if(save, filename = 'Direct_error')

        q = plt.figure(4)
        plt.plot(epsilon_list, error_list, marker='o')
        plt.xlabel('Epsilon (LOG)'.format(first_epsilon, factor))
        plt.ylabel('Relative error (%)')
        plt.title('(DIRECT) Convergence of relative error for DOF {}'.format(index))
        plt.text(np.mean(epsilon_list), 0.9*np.max(error_list), 'Minimum error is {} %'.format(np.min(error_list)))
        plt.gca().invert_xaxis()
        plt.xscale('log')
        plot_if(graph, q)
        save_plot_if(save, filename = 'Direct_error_log')
    return error_list, grad_list

def find_planar_error_mult_epsilon(index, init_dofs, first_epsilon = 0.1, num_epsilon = 5, factor = 10, printout = True, graph = True, save = False, grad = None):
    '''
    This function evaluates the Simsopt given gradient of JF (objective function)
    at initial parameters determined by the initial planar parameters ''init_dofs''
    and transforms this into the gradient of JF with respect to the planar parameters
    via the jax determined jacobian of the transformation between these sets of
    parameters if ''grad'' is None and otherwise takes ''grad'' to be this gradient of JF.
    It then evaluates the finite difference gradient of a single index of ''init_dofs''
    (the ''index'' index of the array) for finite differences beginning at
    ''first_epsilon'' (or ''init[index] / 10'' if first_epsilon = 'Scaled' (0.1 if 
    ''init[index]'' = 0)) and for a total number of finite differences
    ''num_epsilon'' where each subsequent finite difference is a factor of
    ''factor'' smaller than the previous. If ''printout'' is True, this function will
    print out information about the ongoing validation. If ''graph'' is True, this
    function will plot the determined data in a number of formats. If save is true,
    this function will save the aforementioned plots.
    Lastly, the function returns a vector of the relative errors at each finite difference.
    Input: index (int), init_dofs (array), first_epsilon (float or 'Scale'),
           num_epsilon (int), factor (float), printout (binary),
           graph (binary), save (binary), grad (array or None)
    Output: error_list (array), grad_list (array)
    '''
    from optimization_functions import multiple_sin_cos_components_to_xyz, multiple_change_jacobian, change_arbitrary_windowpanes
    if first_epsilon == "Scale":
        first_epsilon = set_scale_epsilon(init_dofs[index], printout)
    error_list = jnp.array([])
    grad_list = jnp.array([])
    JF.x = init_dofs[:num_currents] + list(JF.x[num_currents:])
    dofsarr = jnp.array(init_dofs[num_currents:])
    curves_xyz_dofs = multiple_sin_cos_components_to_xyz(dofsarr, ntoroidalcoils, npoloidalcoils, nfp, stellsym,
                                 unique_shapes, winding_surface_function = winding_surface_function, order=order)
    multiple_change_jacobian = multiple_change_jacobian()
    change_arbitrary_windowpanes(curves, curves_xyz_dofs)
    J_init = JF.J()
    if grad is None:
        dj_simsopt = JF.dJ()
        jacobian = multiple_change_jacobian(dofsarr, ntoroidalcoils, npoloidalcoils, nfp, stellsym,
                                     unique_shapes, winding_surface_function = winding_surface_function, order=order)
        grad = jnp.matmul(dj_simsopt[num_currents:].reshape((1,len(dj_simsopt)-num_currents)), jacobian)
        grad = jnp.append(jnp.array(dj_simsopt[:num_currents]), grad)
    dj = grad[index]
    new = list(init_dofs)
    print_if(printout, "J initially is:\n{}\nAnd Simsopt derivative is:\n{}".format(J_init, dj))
    for i in range(num_epsilon):
        epsilon = first_epsilon / (factor ** i)
        new[index] = init_dofs[index] + epsilon
        JF.x = new[:num_currents] + list(JF.x[num_currents:])
        dofsarr = jnp.array(new[num_currents:])
        curves_xyz_dofs = multiple_sin_cos_components_to_xyz(dofsarr, ntoroidalcoils, npoloidalcoils, nfp, stellsym,
                                     unique_shapes, winding_surface_function = winding_surface_function, order=order)
        change_arbitrary_windowpanes(curves, curves_xyz_dofs)
        J = JF.J()
        approx_derv = (J - J_init) / epsilon
        error = abs((dj - approx_derv) / dj) * 100
        error_list = jnp.append(error_list, error)
        grad_list = jnp.append(grad_list, approx_derv)
        print_if((printout) and (not i == 0) and (abs(error_list[i]) > abs(error_list[i-1])),
                "\n########\n########\n########\nTaylor test fails\n########\n########\n########")
        print_if(printout, "\n########\n########\nFor epsilon of {}:\nJ is:\n{}\nApproximated derivative is:\n{}\nAnd the error \
        (assuming Simsopt as truth) is:\n{} %\n########\n########".format(epsilon, J, approx_derv, error))
    if graph or save:
        f = plt.figure(1)
        plt.plot(range(num_epsilon), grad_list)
        plt.axhline(y=dj, color='r', linestyle='--')
        plt.text(0,np.mean(grad_list),'Simsopt Derivative', color='r')
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('Approximated derivative')
        plt.title('(PLANAR) Convergence of approximated derivatives for DOF {}'.format(index))
        plot_if(graph, f)
        save_plot_if(save, filename = 'Planar_convergence')

        g = plt.figure(2)
        plt.plot(range(num_epsilon), grad_list)
        plt.axhline(y=dj, color='r', linestyle='--')
        plt.text(0,np.mean(grad_list),'Simsopt Derivative', color='r')
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('LOG Approximated derivative')
        plt.title('(PLANAR) Convergence of approximated derivatives for DOF {}'.format(index))
        plt.yscale('log')
        plot_if(graph, g)
        save_plot_if(save, filename = 'Planar_convergence_log')
        
        epsilon_list = [first_epsilon / (factor ** i) for i in range(num_epsilon)]

        h = plt.figure(3)
        plt.plot(epsilon_list, error_list, marker='o')
        plt.xlabel('Epsilon'.format(first_epsilon, factor))
        plt.ylabel('Relative error (%)')
        plt.title('(PLANAR) Convergence of relative error for DOF {}'.format(index))
        plt.text(0.8*np.max(epsilon_list), 0.9*np.max(error_list), 'Minimum error is {} %'.format(np.min(error_list)))
        plt.gca().invert_xaxis()
        plot_if(graph, h)
        save_plot_if(save, filename = 'Planar_error')
        
        q = plt.figure(4)
        plt.plot(epsilon_list, error_list, marker='o')
        plt.xlabel('Epsilon (LOG)'.format(first_epsilon, factor))
        plt.ylabel('Relative error (%)')
        plt.title('(PLANAR) Convergence of relative error for DOF {}'.format(index))
        plt.text(np.mean(epsilon_list), 0.9*np.max(error_list), 'Minimum error is {} %'.format(np.min(error_list)))
        plt.gca().invert_xaxis()
        plt.xscale('log')
        plot_if(graph, q)
        save_plot_if(save, filename = 'Planar_error_log')
    return error_list, grad_list

def find_sum_error_mult_epsilon(init, first_epsilon = 0.1, num_epsilon = 5, factor = 10, printout = True, graph = True, save = False):
    '''
    This function evaluates the Simsopt given gradient of JF (objective function)
    at initial parameters ''init''.
    It then evaluates the finite difference gradient of each index of ''init''
    for finite differences beginning at ''first_epsilon'' and for a total number
    of finite differences ''num_epsilon'' where each subsequent finite
    difference is a factor of ''factor'' smaller than the previous.
    It does this via the find_error_mult_epsilon function with
    printing, graphing, and saving suppressed.
    If ''printout'' is True, this function will print out information about 
    the ongoing validation. If ''graph'' is True, this function will plot the
    determined data in a number of formats. If save is true, this function will
    save the aforementioned plots.
    Lastly, the function returns a vector of the aggregated relative errors at
    each finite difference.
    Input: init (array), first_epsilon (float), num_epsilon (int), factor (float),
           printout (binary), graph (binary), save (binary)
    Output: error_vec (array)
    '''
    assert type(first_epsilon) is not str, 'Cannot set first_epsilon = \'Scaled\' for consistent finite differences'
    error_mat = np.zeros((len(init), num_epsilon))
    grad_mat = np.zeros((len(init), num_epsilon))
    sum_grad_vec = np.zeros(num_epsilon)
    RMS_vec = np.zeros(num_epsilon)
    error_vec = np.zeros(num_epsilon)
    dj_simsopt = JF.dJ()
    sum_simsopt_dj = np.sum(dj_simsopt ** 2)
    for index in range(len(init)):
        print_if((printout) and (index % 100 == 0), '{} out of {}'.format(index, len(init)))
        error_list, grad_list = find_error_mult_epsilon(index, init, first_epsilon = first_epsilon, num_epsilon = num_epsilon, factor = factor, printout = False, graph = False, dj_simsopt = dj_simsopt)
        error_mat[index] = error_list
        grad_mat[index] = grad_list
    grad_mat_trans = grad_mat.T
    error_mat_trans = error_mat.T
    for i in range(num_epsilon):
        sum_grad_vec[i] = np.sum(grad_mat_trans[i] ** 2)
        RMS_vec[i] = np.sqrt(np.mean(error_mat_trans[i] ** 2))
    error_vec = np.abs((sum_grad_vec - sum_simsopt_dj) / sum_simsopt_dj) * 100
    if graph or save:
        f = plt.figure(1)
        plt.plot(range(num_epsilon), error_vec)
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('Relative error')
        plt.title('(DIRECT) Convergence of relative error ((sum**2)-(sum**2))/(sum**2)')
        plt.axhline(y=0, color='r', linestyle='--')
        plot_if(graph, f)
        save_plot_if(save, filename = 'Direct_sum_error')
        
        g = plt.figure(2)
        plt.plot(range(num_epsilon), error_vec)
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('LOG Relative error')
        plt.title('(DIRECT) Convergence of relative error ((sum**2)-(sum**2))/(sum**2)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.yscale('log')
        plot_if(graph, g)
        save_plot_if(save, filename = 'Direct_sum_error_log')

        h = plt.figure(3)
        for i in range(len(init)):
            plt.plot(range(num_epsilon), error_mat[i])
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('Relative error')
        plt.title('(DIRECT) Convergence of relative error for individual DOFs')
        plt.axhline(y=0, color='r', linestyle='--')
        plot_if(graph, h)
        save_plot_if(save, filename = 'Direct_individual_error')

        q = plt.figure(4)
        plt.plot(range(num_epsilon), RMS_vec)
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('RMS of individual errors')
        plt.title('(DIRECT) Convergence of relative error SQRT( MEAN( ((true - approx) / true ))^2 )')
        plt.axhline(y=0, color='r', linestyle='--')
        plot_if(graph, q)
        save_plot_if(save, filename = 'Direct_RMS_error')
    return error_vec

def find_sum_planar_error_mult_epsilon(init_dofs, first_epsilon = 0.1, num_epsilon = 5, factor = 10, printout = True, graph = True, save = True):
    '''
    This function evaluates the Simsopt given gradient of JF (objective function)
    at initial parameters determined by the initial planar parameters ''init_dofs''
    and transforms this into the gradient of JF with respect to the planar parameters
    via the jax determined jacobian of the transformation between these sets of
    parameters.
    It then evaluates the finite difference gradient of each index of ''init_dofs''
    for finite differences beginning at ''first_epsilon'' and for a total number
    of finite differences ''num_epsilon'' where each subsequent finite
    difference is a factor of ''factor'' smaller than the previous.
    It does this via the find_planar_error_mult_epsilon function with
    printing, graphing, and saving suppressed.
    If ''printout'' is True, this function will print out information about 
    the ongoing validation. If ''graph'' is True, this function will plot the
    determined data in a number of formats. If save is true, this function will
    save the aforementioned plots.
    Lastly, the function returns a vector of the aggregated relative errors at
    each finite difference.
    Input: init_dofs (array), first_epsilon (float), num_epsilon (int), factor (float),
           printout (binary), graph (binary), save (binary)
    Output: error_vec (array)
    '''
    from optimization_functions import multiple_sin_cos_components_to_xyz, multiple_change_jacobian, change_arbitrary_windowpanes
    assert type(first_epsilon) is not str, 'Cannot set first_epsilon = \'Scaled\' for consistent finite differences'
    error_mat = np.zeros((len(init_dofs), num_epsilon))
    grad_mat = np.zeros((len(init_dofs), num_epsilon))
    sum_grad_vec = np.zeros(num_epsilon)
    RMS_vec = np.zeros(num_epsilon)
    error_vec = np.zeros(num_epsilon)
    dofsarr = jnp.array(init_dofs[num_currents:])
    dj_simsopt = JF.dJ()
    multiple_change_jacobian = multiple_change_jacobian()
    jacobian = multiple_change_jacobian(dofsarr, ntoroidalcoils, npoloidalcoils, nfp, stellsym,
                                 unique_shapes, winding_surface_function = winding_surface_function, order=order)
    grad = jnp.matmul(dj_simsopt[num_currents:].reshape((1,len(dj_simsopt)-num_currents)), jacobian)
    grad = jnp.append(jnp.array(dj_simsopt[:num_currents]), grad)
    sum_grad = np.sum(grad ** 2)
    for index in range(len(init_dofs)):
        print_if((printout) and (index % 10 == 0), '{} out of {}'.format(index, len(init_dofs)))
        error_list, grad_list = find_planar_error_mult_epsilon(index, init_dofs, first_epsilon = first_epsilon, num_epsilon = num_epsilon, factor = factor, printout = False, graph = False, grad = grad)
        error_mat[index] = error_list
        grad_mat[index] = grad_list
    grad_mat_trans = grad_mat.T
    error_mat_trans = error_mat.T
    for i in range(num_epsilon):
        sum_grad_vec[i] = np.sum(grad_mat_trans[i] ** 2)
        RMS_vec[i] = np.sqrt(np.mean(error_mat_trans[i] ** 2))
    error_vec = np.abs((sum_grad_vec - sum_grad) / sum_grad) * 100
    if graph or save:
        f = plt.figure(1)
        plt.plot(range(num_epsilon), error_vec)
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('Relative error')
        plt.title('(PLANAR) Convergence of relative error ((sum**2)-(sum**2))/(sum**2)')
        plt.axhline(y=0, color='r', linestyle='--')
        plot_if(graph, f)
        save_plot_if(save, filename = 'Planar_sum_error')
        
        g = plt.figure(2)
        plt.plot(range(num_epsilon), error_vec)
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('LOG Relative error')
        plt.title('(PLANAR) Convergence of relative error ((sum**2)-(sum**2))/(sum**2)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.yscale('log')
        plot_if(graph, g)
        save_plot_if(save, filename = 'Planar_sum_error_log')
        
        h = plt.figure(3)
        for i in range(len(init_dofs)):
            plt.plot(range(num_epsilon), error_mat[i])
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('Relative error')
        plt.title('(PLANAR) Convergence of relative error for individual DOFs')
        plt.axhline(y=0, color='r', linestyle='--')
        plot_if(graph, h)
        save_plot_if(save, filename = 'Planar_individual_error')
        
        q = plt.figure(4)
        plt.plot(range(num_epsilon), RMS_vec)
        plt.xlabel('i, where epsilon = {} / ({} ** i)'.format(first_epsilon, factor))
        plt.ylabel('RMS of individual errors')
        plt.title('(PLANAR) Convergence of relative error SQRT( MEAN( ((true - approx) / true ))^2 )')
        plt.axhline(y=0, color='r', linestyle='--')
        plot_if(graph, q)
        save_plot_if(save, filename = 'Planar_RMS_error')
    return error_vec
