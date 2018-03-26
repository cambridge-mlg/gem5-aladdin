import time

import numpy as np
from aquisition.smsego import SMSego
from models.gp.gplib import GPlib
from models.random.randommodel import RandomModel
from optimizer_helper import InvalidParameters, shift, find_frontier, get_hypervolume


def optimize(f,
             candidates,
             model_params,
             aquisition_params,
             num_init,
             num_max,
             reference_point,
             output_dir=None,
             plots=False):

    print('Initializing values')

    # Points of the initial evaluations
    init_index = []
    init_values = []
    for i in range(num_init):
        index = np.random.randint(0, candidates.shape[0])
        try:
            value = f(candidates[index, :])
            init_index.append(index)
            init_values.append(value)
        except InvalidParameters:
            i -= 1	
    init_points = candidates[init_index, :]
    candidates = np.delete(candidates, init_index, 0)

    init_values = np.reshape(np.array(init_values), (len(init_values), -1))
    init_values_mean = np.mean(init_values, axis=0)
    init_values_std = np.std(init_values, axis=0) + 1e-10
    print(zip(init_points, init_values))

    # Model
    print('Initializing the Predictive Model')

    if model_params.name == 'gp':
        model = GPlib(init_points, init_values, model_params)
    elif model_params.name == 'rng':
        model = RandomModel(init_points, init_values, model_params)
    else:
        raise Exception('Ill-specified model name')

    # Aquisition
    print('Initializing the Acquisition Function')

    if aquisition_params.name == 'smsego':
        aquisition_function = SMSego(aquisition_params, reference_point, init_values_mean, init_values_std)
    else:
        raise Exception('Ill-specified aquisition function')

    frontier = find_frontier(init_values)
    hv_over_iterations = [aquisition_function.get_hypervolume(frontier, reference_point)]

    # Iteration
    print('Iterating')
    current_points = init_points.copy()

    for iter in range(0, num_max):
        print('Iteration {0}'.format(iter))

        aquisition_values = aquisition_function.getAquisitionBatch(candidates, model, frontier)
        max_aquisition_index = np.argmax(aquisition_values)
        new_point = candidates[max_aquisition_index]
        try:
            new_point_value = np.reshape(np.array(f(new_point)), (1, -1))
            current_points = np.vstack((new_point, current_points))
        except InvalidParameters:
            iter -= 1
            print('Invalid Parameters')
        finally:
            candidates = np.delete(candidates, max_aquisition_index, 0)
        
        print('   New point at {0} with value {1}, {2}'.format(new_point, new_point_value, shift(new_point_value[None, :], init_values_mean, init_values_std)))

        model_update_start = time.time()
        model.addPoint(new_point, new_point_value)
        print('   Model updated in {0}'.format(time.time() - model_update_start))

        frontier = find_frontier(np.vstack((new_point_value, frontier)))
        hv_over_iterations.append(aquisition_function.get_hypervolume(frontier, reference_point))
        print('   Hypervolume improved from {0} to {1}'.format(hv_over_iterations[-2], hv_over_iterations[-1]))

    print('The final Hypervolume is {0}'.format(hv_over_iterations[-1]))
    return frontier, np.array(hv_over_iterations)

