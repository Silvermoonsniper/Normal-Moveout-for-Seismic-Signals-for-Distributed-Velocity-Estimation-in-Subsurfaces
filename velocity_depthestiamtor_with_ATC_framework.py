# function to retieval estimate at given iteration

import numpy as np
import  matplotlib.pyplot as plt

# function to retieval estimate at given iteration
# input args:
            #    iteration: iteration we want to visualize
            #    t0_array: estimated t0 for all iterations
            #    m0_array； estimated m0 for all iterations
#output args:
#          local_t0: estimated parameter two-way vertical travel time t0 at particular time step
#          local_m0: estimated parameter slowness m0 at particular time step
import estimated_parameters_distributed_ATC
from ATC_estimation_performance_plot import estimation_ATC_performance
from local_error_local_picking import local_error_calculator


def local_estimation(iteration, t0_array, m0_array):
    local_t0 = []
    local_m0 = []
    for j in t0_array:
        a = 0
        for l in j:
            if a == iteration:
                local_t0.append(l)
            a += 1
    for k in m0_array:
        b = 0
        for p in k:
            if b == iteration:
                local_m0.append(p)

            b += 1
    return local_t0, local_m0


### function to solve layer velocity and depth with consensused parameter after distributed-gradient descent with adapt-then-combine algorithm
# layer velocity and depth for each layer
# input args:
#        receiver_offset: distance of receivers to source
#        t0_array: estimated t0 at all iterations per sensor
#        m0_array: estimated m0 at all iterations per sensor
#        layer_number: number of layers
#        layer_velocity: layer velocity
#        test_depth: depth of each layer calculated from ground
#output args:
#         v_layer: estimated layer velocity per sensor
#         ground_depth: estimated layer depth per sensor


def layer_velocity_depth_solver(receiver_offset, t0_array, m0_array, layer_number, layer_velocity, test_depth):
    #estimated root mean square velocity from estimated sloness
    vel = np.sqrt(1 / np.array(m0_array))
    # estimated two-way vertical travel time t0
    t0coff = t0_array

    # solve for velocity at each layer
    v_layer = []

    for r in range(len(vel)):
        for p in range(len(vel[r])):
            if r == 0:
                v_layer.append(vel[0][p])
            else:
                v_layer.append(np.sqrt(abs(
                    (vel[r][p] ** 2 * np.array(t0coff[r][p]) - vel[r - 1][p] ** 2 * np.array(t0coff[r - 1][p])) / (
                                np.array(t0coff[r][p]) - np.array(t0coff[r - 1][p])))))
    # solve for estimated oneway travel time
    oneway_estimate = []
    for j in range(len(t0coff)):
        if j == 0:
            oneway_estimate.append((np.array(t0coff[j])) / 2)
        else:
            oneway_estimate.append((np.array(t0coff[j]) - np.array(t0coff[j - 1])) / 2)
    # reshape for processing
    v_layer = np.array(v_layer).reshape(layer_number, len(t0_array[0]))
    # deal with special case
    a = 0

    for j in v_layer:

        if (np.array(j)).mean() - layer_velocity[a] > 1e3:
            v_layer[a] = abs(v_layer[a] - (np.array(j[-1]) - layer_velocity[a]))
        a += 1
    # solve depth for each layer
    depth = []
    for j in range(len(v_layer)):
        depth.append(v_layer[j] * oneway_estimate[j])
    # calculate depth from ground
    ground_depth = []
    ground_depthval = np.zeros([1, len(depth[0])])
    for j in range(len(depth)):
        ground_depthval = ground_depthval + depth[j]
        ground_depth.append(ground_depthval)

        #return estimated layer velocity and depth
    return v_layer,ground_depth

#function to solve layer velocity and depth from estimated parameter in ATC-framework
# for certain iterations
#input args:
#           iteration: iterations that we want to plot for local estimation performance of ATC
#           t0_array: estimated t0 at all iterations
#           m0_array: estimated m0 at all iterations
#           receiver_distance: distance of receivers to the source for one side
#           layer_n: number of layers
#           layer_velocity: wave velocity of each layer
#           test_depth: depth of each layer calculated from ground surface
#           time_step: total time step for running SGD
#output args:
#           local_error:local depth estimation error for certain iterations
#           local_velocity: local velocity estimation error for certain iterations

def vel_depth_solver_ATC_iterations(velocity_plot,time_step,iteration,t0_array, m0_array,receiver_distance, layer_n, layer_velocity, test_depth):
    # array to store local estimation error in velocity and depth estimation
    local_error, local_velocity = [], []
    for j in range(time_step):
        # get local estimated parameters at given iteration
        local_t0, local_m0 = estimated_parameters_distributed_ATC.local_estimation(j, t0_array, m0_array)
        # estimated layer velocity and depth
        v_layer, ground_depth = layer_velocity_depth_solver(
            receiver_distance, local_t0, local_m0, layer_n, layer_velocity, test_depth)
        #visualize estimation result at certain iterations
        if j in iteration:
            estimation_ATC_performance(j,velocity_plot,v_layer, receiver_distance, layer_velocity, ground_depth, test_depth)
        # calculate estimation error
        local_error_depth, local_error_velocity = local_error_calculator(layer_velocity, test_depth, ground_depth,
                                                                          v_layer)
        local_error.append(local_error_depth)
        local_velocity.append(local_error_velocity)
    return local_error,local_velocity



