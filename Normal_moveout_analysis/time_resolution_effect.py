# investigate how the time discretization level influences reconstruction error of NMO
# a function to calculate reconstruction error
from main_algorithms.centralized_NMO import normal_moveout
from time_resolution import time_resolution_generator, interpolation
from travel_time_picking import arrival_time_picking
import numpy as np
import matplotlib.pyplot as plt
#input args:
#          test_depth：ground truth of layer depth
#          layer_velocity：ground truth of layer velocity
#          alpha：factor to choose time level
#          reflectivity：reflectivity series
#          time_array： time series for plot
#          newsynthetic_arriavl_1：ground truth arrival time at receievers
#          finaltime：ground truth of arrival time at receivers
#          trace_number：indice of receivers in the receiver array on one side
#          receiver_distance： distance of receivers to source
#          layer_n： number of layers
#          delta_x： space discretization level in x direction
#output args:
#         depth_error:normalized depth estimation error
#         velocity_error: normalized estimated velocity error
#         new_delta_t: time resolution
def estimation_error(test_depth,layer_velocity,alpha, reflectivity, time_array, newsynthetic_arriavl_1,finaltime,trace_number,receiver_distance,layer_n, delta_x):

    # interpolated deconvolved profile

    new_delta_t, new_time_array = time_resolution_generator(alpha, time_array[0:4000].max(), time_array[0], finaltime)
    reflectivity_inter, new_delta_t = interpolation(new_time_array, time_array, reflectivity)
    # arrival time data for NMO
    final_arrival = arrival_time_picking(reflectivity_inter, trace_number, new_delta_t, new_time_array,
                                         newsynthetic_arriavl_1)
    # set flag don't plot NMO results
    vel_flag = 10
    peak, optimal_time, ground_depth, v_layer, t0coff, t0coffop, time_level = normal_moveout(finaltime, vel_flag, final_arrival, receiver_distance, layer_velocity, test_depth, layer_n, delta_x,new_delta_t)
    # define reconstruction error array for depth and velocity reconstruction
    depth_error, velocity_error = [], []
    a = 1
    ground_depth = ground_depth[0:3]
    v_layer = v_layer[0:3]
    for j in ground_depth:
        depth_error.append(np.array(abs(j - test_depth[a])).mean() / test_depth[a])
        a += 1
    b = 0
    for k in v_layer:
        velocity_error.append(np.array(abs(k - layer_velocity[b])).mean() / layer_velocity[b])
        b += 1

    return depth_error, velocity_error, new_delta_t


# implement a function to study how different time discretization level changes reconsturction performance of NMO
def NMO_performance_simulator(reflectivity, time_array, newsynthetic_arriavl_1):
    # time discretization set
    coff = np.array(np.linspace(0.01, 1, num=10))
    # initial array to store depth and velocity reconstruction error and time level
    final_depth_error, final_vel_error, final_time_level = [], [], []
    for j in coff:
        depth_error, velocity_error, time_level = estimation_error(j, reflectivity, time_array, newsynthetic_arriavl_1)
        final_depth_error.append(depth_error)
        final_vel_error.append(velocity_error)
        final_time_level.append(time_level)
    # reshape estimation error array

    final_depth_error = np.array(final_depth_error).flatten().reshape([10, 3]).T

    final_vel_error = np.array(final_vel_error).flatten().reshape([10, 3]).T
    final_time_level = final_time_level * np.array(1e6)
    # visualize NMO performance with different time level
    vel = 0
    if vel == 0:
        for j in range(3):
            plt.plot(final_time_level, final_depth_error[j])
            plt.xlabel('Time Discretization Level $\delta t$ (us)')
            plt.ylabel('Depth reconstruction error $e_d$')
            plt.title('NMO depth reconstruction error performance analysis')
            plt.legend(['Layer 1', 'Layer 2', 'Layer 3'], loc='best')
    else:
        for j in range(3):
            plt.plot(final_time_level, final_vel_error[j])
            plt.xlabel('Time Discretization Level $\delta t$ (us)')
            plt.ylabel('Layer velocity reconstruction error $e_d$')
            plt.title('NMO velocity reconstruction error performance analysis')
            plt.legend(['Layer 1', 'Layer 2', 'Layer 3'], loc='best')

