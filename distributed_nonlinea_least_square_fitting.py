# distributed nonlinear-least square fitting per receiver to get an estimate of parameters
# each receiver only have knowledge of a small fraction of all receivers

import numpy as np
from scipy.optimize import curve_fit

from main_algorithms.distributed_gradient_descent_ATC import findClosestElements


def func(x, a, c):
    return a * x ** 2 + c ** 2


# function to retrieve picking arrival time of available receivers for one sensor
# input args:
#         k: receiver indice array
#         picking: all picking arrival time
#output args:
#       single_sensor_measurement: the picking travel time from neighbors for single receiver
def single_picking(k, picking):
    single_sensor_measurement = []
    for j in k:
        single_sensor_measurement.append(picking[j])
    return single_sensor_measurement


# function to perform fitting for single receiver
#input args:
#     line_topology_flag: flag to use line topology for sensor network,if it sets to 1
#     delta_x: space discretization level in x direction
#     receiver_number: number of receivers
#     layer_n: number of layers
#     picking: picking travel time
#     accessible_number:  number of neighbors
#     trace_number: indice of receivers on one side
#output args:
#     t0_estimate: estimated t0
#     para_estimate: estimated m0
#     accessible_sensor: number of neighbors
def fitting_sensor_networking(trace_number,line_topology_flag,delta_x, receiver_number, layer_n, picking, accessible_number):
    # estimated t0 for each receiver
    t0_estimate = []
    accessible_sensor = []
    # estimated parameter array for hyperbola
    para_estimate = []
   #loop over sensor network
    for j in range(receiver_number):
        # fit for each receiver

        # accessible receiver indices, here we use a random topology of neigbour receivers arangement
        if line_topology_flag=='random_topology':
             single_receiver_indice = np.random.randint(low=0, high=receiver_number, size=accessible_number)
        # if we switch to a line topology of neighbouring receivers in the sensor network
        if line_topology_flag=='line_topology':
        # whole receiver array indice
             whole_receiver_array_indice = np.arange(receiver_number)
        # if the receiver's communication region covers boundary
             if receiver_number-int(accessible_number/2)<j<=receiver_number-1:
                 single_receiver_indice =np.array(findClosestElements(whole_receiver_array_indice, int(receiver_number-j+accessible_number/2), j))

        # if communication region covers source
             elif j-int(accessible_number/2)<int(receiver_number/2)< j+int(accessible_number/2):
                 single_receiver_indice = np.array(findClosestElements(whole_receiver_array_indice, int(accessible_number)-1, j))

         # for other receivers we have full l_d neighbors
             elif j-int(accessible_number/2)>=int(receiver_number/2) or  j+int(accessible_number/2)<=int(receiver_number/2):
                 single_receiver_indice = np.array(findClosestElements(whole_receiver_array_indice, int(accessible_number), j))

        # append receiver indice for each receiver to construct connectivity matrix in average consensus algorithm
        accessible_sensor.append(list(single_receiver_indice - 1) + [j])
        # receiver offset
        receiver_array = (trace_number[single_receiver_indice]+1) * np.array(delta_x)

        # sensor measurement
        single_sensor_measurement = single_picking(np.array(single_receiver_indice), picking)
        # reshape for processing
        number_ele = len(np.array(single_sensor_measurement).flatten())

        single_sensor_measurement = np.array(single_sensor_measurement).T

        # estimate t0 via regression
        for j in range(layer_n):
            popt, pcov = curve_fit(func, receiver_array, single_sensor_measurement[j] ** 2)
            #    para,t0=t_0estimator(receiver_array,single_sensor_measurement[j])

            t0_estimate.append(abs(popt[1]))
            # append parameter estimate

            para_estimate.append(abs(np.array(popt[0])))
    # reshape parameter estimate for processing
    para_estimate = np.array(para_estimate).reshape([receiver_number, layer_n]).T

    t0_estimate = np.array(t0_estimate).reshape([receiver_number, layer_n])
    return t0_estimate, para_estimate, accessible_sensor









