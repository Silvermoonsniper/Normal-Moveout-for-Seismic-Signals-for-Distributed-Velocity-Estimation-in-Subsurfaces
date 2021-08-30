#####  Implement a distributed gradient descent approach to find optimum parameters govern travel time normal moveout equation
# local cost function :
##  c(m,t_0)=(|t_{i,j}-\hat t_{i,j}|^2)


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def func(x, a, c):
    return a * x ** 2 + c ** 2


# ground truth t0
def t0_solver(test_depth, layer_velocity):
    t0ground = []
    for a in range(len(test_depth) - 1):
        if a == 0:
            t0ground.append(2 * (test_depth[a + 1]) / layer_velocity[a])
        else:
            t0ground.append((2 * (test_depth[a + 1] - test_depth[a]) / layer_velocity[a]))

    t0d = []
    for k in range(len(t0ground)):
        t0d.append(np.array(t0ground[0:k + 1]).sum())

    return t0d


# generate synthesized travel time
#input args:
#         layer_number: number of layers
#         receiver_offset: distance of each receiver to shot source (m)
#output args:
#        travel_time: generate synthesized travel time at receivers
def generate_sysnthetic_traveltime(layer_number, receiver_offset):
    travel_time = []
    # ground truth parameter
    ground_truth_m0, ground_truth_t0 = [5.5, 11.8, 17.5], [5.8, 14.5, 22.8]
    # AWGN for noisy syntheic travel time data
    # stadard deviation
    std = 3
    noise = std * np.random.randn(len(receiver_offset))
    for j in range(layer_number):
        t = np.sqrt(abs(ground_truth_m0[j] * receiver_offset ** 2 + ground_truth_t0[j] ** 2)) + noise
        travel_time.append(t)
    return travel_time

#find given number of cloest neighbour indices to from a line topology
#in distributed algorithm setting
def findClosestElements( arr, k, x):
    return sorted(sorted(arr, key=lambda p: abs(p - x))[:k])
# form neighbours for each receiver and generate available measured travel time for each receiver
#         line_topology_flag: flag to use line topology in distributed setting
#         receiver_offset: receiver distance to source
#         travel_time: picking travel time at receivers
#         neighbour_number: number of neighbours
#output args:
#         final_local_traveltime: the picking travel time at each receiver gathered from neighbors
#         receiver_distance: receiver distance to source
#         neighbour_numberarray: array to store number of neighbours for each receiver
#         neighbourindice_array: array to store indice of neighbours for each receiver
def neighbours_generator(line_topology_flag,receiver_offset, travel_time,neighbour_number):
    # find neghbours for each receiver
    final_local_traveltime = []
    # reciever offset array per receiver
    receiver_distance = []
    # array to store number of neighbours for all receivers
    neighbour_numberarray = []
    # indice array for neighbours
    neighbourindice_array = []
    for j in range(len(receiver_offset)):
        # define number of neighbours with respect to each single receiver
        # indice of neighbours for jth receiver
        #if we consider a line topology of receiver constellation
        #whole receiver array indice
        if line_topology_flag=='line_topology':
            whole_receiver_array_indice=np.arange(len(receiver_offset))
            #indice of neighbors for each receiver

            # if the receiver is not the last receiver
            if j != len(receiver_offset) - 1:
                ind_pos=findClosestElements(whole_receiver_array_indice,neighbour_number, j)
            # last receiver only has connection of half number of neighbours compared with receiver in between
            else:
                ind_pos = findClosestElements(whole_receiver_array_indice, int(neighbour_number/2), j)
            # the first and second receivers only have connection with (n/2+1) neighbors (n: total number of neighbor)
            #    , due to communication radius
            if j <= 1:
                single_receiver_indice = np.array(
                    findClosestElements(whole_receiver_array_indice, int(neighbour_number / 2) + 1, j))


        elif line_topology_flag=='random_topology':
            ind_pos = np.random.randint(0, len(receiver_offset) , size=neighbour_number)
        # append neighbour number
        neighbour_numberarray.append(neighbour_number)
        # extract synthesized travel time from its neighbours
        local_traveltime = []
        # extract receiver offset of corresponding neighbours
        receiver_distance.append(receiver_offset[ind_pos])
        # append array for indice of neighbours for each receiver
        neighbourindice_array.append(np.array(ind_pos))
        for k in travel_time:
            local_traveltime.append(k[ind_pos])

        final_local_traveltime.append(local_traveltime)

    return final_local_traveltime, receiver_distance, neighbour_numberarray, neighbourindice_array


# function to construct weight matrix
#input args:
#      neighbourindice_array: array to store indice of neighbours for each receiver
#      local_gradient_error_m0：local gradient of cost function with respect to m0
#output args:
#      weight: weight matrix associated with gradient exchange (adapt stage)

def weight_matrix(neighbourindice_array, local_gradient_error_m0):
    # construct weight matrix
    weight = np.zeros([len(local_gradient_error_m0), len(local_gradient_error_m0)])
    for p in range(len(local_gradient_error_m0)):
        for j in range(len(local_gradient_error_m0)):
            if j in neighbourindice_array[p]:
                # assign weight assocaited with neighbour
                # if j!=p:

                weight[p][j] = 1 / (len(neighbourindice_array[p]))
        if np.sum(weight[p]) != 1:
            for l in range(len(local_gradient_error_m0)):
                weight[p][l] = 1 - np.sum(weight[p])
                l = len(local_gradient_error_m0)

            # assign weight associate with receiver itself
            # weight[p][j]=1- np.sum(weight[p][:])
    return weight


# function to calculate second order derivative at given time step with resepect to parametes and check Hessian matrix determinant
#input args:
#           local_m0: local estimate m0 at step k
#           local_t0:local estimate t0 at step k
#           local_traveltime: measured travel time
#           x_i: receiver position array
#output args:
#          t0_secondorder_derivative: second order derivative of t0
#          determinant: determinant of Hessian matrix
#          m0_secondorder_derivative: second order derivative of m0
def second_derivative_t0(local_m0, local_t0, local_traveltime, x_i):
    # ground truth time
    ground_truthtime = np.sqrt(abs(local_m0 * x_i ** 2 + local_t0 ** 2))
    term_1 = local_traveltime * ground_truthtime
    term_2 = local_t0 ** 2 * local_traveltime * ground_truthtime ** 3
    # calcualte 2nd order t0 derivative
    t0_secondorder_derivative = 2 * (1 - term_1 + term_2)
    # hessian determinant
    determinant = ground_truthtime ** 3 / (ground_truthtime ** 2 + 2 * local_t0 ** 2) - local_traveltime
    # calculate 2nd order m0 derivative
    m0_secondorder_derivative = 0.5 * (local_traveltime * x_i ** 4) / ground_truthtime ** 3
    return t0_secondorder_derivative, determinant, m0_secondorder_derivative


# function to retrieve picking arrival time of avaiable receivers for one sensor
# input k: receiver indice array
# picking: all picking arrival time
def single_picking(k, picking):
    single_sensor_measurement = []
    for j in k:
        single_sensor_measurement.append(picking[j])
    return single_sensor_measurement


# function to perform nonlinear least-square fitting for each receiver and
# get initial estimate of parameters from nonlinear least-square fitting
# input args:
#          receiver_number: number of receivers
#          picking: picked travel time
#          neighbourindice_array: array to store indice of neighbors for each receiver
#          delta_x: space discetization level in x direction
#          layer_n: number of layers
#output args:
#         t0_estimate, para_estimate: initial estimated t0 and m0 per receiver
def initial_estimate_parameters(trace_number,receiver_number, picking, neighbourindice_array,delta_x,layer_n):
    # estimated t0 for each receiver
    t0_estimate = []
    accessible_sensor = []
    # estimated parameter array for hyperbola
    para_estimate = []

    for j in range(receiver_number):
        # fit for each receiver

        # receiver offset
        receiver_array = (trace_number[neighbourindice_array[j]] + 1) * np.array(delta_x)

        # sensor measurement
        single_sensor_measurement = single_picking(np.array(neighbourindice_array[j]), picking)
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
    return t0_estimate, para_estimate


# function to get weighted combination of gradient from neighbouring receivers with respect to single receiver
def weighted_gradient(weight, local_gradient_error_m0):
    # calculated weighted combination of gradient at given time step
    weighted_gradient = np.matmul(weight, local_gradient_error_m0)
    return weighted_gradient


# function to get weighted combination of local estimate
def weighted_localestimate(weight, intermediate_parameter_m01):
    # calculated weighted combination of gradient at given time step
    weighted_gradient1 = np.matmul(weight, intermediate_parameter_m01)
    return weighted_gradient1



#function to implement adapt-then-combine diffusion scheme for distributed Normal-moveout
#for single layer
# input args:
#       initial_m0: initial guess of m0
#       initial_t0: initial guess of t0
#       time_step: time step to perform SGD
#       final_local_traveltime: availiable travel time data per receiver
#       yita: learning rate for SGD
#       SNR_noisy_link: SNR associated with noisy link
#       noisy_link: flag to indicate if its noisy link
#       neighbourindice_array: array to store indice of neighbors for each receiver
#       plot_flag: flag to plot
#       receiver_distance: distance of each receiver to source
#output args:
#       t0: estimated t0 at all iterations for all receivers
#       m0_new：estimated m0 at all iterations for all receivers
#       loss：global cost function at all time steps
#       determinant_array：determinant of Hessian matrix at all iterations
def SGD_optimizer(SNR_noisy_link,noisy_link,  neighbourindice_array, plot_flag, initial_m0,
                  initial_t0, time_step, final_local_traveltime, yita, receiver_distance):
    # estimated t0 and m0 array per sensor
    receiver_offset = receiver_distance
    m0 = np.zeros([time_step, len(receiver_offset)])
    m0_new = np.zeros([time_step, len(receiver_offset)])
    t0 = np.zeros([time_step, len(receiver_offset)])
    # loop over all receivers
    #    m0[j][k] denotes m0 estimate at jth iteration for kth receiver
    # set initial estimated t0 and m0
    m0_new[0] = initial_m0 * np.ones([len(receiver_offset)])
    m0[0] = initial_m0 * np.ones([len(receiver_offset)])
    t0[0] = initial_t0 * np.ones([len(receiver_offset)])
    # average of initial estimate
    t0_average = t0[0].mean()
    # loss array
    loss = np.zeros([time_step, 1])
    # initialize array for local gradient with respect to m0 and t0 for each receiver at given iteration
    local_gradient_error_m0 = np.zeros([time_step, len(receiver_offset)])
    gradient_error_t0 = np.zeros([time_step, len(receiver_offset)])
    # get weight matrix in distributed network
    weight = weight_matrix(neighbourindice_array, gradient_error_t0[0])
    # initial momentum value
    momentum_term = 0
    momentum_termt0 = 0
    #array to store determinant of Hessian matrix at each iteration
    determinant_array=[]
    # firstly, we calculate gradient of each individual cost function with respect to parameters and then apply adapt-then-combine
    # strategy
    for j in range(time_step - 1):

        # calculate initial error function

        # estimated travel time from estimated paramters
        estimated_travel_time = np.sqrt(abs(m0[0] * np.array(receiver_distance) ** 2 + np.array(t0[0]) ** 2))
        # subtract by synthesized travel time
        loss[0] = np.linalg.norm(final_local_traveltime - estimated_travel_time) ** 2
        # perform SGD update recursively
        for k in range(len(receiver_offset)):
            # calculate gradient of error function with respect to t0 and m0
            numerator = (np.sqrt(abs(m0[j][k] * receiver_distance[k] ** 2 + t0[j][k] ** 2)) - final_local_traveltime[
                k]) * receiver_distance[k] ** 2

            # calculate denominator expression in gradient formula with respect to m0
            denominator = np.sqrt(abs(m0[j][k] * receiver_distance[k] ** 2 + t0[j][k] ** 2))

            # calculate gradient with respect to m0

            denominator_norm = numerator / (1 * denominator)
            local_gradient_error_m0[j][k] = np.array(
                (1 - final_local_traveltime[k] / denominator) * receiver_distance[k] ** 2)

            # with respect to t0
            inter_term = (1 - final_local_traveltime[k] / denominator) * (2 * t0[j][k])
            gradient_error_t0[j][k] = np.array(inter_term)
        # calculate second order derivative of t0
        t0_hessian, determinant, m0_secondorder_derivative = second_derivative_t0(m0[j], t0[j], final_local_traveltime,receiver_distance)
        #append determinant of Hessian matrix
        determinant_array.append(determinant[1])

        # check convexity
        for l in range(len(t0_hessian)):
            if t0_hessian[l] <= 0 or determinant[l] <= 0 or m0_secondorder_derivative[l] <= 0:
                convexity = 'non-convex'
        # add momentum term into SGD optimization
        if j < 0:
            # momentum coefficient
            momentum_constant = 0.8

            momentum_term = momentum_constant * momentum_term + yita * local_gradient_error_m0[j]
            momentum_termt0 = momentum_constant * momentum_termt0 + yita * gradient_error_t0[j]
            intermediate_parameter_m0 = m0[j] - weighted_gradient(weight, momentum_term)
            intermediate_parameter_t0 = t0[j] - weighted_gradient(weight, momentum_termt0)

        # adapt stage
        else:
            # calculate standard deviation associated with gradient noise for m and t0 with SNR
            if j==0:
               std_gradient = np.sqrt(np.var(local_gradient_error_m0[j]))/10**(SNR_noisy_link/20)
               std_t0=np.sqrt(np.var(gradient_error_t0[j]))/10**(SNR_noisy_link/20)

               gradient_noise = std_gradient*np.random.randn(len(local_gradient_error_m0[j]))
               t0_noise = std_t0 * np.random.randn(len(local_gradient_error_m0[j]))
            # diminishing step size
            #yita=yita/(1+np.exp(j))
            # if we consider noisy link
            if noisy_link == 0:
                intermediate_parameter_m0 = m0[j] - yita * weighted_gradient(weight, local_gradient_error_m0[j])
                intermediate_parameter_t0 = t0[j] - yita * weighted_gradient(weight, gradient_error_t0[j])
            # if we consider noiseless link
            else:
                intermediate_parameter_m0 = m0[j] - yita * weighted_gradient(weight, local_gradient_error_m0[
                    j]) + gradient_noise
                intermediate_parameter_t0 = t0[j] - yita * weighted_gradient(weight,
                                                                             gradient_error_t0[j]) + t0_noise
    # combine stage

                # calculate standard deviation associated with noise in exchanging local estimate at receivers
        if j == 0:
           std_m0_local_estimate = np.sqrt(np.var(intermediate_parameter_m0)) / 10 ** (SNR_noisy_link / 20)
           std_t0_local_estimate = np.sqrt(np.var(intermediate_parameter_t0)) / 10 ** (SNR_noisy_link / 20)

        # if we consider noisy link
        if noisy_link == 1:
            m0[j + 1] = weighted_localestimate(weight, intermediate_parameter_m0) + std_m0_local_estimate*np.random.randn(len(local_gradient_error_m0[j]))
            m0_new[j + 1] = m0[j + 1]
            t0[j + 1] = weighted_localestimate(weight, intermediate_parameter_t0) + std_t0_local_estimate*np.random.randn(len(local_gradient_error_m0[j]))
        else:
            m0[j + 1] = weighted_localestimate(weight, intermediate_parameter_m0)
            m0_new[j + 1] = m0[j + 1]
            t0[j + 1] = weighted_localestimate(weight, intermediate_parameter_t0)
        # calculate local error function
        # estimated travel time from estimated paramters
        estimated_travel_time = np.sqrt(abs(m0[j + 1] * receiver_distance ** 2 + t0[j + 1] ** 2))
        # subtract by synthesized travel time
        loss[j + 1] = np.linalg.norm(final_local_traveltime - estimated_travel_time) ** 2

    # plot loss variation with different iterations

    if plot_flag == 1:

        for l in range(len(receiver_offset)):
            iteration = np.arange(time_step)
            plt.plot(iteration, np.array(loss.T[l]))

            plt.xlabel('Iteration')
            plt.ylabel('global cost $c_g$')
            plt.title('global cost $c_g$ with weight updating')


    # return final estimated t0 m0 and loss array per receiver
    return t0[10], m0_new[10], t0, m0_new, loss,determinant_array

#adapt-then-combine to get optimal estimated parameters at each layer
#input args:
#         para_estimate: estimated slowness m0 for each layer
#         t0_estimate: estimated t0 for each layer
#         SNR_noisy_link: SNR for noisy link
#         noisy_link: flag to indicate if it is noisy link
#         neighbourindice_array: indice of neighbors for each receiver
#         plot_flag: flag to plot global cost function variation
#         time_step: time step for running adapt-then-combine framework
#         travel_time: picking travel time at receivers
#         yita: learning rate
#         layer_n: number of layers
#         receiver_distance: distance between receivers and shot source.
#output args:
#         t0_array: estimated t0 for each layer per sensor per iteration
#         m0_array: estimated m0 for each layer per sensor per iteration
#         determinant_array: determinant of Hessian at all iterations
def adapt_then_combine_optimizer(layer_n,para_estimate,t0_estimate,SNR_noisy_link, noisy_link,
                                                                            neighbourindice_array, plot_flag,
                                                                            time_step, travel_time, yita,
                                                                            receiver_distance):
    # array to store estimated m0 and t0 per layer
    t0_array, m0_array = [], []
    # do adapt-then-combine to estimate layer velocity and depth for each layer
    for k in range(layer_n):
        ## initial estimate of t0 and m0
        initial_m0 = 1 * para_estimate[k]
        # calculate initial estimate of t0 based on initial estimate of m0
        initial_t0 = 1 * t0_estimate.T[k]
        # apply SGD optimizer with adapt-then-combine strategy
        final_t0, final_m0, t0, m0, loss, determinant_array = SGD_optimizer(SNR_noisy_link, noisy_link,
                                                                            neighbourindice_array, plot_flag,
                                                                            initial_m0, initial_t0,
                                                                            time_step, travel_time[k], yita,
                                                                            receiver_distance)
        # append for estimated parameter for each layer
        t0_array.append(abs(t0))
        m0_array.append(abs(m0))
    return t0_array,m0_array,determinant_array
