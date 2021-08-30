
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from local_error_local_picking import *
import math
def func(x, a, c):
    return a * x ** 2 + c ** 2
#noisy travel time data generator with noisy link
#input args:
#        peak: picking travel time per sensor
#        SNR_noisy_link: SNR for noisy link
#        receiver_number: number of receivers on one side
#output args:
#       noisy_peak: noisy picking time after transmission
#       DEVIATION_array: standard deviation on noisy picking time per sensor
def noisy_picking(peak, SNR_noisy_link, receiver_number):
    peak = np.array(peak).T
    noisy_peak = []
    DEVIATION_array=[]
    for j in peak:
        #calculate standard deviation based on SNR
        DEVIATION = np.sqrt( np.var(peak))/10**(SNR_noisy_link/20)
        # noisy picking
        noisy_picking = np.add(j, np.sqrt(abs(DEVIATION)) * np.random.randn(len(j), 1))
        # append noisy picking and std array
        noisy_peak.append(noisy_picking[0][0:receiver_number])
        DEVIATION_array.append(DEVIATION)

    return noisy_peak, DEVIATION_array
# define root-mean square velocity solver
fig = plt.figure(figsize=(6, 6))


def rms_velocity(t0d, layer_velocity):
    initial_time = 0.5 * np.array(t0d)
    oneway = []

    for i in range(len(initial_time)):
        if i > 0:
            oneway.append(initial_time[i] - initial_time[i - 1])
        else:
            oneway.append(initial_time[i])
    oneway = np.array(oneway)

    v_rms = []
    k = 0
    nu = 0
    bu = 0
    for j in range(len(np.array(t0d))):
        while k <= j:
            nu = nu + layer_velocity[j] ** 2 * oneway[j]
            bu = bu + oneway[j]
            k += 1
        val = np.sqrt(abs(nu / bu))

        v_rms.append(val)
    return v_rms, oneway


# ground truth t0
# input args:
#          test_depth: ground truth layer depth
#          layer_velocity: ground truth layer velocity
#ouput args:
#          t0d: ground truth t0
def t0_solver(test_depth, layer_velocity):
    t0ground = []
    for a in range(len(test_depth) - 1):
        if a == 0:
            t0ground.append(2 * (test_depth[a + 1]) / layer_velocity[a])
        else:
            t0ground.append((2 * (test_depth[a + 1] - test_depth[a]) / layer_velocity[a]))

    t0d = []
    for k in range(len(t0ground)):
        t0d.append(np.array(t0ground[0:k]).sum())

    return t0d


# t0 solver
def t0_withreceiver(offset, peak):
    # estimate t0 from receiver measurement
    t0coff = []
    parameter = []
    for j in range(len(peak)):
        popt, pcov = curve_fit(func, np.array(offset[j]).flatten(), np.array(peak[j]).flatten() ** 2)
        t0coff.append(popt[1])
        parameter.append(popt[0])
    return t0coff, parameter


# velocity and depth estimator with NMO
#estimate layer velocity and depth from NMO
#input args:
#          receiver_distance: receiver distance to source
#          peak: picking arrival time of reflections at receivers
#          layer_velocity: velocity of each layer
#          test_depth: depth from ground for each layer
#          receiver_number: number of receivers
#          consensus_t0: final consensus estimated t0 for each layer after average consensus
#          central_flag: flag to perform centralized NMO
#          t0d:ground truth of t0
#          optimal_flag: flag to perform optimum estimation with perfect picking and estimated m0 and t0
#output args:
#          ground_depth: estimated layer depth (from ground surface)
#          v_layer: estimated layer velocity
#          t0coff: estimated t0
def vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity, offset, peak, receiver_number, consensus_t0,
                        central_flag, optimal_flag):
    # estimate t0 from receiver measurement
    # if apply nmo in centralized manner

    if central_flag == 1:
        #estimated parameters via nonlinear least-square fitting
        t0coff, para = t0_withreceiver(offset, peak)
        if pattern_analzye_flag==0 and osicillation_pattern==0:
            # recalculate peak in centralized case
            peak=re_picking_arrival(t0coff,para,receiver_distance)
    # if apply nmo in distributed manner with final consensus t0 and m0

    elif central_flag == 0:
        t0coff = consensus_t0
    # apply nmo distributedly with local estimate t0 and m0
    elif central_flag == 2:
        t0coff = consensus_t0

    t0ground = []

    for a in range(len(test_depth) - 1):
        if a == 0:
            t0ground.append(2 * (test_depth[a + 1]) / layer_velocity[a])
        else:
            t0ground.append((2 * (test_depth[a + 1] - test_depth[a]) / layer_velocity[a]))

    t0d = []
    for k in range(len(t0ground)):
        t0d.append(np.array(t0ground[0:k + 1]).sum())
    t0 = t0d

    # reconstruct velocity
    # if its optimal estimate, use ground truth
    if optimal_flag == 1 and central_flag == 1:
        t0coff = t0
    # calculate time difference

    vel = []

    for l in range(len(peak)):
        time_diff = peak[l] - t0coff[l]

        # velocity approximation with binomial expansion
        #  vel.append(np.array(offset[l]/np.sqrt(abs(2*time_diff*t0coff[l]))))
        # velocity approximation
        int_term = abs(np.array(peak[l]) ** 2 - np.array(t0coff[l]) ** 2)
        vel.append(np.array(offset[l] / np.sqrt(int_term)))

        # solve for velocity at each layer
    v_layer = []
    for r in range(len(vel)):
        for p in range(len(vel[r])):
            if r == 0:
                v_layer.append(vel[0][p])
            else:

                v_layer.append(np.sqrt(abs(
                    (vel[r][p] ** 2 * np.array(t0coff[r]) - vel[r - 1][p] ** 2 * np.array(t0coff[r - 1])) / (
                                np.array(t0coff[r]) - np.array(t0coff[r - 1])))))

    # reconstruct depth
    v_rms, oneway = rms_velocity(t0coff, layer_velocity)

    l = 0
    depth = []
    v_layer = np.array(v_layer)

    # solve for estimated oneway travel time
    oneway_estimate = []
    for j in range(len(t0coff)):
        if j == 0:
            oneway_estimate.append((np.array(t0coff[j])) / 2)
        else:
            oneway_estimate.append((np.array(t0coff[j]) - np.array(t0coff[j - 1])) / 2)
    # reshape for processing

    v_layer = v_layer.reshape(len(peak), receiver_number)

    # deal with special case
    a = 0

    for j in v_layer:
        if central_flag != 2:
            if (np.array(j)).mean() - layer_velocity[a] > 1e3:
                v_layer[a] = abs(v_layer[a] - (np.array(j[-1]) - layer_velocity[a]))

            a += 1

    for j in range(len(v_layer)):
        depth.append(v_layer[j] * oneway[j])
    # calculate depth from ground
    ground_depth = []
    ground_depthval = np.zeros([1, len(depth[0])])
    for j in range(len(depth)):
        ground_depthval = ground_depthval + depth[j]
        ground_depth.append(ground_depthval)
    return ground_depth, v_layer, t0coff


# function to investigate root mean square velocity estimation error
# input args；
#          peak: picking arrival time
#          t0coff: estimated t0
#          receiver_distance: receiver offset

#output args:
#         f_mean: calculated root mean square velocity per layer per receiver

def root_mean_vel_error(peak, t0coff, receiver_distance, synthetic_arriavl):
    # calcualte value of function f
    f = []


    for j in range(len(peak)):
        #root mean-square velocity expression
        term = np.sqrt(abs((np.array(peak[j]) ** 2 - np.array(t0coff[j]) ** 2)))
        f.append(np.array(np.divide(receiver_distance, (term))))
    #   f.append(np.array((term)))

    #
    f_mean = np.array(f)


    return f_mean





# function for normal moveout with picking arrival time and receiver offsets in centralized and distributed manner
# input args:
#        vel_flag: 1: plot centralized Normal moveout estimation result
#        vel_flag1: 1：plot distributed normal moveout velocity estimation results
#                   0: plot  distributed normal moveout depth estimation results
#        pattern_analzye_flag: set 1: plot root-mean square velocity estimation curve analysis
#        osicillation_pattern: set 1: plot how picking time deviation influences estimation in classic normal moveout
# without recalculating picking time
#           finaltime: ground truth arrival time at receivers
#           local_information: estimated parameter t0 at all iterations for all receivers
#           local_information1: estimated parameter m0 at all iterations for all receivers
#           local_information11: estimated parameter t0 at all iterations for all receivers with Dirls algorithm
#           local_information111: estimated parameter m0 at all iterations for all receivers with Dirls algorithm
#           receiver_distance: receiver offset for each receiver on one side
#           percen: noisy link flag
#           consensus_t0: final consensus estimated t0 per layer
#           consenus_para: final consensus estimated m0 per layer
#           peak: picking travel time
#           layer_velocity: layer propagation velocity
#           test_depth: depth of each layer calculated from ground
#           layer_n: number of layers
#           receiver_number: number of receivers on one side
#output args:
#          peak: picking travel time at receivers
#          optimal_time: optimal picking travel time at receivers
#          ground_depth: estimated layer depth from centralized NMO
#          v_layer: estimated layer velocity from centralized NMO
#          t0coff: estimated t0
#          t0coffop:optimal estimated t0 (ground truth)
#          ground_depth_dis: final consensus estimated layer depth
#          v_layer_dis: final consensus estimated layer velocity
def normal_moveout(vel_flag, vel_flag1, pattern_analzye_flag, osicillation_pattern, finaltime, local_information,
                   local_information1, local_information11, local_information111, receiver_distance, percen,
                   consensus_t0, consenus_para, peak, layer_velocity, test_depth, layer_n, receiver_number):
    # generate ground truth receiver offset

    synthetic_offset=[]
    for i in range(layer_n):
        synthetic_offset.append(receiver_distance)
    synthetic_offset = np.array(synthetic_offset)
    synthetic_arriavl = sorted(np.array(finaltime).flatten())

    # use recalculated picking arrival time from distributed nmo
    newpeak11 = re_picking_arrival(np.array(local_information11).T[-1].T, np.array(local_information111).T[-1].T, receiver_distance)
    # use repicking from dirls
    arrival = re_picking_arrival(np.array(local_information).T[-1].T, np.array(local_information1).T[-1].T, receiver_distance)
    peak2 = arrival
    # use repicking with noiseless link
    peak1 = re_picking_arrival(consensus_t0, consenus_para, receiver_distance)
    #
    newpeak1 = newpeak11
    central_flag = 1

    # optimal estimation
    optimal_time, optimal_offset = np.array(synthetic_arriavl).reshape(layer_n, receiver_number), synthetic_offset
    optimal_flag = 1

    opground_depth, opv_layer, t0coffop = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                              optimal_offset, optimal_time, receiver_number,
                                                              consensus_t0, central_flag, optimal_flag)

    # FOR both sides
    if central_flag == 1:
        optimal_flag = 0
        # consider noisy transmission channel
        if percen != 0:
            # noisy picking arrival time
            noisypeak, SNR = noisy_picking(peak, percen, receiver_number)

            ground_depth, v_layer, t0coff = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                synthetic_offset, noisypeak, receiver_number,
                                                                consensus_t0, central_flag, optimal_flag)
            ground_depth1, v_layer1, t0coff1 = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                   synthetic_offset, noisypeak, receiver_number,
                                                                   consensus_t0, central_flag, optimal_flag)
        # noiseless transimmison channel
        else:
            peak_transpose = np.array(peak).T
            ground_depth, v_layer, t0coff = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                synthetic_offset, peak_transpose, receiver_number,
                                                                consensus_t0, central_flag, optimal_flag)
            ground_depth1, v_layer1, t0coff1 = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                   synthetic_offset, peak_transpose, receiver_number,
                                                                   consensus_t0, central_flag, optimal_flag)

    # apply nmo in distributed fashion
    # FOR both sides
    ground_depth_dis1, v_layer_dis1, t0coff_dis1 = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                       synthetic_offset, peak2, receiver_number,
                                                                       consensus_t0, 0, optimal_flag)
    ground_depth1_dis1, v_layer1_dis1, t0coff1_dis1 = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                          synthetic_offset, peak2, receiver_number,
                                                                          consensus_t0, 0, optimal_flag)
    # without Dirls
    ground_depth_dis, v_layer_dis, t0coff_dis = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                    synthetic_offset, newpeak1, receiver_number,
                                                                    consensus_t0, 0, optimal_flag)
    ground_depth1_dis, v_layer1_dis, t0coff1_dis = vel_depth_estimator(pattern_analzye_flag, osicillation_pattern,receiver_distance, test_depth, layer_velocity,
                                                                       synthetic_offset, newpeak1, receiver_number,
                                                                       consensus_t0, 0, optimal_flag)

    # analyze the osicillation in nmo

    if osicillation_pattern == 1:
        peak = np.array(peak).T
        plt.plot(synthetic_offset[0],abs(peak[2]-optimal_time[2]))
       # plt.plot(synthetic_offset[0], abs(np.divide(peak[2] - optimal_time[2], synthetic_offset[0] ** 2)))

        #plt.plot(synthetic_offset[0],abs(-np.divide(peak[2]-optimal_time[2],synthetic_offset[0]**2)+(np.array(optimal_time[2])**2-np.array(t0coff[2])**2)/synthetic_offset[0]**2))
        plt.xlabel('receiver offset (m)')
        plt.ylabel('$\Delta t_{d,f}$')
        plt.legend(['layer 3'])
        plt.title('picking time error $\Delta t_{d,f}$')
        plt.tight_layout()
        #plt.title('$|e_{d,f}|$ variation with different receiver offset', pad=15)
        plt.show()
    # plot analysis of estimation pattern asscoiated with picking time

    if pattern_analzye_flag == 1 and percen == 0:
        # noiseless travel time picking


        f = root_mean_vel_error(peak, t0coff, synthetic_offset[0], synthetic_arriavl)
        distance = [-receiver_distance[::-1], receiver_distance]
        for j in f:
            a = np.array(j).flatten()
            function = [np.array(j).flatten()[::-1], a]
            p1 = plt.plot(np.array(distance).flatten(), np.array(function).flatten())
            plt.legend(['layer 1', 'layer 2', 'layer 3'])
            #   plt.title('$a(t_{i,n},t_{0,n})$ variation with different receiver offset')
            plt.title('Root mean square velocity $v^{rms}_{j,n}$ variation with different receiver offset', pad=15)
            plt.xlabel('offset distance (m)')
            #  plt.ylabel('$a(t_{i,n},t_{0,n})$')
            plt.ylabel('$v^{rms}_{j,n}$')
        plt.show()
    # plot comparison between distributed and centralized nmo

    if vel_flag1 == 0:
        for j in range(len(test_depth) - 1):
            distance = [-receiver_distance[::-1], receiver_distance]
            a = np.array(ground_depth[j]).flatten()
            depth = [np.array(ground_depth[j]).flatten()[::-1], a]

            b = np.array(ground_depth_dis[j]).flatten()
            optimumdepth = [np.array(ground_depth_dis[j]).flatten()[::-1], b]
            optimumdepth1 = [np.array(ground_depth_dis1[j]).flatten()[::-1], b]
            # Dirls scheme
            plt.plot(np.array(distance).flatten(), -np.array(optimumdepth1).flatten(), label='DiRls')
            # centralized scheme
            p1 = plt.scatter(np.array(distance).flatten(), -np.array(depth).flatten(),
                             label='centralized estimated depth')
            true = test_depth[j + 1] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
            p2 = plt.plot(np.array(distance).flatten(), -true.T, linestyle='dashed', label='true depth')
            #  p3=plt.plot(np.array(distance).flatten(),-np.array(optimumdepth).flatten()[::-1],linestyle='dashed',label='distributed Estimated depth')
            # plt.legend(p2,['True Depth'])
            plt.xlabel('offset distance (m)')
            plt.ylabel('depth from ground')
            plt.title('Reconstruction of Subsurfaces of Distributed NMO')
            plt.legend(loc='best')
        plt.show()

    elif vel_flag1 == 1:
        for j in range(len(test_depth) - 1):
            distance = [-receiver_distance[::-1], receiver_distance]
            velociy = [v_layer[j][::-1], v_layer[j]]
            opyimum_vel = [v_layer_dis[j][::-1], v_layer_dis[j]]
            # DiRls scheme
            opyimum_vel1 = [v_layer_dis1[j][::-1], v_layer_dis1[j]]
            # centralized case with repicking
         #   plt.scatter(np.array(distance).flatten(), np.array(velociy).flatten(),
          #              label='centralized estimated layer velocity')
            true = layer_velocity[j] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
            plt.plot(np.array(distance).flatten(), true.T, label='True Layer Velocity')
          #  plt.plot(np.array(distance).flatten(), np.array(opyimum_vel1).flatten(), label='DiRls')

            plt.plot(np.array(distance).flatten(),np.array(opyimum_vel).flatten(),linestyle='dashed',label='distributed estimated layer velocity')
            # plt.legend(['estimated depth','true depth'])
            plt.xlabel('offset distance (m)')
            plt.ylabel('Layer Velocity (m/s)')
            plt.title('Reconstruction of Layer Velocity of Distributed NMO')
            plt.legend(bbox_to_anchor=[1, 1])
        plt.show()

    # visulize final reconstruction results for centralized NMO

    # true layer depth
    if vel_flag == 1:

        for j in range(len(test_depth) - 1):
            distance = [-receiver_distance[::-1], receiver_distance]
            a = np.array(ground_depth[j]).flatten()
            depth = [np.array(ground_depth[j]).flatten()[::-1], a]

            b = np.array(opground_depth[j]).flatten()
            optimumdepth = [np.array(opground_depth[j]).flatten()[::-1], b]
            p1 = plt.plot(np.array(distance).flatten(), -np.array(depth).flatten(), label='Estimated depth')
            true = test_depth[j + 1] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
            p2=plt.scatter(np.array(distance).flatten(),-true,label='true depth')
           # p3 = plt.plot(np.array(distance).flatten(), -np.array(optimumdepth).flatten(), linestyle='dashed',
                        #  label='Optimal Estimated depth')
            # plt.legend(p2,['True Depth'])
            plt.xlabel('offset distance (m)')
            plt.ylabel('depth from ground')
            plt.title('Reconstruction of Subsurfaces of Centralized NMO')
            if j==0:
                plt.legend(loc='best')
        plt.show()

  #  elif vel_flag == 1:
        for j in range(len(test_depth) - 1):
            distance = [-receiver_distance[::-1], receiver_distance]
            velociy = [v_layer[j][::-1], v_layer[j]]
            opyimum_vel = [opv_layer[j][::-1], opv_layer[j]]
            plt.plot(np.array(distance).flatten(), np.array(velociy).flatten(), label='Estimated Layer Velocity')
            true = layer_velocity[j] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
            plt.scatter(np.array(distance).flatten(), np.array(true).T, label='True Layer Velocity')
            # plt.plot(np.array(distance).flatten(),np.array(opyimum_vel).flatten(),linestyle='dashed',label='Optimal Estimated Layer Velocity')
            # plt.legend(['estimated depth','true depth'])
            plt.xlabel('offset distance (m)')
            plt.ylabel('Layer Velocity (m/s)')
            plt.title('Reconstruction of Layer Velocity of Centralized NMO')
            if j==0:
               plt.legend(loc='best')

        plt.show()
    # return chosen peak arrival time,ground truth time, estimated depth and velocity
    return peak, optimal_time, ground_depth, v_layer, t0coff, t0coffop, ground_depth_dis, v_layer_dis
