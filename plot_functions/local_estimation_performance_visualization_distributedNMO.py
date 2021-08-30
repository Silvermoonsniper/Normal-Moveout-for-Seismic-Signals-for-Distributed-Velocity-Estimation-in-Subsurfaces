# plot estimation performance at given iteration
import numpy as np
import matplotlib.pyplot as plt
from local_error_local_picking import re_picking_arrival, local_error_calculator
from main_algorithms.real_time_velocity_depth_estimatorNMO import specific_velocity_depth_estimator

#input args:
#       layer_n: number of layers
#       vel_flag1: flag to plot estimated layer velocity
#       layer_velocity: ground truth of layer velocity
#       test_depth: ground truth of layer depth
#       receiver_distance: distance between receivers and source
#       receiver_number: number of receivers on one side
#       local_information1: local estimated parameter t0
#       local_information: local estimated parameter m0
#       k: time step k

def plot_distributed_nmo(layer_n, vel_flag1, layer_velocity, test_depth, peak, finaltime, receiver_distance,
                         receiver_number, local_information1, local_information, ground_depth, v_layer,
                         ground_depth_dis, v_layer_dis, k):
    ## visualize the distributed normal moveout for given iterations

    # average consensus

    # iteration
    iteration = len(np.array(local_information[0]))
    ims = []

    # apply nmo with current estimate t0 and m0 at each receiver
    # FOR both sides


    synthetic_offset = []
    for i in range(layer_n):
        synthetic_offset.append(receiver_distance)
    synthetic_offset = np.array(synthetic_offset)


    # local estimate t0 at receiver
    arrivaltime = np.array(local_information).T[k].T

    # local estimate m0 at receiver
    m0_estimate = np.array(local_information1).T[k].T

    # perform a correction stage to recalculate picking
    newpeak = re_picking_arrival(arrivaltime, m0_estimate, receiver_distance)
    # implement NMO locally

    v_layer_dis, ground_depth_dis = specific_velocity_depth_estimator(k, receiver_number, layer_n, layer_velocity,
                                                                      newpeak, synthetic_offset, arrivaltime)

    if vel_flag1 == 1:
        for j in range(layer_n):
            distance = [-receiver_distance[::-1], receiver_distance]

            b = np.array(ground_depth_dis[j]).flatten()
            optimumdepth = [np.array(ground_depth_dis[j]).flatten()[::-1], b]
            true = test_depth[j + 1] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
            p2 = plt.scatter(np.array(distance).flatten(), -true, label='true depth')
            p3 = plt.plot(np.array(distance).flatten(), -np.array(optimumdepth).flatten(), linestyle='dashed',
                         label='distributed Estimated depth')

            plt.xlabel('offset distance (m)')
            plt.ylabel('depth from ground')
            title='estimated Subsurfaces for distributed NMO at iteration'+" " +str(int(k)+1)
            plt.title(title)
            if j==0:
                plt.legend(loc='best')
            ims.append(p3)
        plt.show()
        for j in range(layer_n):
            distance = [-receiver_distance[::-1], receiver_distance]

            opyimum_vel = [v_layer_dis[j][::-1], v_layer_dis[j]]

            true = layer_velocity[j] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
            p5 = plt.plot(np.array(distance).flatten(), true.T, label='True Layer Velocity')
            p4 = plt.plot(np.array(distance).flatten(), np.array(opyimum_vel).flatten(), linestyle='dashed',
                          label='distributed estimated layer velocity')
            # plt.legend(['estimated depth','true depth'])
            plt.xlabel('offset distance (m)')
            plt.ylabel('Layer Velocity (m/s)')
            title = 'estimated velocity for distributed NMO at iteration' + " " + str(int(k) + 1)
            plt.title(title, pad=15)
            if j==0:
                 plt.legend(loc='best')
        plt.show()
