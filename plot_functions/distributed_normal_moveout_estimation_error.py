from matplotlib import animation
from IPython.display import HTML
from local_error_local_picking import *
from main_algorithms.real_time_velocity_depth_estimatorNMO import specific_velocity_depth_estimator
import matplotlib.pyplot as plt
# function to calculate estimation error of distributed normal-moevout with average consensus
# network
#input args:
            # layer_n：number of layers
            # test_depth: depth of each layer calculated from ground
            # layer_velocity: velocity of each layer
            # ground_depth: estimated depth
            # v_layer:estimated layer velocity
            # receiver_distance: estimated
            # finaltime：ground truth arrival time of reflections at receievers
            # local_information：estimated paramter t0 at all time steps
            # local_information1：estimated paramter m0 at all time steps
            # receiver_number: number of receivers on one side
#output args:
#       v_error_array: normalized average velocity estimation error for all iterations with distributed NMO
#       d_error_array: normalized average depth estimation error for all iterations with distributed NMO
#       v_error_array_cen: normalized average velocity estimation error for all iterations with centralized NMO
#       d_error_array_cen: normalized average velocity estimation error for all iterations with centralized NMO

def distributed_NMO_estimation(layer_n, test_depth, layer_velocity, ground_depth, v_layer, receiver_distance, finaltime,
                               local_information, local_information1, receiver_number, peak, t0_estimate, para_estimate,
                               accessible_sensor):
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=(6,6))

    ## visualize the distributed normal moveout for given iterations


    # iteration
    iteration = len(np.array(local_information[0][0]))
    print(iteration)
    ims = []
    v_error_array, d_error_array = [], []
    v_error_array_cen, d_error_array_cen = [], []
    # time step array
    time_step = np.arange(iteration)
    for k in range(iteration):
        # apply nmo with current estimate t0 and m0 at each receiver
        # FOR both sides
        synthetic_arriavl = sorted(np.array(finaltime).flatten())
        # receiver offset for each layer
        synthetic_offset = []
        for i in range(layer_n):
            synthetic_offset.append(receiver_distance)
        synthetic_offset = np.array(synthetic_offset)

        # flag for optimimum estimation
        optimal_flag = 0

        # local estimate t0 at receiver
        arrivaltime = np.array(local_information).T[k].T
        # local estimate m0 at receiver
        m0_estimate = np.array(local_information1).T[k].T
        # perform a correction stage to recalculate picking
        # newpeak=local_re_picking_arrival(arrivaltime,m0_estimate,receiver_distance)
        newpeak = re_picking_arrival(arrivaltime, m0_estimate, receiver_distance)
        # implement NMO locally

        v_layer_dis, ground_depth_dis = specific_velocity_depth_estimator(k, receiver_number, layer_n, layer_velocity,
                                                                          newpeak, synthetic_offset, arrivaltime)

        # calculate estimation error
        d_error, v_error = local_error_calculator(layer_velocity, test_depth, ground_depth_dis, v_layer_dis)
        v_error_array.append(v_error)
        d_error_array.append(d_error)

        # calculate estimation error for centralzied NMO
        d_error_cen, v_error_cen = local_error_calculator(layer_velocity, test_depth, ground_depth, v_layer)
        v_error_array_cen.append(v_error_cen)
        d_error_array_cen.append(d_error_cen)

        compare_flag = 10
        vel_flag1 = 10

        if compare_flag == 1:
            for j in range(1, 2):
                distance = [-receiver_distance[::-1], receiver_distance]

                b = np.array(ground_depth_dis[j]).flatten()
                optimumdepth = [np.array(ground_depth_dis[j]).flatten()[::-1], b]
                true = test_depth[j + 1] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
                p2 = ax.scatter(np.array(distance).flatten(), -true, label='true depth')
                p3 = ax.plot(np.array(distance).flatten(), -np.array(optimumdepth).flatten()[::-1], linestyle='dashed',
                             label='distributed Estimated depth')

                ax.set_xlabel('offset distance (m)')
                ax.set_ylabel('depth from ground')
                ax.set_title('Reconstruction of Subsurfaces')
                if k == 2:
                    ax.legend()
                ims.append(p3)
        elif vel_flag1 == 1:
            for j in range(1, 2):
                distance = [-receiver_distance[::-1], receiver_distance]

                opyimum_vel = [v_layer_dis[j][::-1], v_layer_dis[j]]

                true = layer_velocity[j] * np.ones([1, 2 * len(np.array(synthetic_offset[j]).flatten())])
                p5 = plt.scatter(np.array(distance).flatten(), true.T, label='True Layer Velocity')
                p4 = plt.plot(np.array(distance).flatten(), np.array(opyimum_vel).flatten(), linestyle='dashed',
                              label='distributed estimated layer velocity')
                # plt.legend(['estimated depth','true depth'])
                plt.xlabel('offset distance (m)')
                plt.ylabel('Layer Velocity (m/s)')
                plt.title('Reconstruction of Layer Velocity', pad=15)
                if k == 2:
                    plt.legend()

            ims.append(p4)

    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
    HTML(ani.to_html5_video())

    return v_error_array, d_error_array, v_error_array_cen, d_error_array_cen, ims
