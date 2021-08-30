# define root-mean square velocity solver
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(6, 6))

#calculate root mean square velocity based on internal layer velocity
#input args:
#          t0d:ground truth t0 for each layer
#          layer_velocity: layer velocity
#output args:
#          v_rms: calculated root mean-square velocity
#          oneway: one way travel time of reflection wave
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
        val = np.sqrt(nu / bu)

        v_rms.append(val)
    return v_rms, oneway


# ground truth t0
def t0_solver(test_depth, layer_velocity):
    t0ground = []
    for a in range(len(test_depth) - 1):
        if a == 0:
            t0ground.append(2 * (test_depth[a + 1]) / layer_velocity[a])
        else:
            t0ground.append((2 * (test_depth[a + 1] - test_depth[a]) / layer_velocity[a]))

    t0d = []
    print(t0ground)
    for k in range(len(t0ground)):
        t0d.append(np.array(t0ground[0:k + 1]).sum())

    return t0d


from scipy.optimize import curve_fit

#function model to apply nonlinear least-square fitting
def func(x, a, c):
    return a * x ** 2 + c ** 2

#estimate layer velocity and depth from NMO
#input args:
#          offset: receiver distance to source
#          peak: picking arrival time of reflections at receivers
#          layer_velocity: velocity of each layer
#          t0d:ground truth of t0
#          optimal_flag: flag to perform optimum estimation with perfect picking and estimated m0 and t0
#output args:
#          ground_depth: estimated layer depth (from ground surface)
#          v_layer: estimated layer velocity
#          t0coff: estimated t0
def vel_depth_estimator(offset, peak, layer_velocity, t0d, optimal_flag):
    # estimate t0 from receiver measurement
    t0coff = []
    plot_flag = 0
    for j in range(len(peak)):
        para,t0=curve_fit(func,offset[j],peak[j]**2)
        #para, t0 = t_0estimator(offset[j], peak[j])
        t0coff.append(para[1])

    if optimal_flag == 1:
        t0coff = t0d
    # reconstruct velocity
    # calculate time difference

    vel = []
    for l in range(len(peak)):
        time_diff = peak[l] - t0coff[l]

        # velocity approximation
        #  vel.append(np.array(offset[l]/np.sqrt(abs(2*time_diff*t0coff[l]))))
        # velocity approximation
        int_term = np.array(peak[l]) ** 2 - np.array(t0coff[l]) ** 2
        vel.append(np.array(offset[l] / np.sqrt(abs(int_term))))

        # solve for velocity at each layer
    v_layer = []
    for r in range(len(vel)):
        for p in range(len(vel[r])):
            if r == 0:
                v_layer.append(vel[0][p])
            else:
                #    v_layer.append(np.sqrt(abs((vel[r][p]**2*peak[r][p]-vel[r-1][p]**2*peak[r-1][p])/(peak[r][p]-peak[r-1][p]))))
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
    v_layer = v_layer.reshape(len(peak), len(np.array(peak[0])))
    # deal with special case
    a = 0

    for j in v_layer:
        if np.array(j).mean() - layer_velocity[a] > 1e3:
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


#function for normal moveout implementation and visualization of
# estimation results
#input args:
#         finaltime:ground truth arrival time at receivers
#         vel_flag: flag to plot velocity estimation result, if it sets to 1
#         final_arrival: picking travel time at receivers
#         receiver_distance: receiver offset
#         layer_velocity: wave velocity of each layer
#         test_depth:depth of each layer calculated from ground
#         layer_n:number of layers
#         delta_x: space discretization level
#         delta_t: time resolution
#output args:
#         peak:picking travel time at receivers
#         optimal_time: perfect estimated travel time
#         ground_depth: estimated depth from ground surface
#         v_layer: estimated layer velocity for all receivers
#         t0coff: estimated t0
#         t0coffop: perfect estimated t0
#         delta_t: time resolution
def normal_moveout(finaltime, vel_flag, final_arrival, receiver_distance, layer_velocity, test_depth, layer_n, delta_x,
                   delta_t):
    # ground truth of arrival time and receiver offset
    synthetic_arriavl = sorted(np.array(finaltime).flatten())
    synthetic_offset = []
    for i in range(layer_n):
        synthetic_offset.append(receiver_distance)
    synthetic_offset=np.array(synthetic_offset)
    # picking arrival time from measurement
    peak = sorted(np.array(final_arrival).flatten())
    offset=[]
    for i in range(layer_n):
        offset.append(receiver_distance)
    offset=np.array(offset).flatten()
    # reshape arrival time and offset array
    peak = np.array(peak).reshape(layer_n, len(receiver_distance))
    offset = np.array(offset).reshape(layer_n, len(receiver_distance))
    # optimal estimation
    optimal_time, optimal_offset = np.array(synthetic_arriavl).reshape(layer_n, len(receiver_distance)), np.array(
        synthetic_offset).reshape(layer_n, len(receiver_distance))
    # ground truth t0
    t0d = t0_solver(test_depth, layer_velocity)
    # flag to enable optimum estimation
    optimal_flag = 1
    opground_depth, opv_layer, t0coffop = vel_depth_estimator(optimal_offset, optimal_time, layer_velocity, t0d,
                                                              optimal_flag)
    # FOR both sides
    optimal_flag = 0
    ground_depth, v_layer, t0coff = vel_depth_estimator(offset, peak, layer_velocity, t0d, optimal_flag)
    ground_depth1, v_layer1, t0coff1 = vel_depth_estimator(offset, peak, layer_velocity, t0d, optimal_flag)

    # true layer depth
    if vel_flag == 0:

        for j in range(len(test_depth) - 1):
            distance = [-receiver_distance[::-1], receiver_distance]
            a = np.array(ground_depth[j]).flatten()
            depth = [np.array(ground_depth[j]).flatten()[::-1], a]

            b = np.array(opground_depth[j]).flatten()
            optimumdepth = [np.array(opground_depth[j]).flatten()[::-1], b]
            p1 = plt.plot(np.array(distance).flatten(), -np.array(depth).flatten(), label='Estimated depth')
            true = test_depth[j + 1] * np.ones([1, 2 * len(np.array(offset[j]).flatten())])
            p2 = plt.scatter(np.array(distance).flatten(), -true, label='true depth')
            p3 = plt.plot(np.array(distance).flatten(), -np.array(optimumdepth).flatten()[::-1], linestyle='dashed',
                          label='Optimal Estimated depth')
            # plt.legend(p2,['True Depth'])
            plt.xlabel('offset distance (m)')
            plt.ylabel('depth from ground')
            plt.title('Reconstruction of Subsurfaces')
            plt.legend(bbox_to_anchor=[1, 1])

    elif vel_flag == 1:
        for j in range(len(test_depth) - 1):
            distance = [-receiver_distance[::-1], receiver_distance]
            velociy = [v_layer[j][::-1], v_layer[j]]
            opyimum_vel = [opv_layer[j][::-1], opv_layer[j]]
            plt.plot(np.array(distance).flatten(), np.array(velociy).flatten(), label='Estimated Layer Velocity')
            true = layer_velocity[j] * np.ones([1, 2 * len(np.array(offset[j]).flatten())])
            plt.scatter(np.array(distance).flatten(), true, label='True Layer Velocity')
            plt.plot(np.array(distance).flatten(), np.array(opyimum_vel).flatten(), linestyle='dashed',
                     label='Optimal Estimated Layer Velocity')
            # plt.legend(['estimated depth','true depth'])
            plt.xlabel('offset distance (m)')
            plt.ylabel('Layer Velocity (m/s)')
            plt.title('Reconstruction of Layer Velocity')
            plt.legend(bbox_to_anchor=[1, 1])
    plt.show()
    # return chosen peak arrival time,ground truth time, estimated depth and velocity
    return peak, optimal_time, ground_depth, v_layer, t0coff, t0coffop, delta_t
