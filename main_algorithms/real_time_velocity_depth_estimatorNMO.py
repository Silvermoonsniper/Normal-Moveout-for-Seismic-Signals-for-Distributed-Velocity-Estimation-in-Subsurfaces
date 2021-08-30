# input args:
#        k: iteration in consensus stage
#        newpeak: picking time with correction at step k
#        local_information: local estimate t0 at step k
#        synthetic_offset: receiver position
#        receiver_number: number of receivers on one side
#output args:
#        v_layer: estimated layer velocity per reciever per layer
#        ground_depth: estimated layer depth per receiver per layer

import  numpy as np
def specific_velocity_depth_estimator(k, receiver_number, layer_n, layer_velocity, newpeak, synthetic_offset,
                                      local_information):
    vel = []
    for l in range(len(newpeak)):
        # velocity approximation
        int_term = abs(np.array(newpeak[l]) ** 2 - np.array(local_information[l]) ** 2)
        vel.append(np.array(synthetic_offset[l] / np.sqrt(int_term)))

        # solve for velocity at each layer
    v_layer = []
    t0coff = local_information

    for r in range(len(vel)):
        for p in range(len(vel[r])):
            if r == 0:
                v_layer.append(vel[0][p])
            else:

                v_layer.append(np.sqrt(abs(
                    (vel[r][p] ** 2 * np.array(t0coff[r][p]) - vel[r - 1][p] ** 2 * np.array(t0coff[r - 1][p])) / (
                                np.array(t0coff[r][p]) - np.array(t0coff[r - 1][p])))))
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

    v_layer = v_layer.reshape(layer_n, receiver_number)

    # deal with special case if ground truth arrival time is not sufficiently accurate
    a = 0
    for j in v_layer:

        if (np.array(j)).mean() - layer_velocity[a] > 1e3:
            v_layer[a] = abs(v_layer[a] - (np.array(j[-1]) - layer_velocity[a]))
        a += 1

    for j in range(len(v_layer)):
        depth.append(np.multiply(v_layer[j], oneway_estimate[j]))

    # calculate depth from ground
    ground_depth = []
    ground_depthval = np.zeros([1, len(depth[0])])
    for j in range(len(depth)):
        ground_depthval = ground_depthval + depth[j]
        ground_depth.append(ground_depthval)

    return v_layer, ground_depth


