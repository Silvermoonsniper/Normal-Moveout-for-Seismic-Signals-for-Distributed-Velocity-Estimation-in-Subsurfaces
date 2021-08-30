# function to recalculate picking arrival time locally estimate at receiver
import  numpy as np
#input args:
#       local_information: locally estimated parameter t0 per sensor per layer
#       local_information1: locally estimated parameter m0 per sensor per layer
#       receiver_distance: distance of each receiver to source
#output args:
#       arrival_time: local picking travel time at sensors with recalculation of picking
#       time with estimated parameters and NMO equation
def local_re_picking_arrival(local_information, local_information1, receiver_distance):
    arrival_time = []

    for j in range(len(local_information)):
        data = np.sqrt(
            np.array(local_information1)[j] * np.array(receiver_distance) ** 2 + np.array(local_information)[j] ** 2)

        arrival_time.append(data)
    return arrival_time


# function to recalculate picking arrival time with final consensused parameters
# input args:
#          consensus_t0: final consensus estimated t0 for each layer
#          consenus_para: final consensus estimated m0 for each layer
#          receiver_distance: receiver offset in (m)
def re_picking_arrival(consensus_t0, consenus_para, receiver_distance):
    arrival_time = []

    for j in range(len(consensus_t0)):
        arrival_time.append(consenus_para[j] * receiver_distance ** 2 + consensus_t0[j])
    return arrival_time


# calculate local estimation error from NMO
#input args:
#         layer_velocity1: ground truth layer velocity
#         test_depth1: ground truth layer depth
#         ground_depth_dis: local estimated layer depth
#         v_layer_dis:local estimated layer velocity
#output args:
#         np.array(depth_errordis).mean(): normalized averaged depth estimation error
#         np.array(velocity_errordis).mean(): normalized averaged velocity estimation error

def local_error_calculator(layer_velocity1, test_depth1, ground_depth_dis, v_layer_dis):
    # define reconstruction error array for depth and velocity reconstruction for centralized or distributed scheme
    depth_error, velocity_error, depth_errordis, velocity_errordis = [], [], [], []
    a = 1
    #loop over estimated depth and velocity for each layer
    for j in ground_depth_dis:
        depth_errordis.append(np.log10(np.array(abs(ground_depth_dis[a - 1] - test_depth1[a])).mean() / test_depth1[a]))

        a += 1
    b = 0
    for k in v_layer_dis:
        velocity_errordis.append(np.log10(np.array(abs(v_layer_dis[b] - layer_velocity1[b])).mean() / layer_velocity1[b]))

        b += 1

    return np.array(depth_errordis).mean(), np.array(velocity_errordis).mean()
