# regularized least square to recover clean local estimate t0 and m0
# input args:
#          sigma_hat: estimate noise variance from data
#          weight_matrix:weight matrix in consensus averaging
#          information: noisy t0 or parameter estimate with noisy link
#output args:
#          next_infomation: recovered information at current time step
import numpy as np
import matplotlib.pyplot as plt
def RLS_processing(weight_matrix, sigma_hat, information):
    wwt = np.matmul(weight_matrix.T, weight_matrix)
    # identity matrix
    identity = np.eye(len(wwt))
    # set regularization coefficient
    nambda = 1 / sigma_hat ** 3
    inverse = wwt / sigma_hat ** 2 + nambda * identity

    # estimated t0 or m0 at step k
    second = np.matmul(weight_matrix.T, information)
    final_information = np.matmul(np.linalg.inv(inverse), second / sigma_hat ** 2)
    # recalculate estimate t0 or m0 at step k+1`
    next_infomation = np.matmul(weight_matrix, final_information)
    # return estimated noiseless t0 or m0 at step k+1
    return next_infomation


# define function to get estimate of single sensor from its neighbours
#input args:
#            information: estimated parameters at all receivers at one time step
#            indice: indice of receiver
def local_data_retriver(information, indice):
    data = []
    for j in range(len(information)):
        if j in indice:
            data.append(information[j])
    return data


#  distributed average consensus with information sharing in the sensor network
# input args:
#              time_step_average_consensus: time step for running average consensus algorithm
#              recover_flag: flag to use regularized-leat square to recover estimated parameters under noisy link
#              sensor_indice: indice of sensor
#              noise_flag: the flag to indicate if it is noisy link
#                          if it is noisy link, this flag is set to 1, else: 0
#              SNR_noisy_link: the SNR for noisy link
#              agentnumber: number of receivers on one side
#              information: initial estimated parameters
#              accessible_sensor: number of neighbours for each receiver
#              t0_truth: ground truth of parameter t0
#              m0_truth: ground truth of parameter m0
#              legend_number: number of legend to draw the legend
#output args:
#              consensus_value: final consensus parameter
#              average: average of initial estimated parameters at receivers
#              information_array: array to store estimated parameters per sensor per iteration

legend1 = ['ground truth $t_{0,1}$', 'ground truth $t_{0,2}$', 'ground truth $t_{0,3}$']
title_legend = ['Consensus averaging on $\hat{t}_{0,d,1}$', 'Consensus averaging on $\hat{t}_{0,d,2}$',
                'Consensus averaging on $\hat{t}_{0,d,3}$']
title_legend1 = ['DiRls on $t_{0,0}$', 'DiRls on $t_{0,1}$', 'DiRls on $t_{0,2}$']
y_label = ['$\hat{t}_{0,d,1}$', '$\hat{t}_{0,d,2}$', '$\hat{t}_{0,d,3}$']


def random_gossip(time_step_average_consensus,recover_flag, sensor_indice, noise_flag, SNR_noisy_link, agentnumber, information, accessible_sensor,
                  t0_truth, m0_truth, legend_number):
    # for j in range(agentnumber):
    robot_xcoordinate = np.random.randint(5, size=(agentnumber, 1))
    robot_ycoordinate = np.random.randint(15, size=(agentnumber, 1))
    # information in the agents

    # create geometrical area size
    area = np.pi * 100
    colors = (0, 0, 0)
    # flag for plotting a randomly connected agent network
    plot_flag = 0
    if plot_flag == 1:
        # plot the agents in the predefined area
        plt.subplot(2, 1, 1)
        plt.scatter(robot_xcoordinate, robot_ycoordinate, s=area, c=colors, alpha=0.5)
        # choose
        # randomly connect the agents
        plt.plot(robot_xcoordinate, robot_ycoordinate)
        plt.xlabel('measurement')
        plt.ylabel('position')
    # construct connectivity matrix
    A = np.identity(len(robot_xcoordinate))
    t = len(robot_xcoordinate)
    # initialize weighting matrix
    weight_matrix = np.zeros([t, t])
    for p in range(t):

        for j in range(t):
            if j in accessible_sensor[p]:
                A[p][j] = 1
                # assign weight assocaited with neighbour
                if j != p:
                    weight_matrix[p][j] = 1 / (len(accessible_sensor[p]) + 1)
        # assign weight associate with receiver itself
        weight_matrix[p][p] = 1 - np.sum(weight_matrix[p][:])



    #array to store local estimated paramters
    local_t0 = []
    iteration = 0
    particular = []
    # number of receivers
    number_ofagents = len(robot_xcoordinate)
    # array to store update estimate for all receivers
    information_array = np.zeros([number_ofagents, time_step_average_consensus])


    # noise vector
    std = np.sqrt(np.var(information))/(10**(SNR_noisy_link/20))
    AWGN_vector = np.random.randn(len(information))

    initial = information
    # average of individual t0 estimate
    average = np.ones([len(information_array[0]), 1]) * np.mean(information)

    for i in range(time_step_average_consensus):


        # select a node k from network randomly

        for k in range(number_ofagents):
            # local t0 estimate
            local_t0.append(information)
            iteration += 1
            #append for estimated parameters at each receiver for each iteration
            information_array[k][i] = information[k]
            for l in range(number_ofagents ):


                # select the neighbour agent l with respect to k and check if sensor already converges

                if A[k][l] == 1:
                    # select the value x_e x_l
                    newxcor = information[k]
                    newxcor1 = information[l]
                    # calcualte the average value of xcor and ycor
                    information[k] = np.array(newxcor + 1 / (len(accessible_sensor[p]) + 1) * (newxcor1 - newxcor))
                    #information[k]=np.matmul(weight_matrix[k][:],information)



            # add AWGN in channel if we consider noisy link
            if noise_flag == 1:
                information[k] = np.add(information[k], std * AWGN_vector[k])

        # calculate mean at first step noisy estimate t0 and m0
        if i == 0:
            # estimated value available in a single sensor
            local_sensor_data = local_data_retriver(information, accessible_sensor[sensor_indice])

            initial_mean = np.mean(local_sensor_data)
        # if we apply RLS to noisy link
        if recover_flag == 1 and noise_flag == 1:
            # estimated noise standard deviation
            sigma_hat = std

            # recovered noiseless estimate t0 or m0 at step i+1R
            information = RLS_processing(weight_matrix, sigma_hat, information) / (i + 1)
            for p in range(number_ofagents):
                local_sensor_data = local_data_retriver(information, accessible_sensor[p])
                intial_mean_array = []

                information[p] = information[p] + initial_mean
            # calculate new average after regularized least square
            new_mean = np.ones([len(information_array[0]), 1]) * np.mean(information)

    # final consensused estimate t0 or m0 with iterative RLS
    if recover_flag == 1:
        consensus_value = abs(initial_mean)
    else:
        consensus_value = information_array[0][-1]
    groundtruth = np.ones([len(information_array[0]), 1]) * np.array(t0_truth[legend_number])
    #groundtruth=np.ones([len(information_array[0]),1])*np.array(m0_truth[legend_number])
       # visualize results
    vi_flag = 1

    if vi_flag == 1:
        x = np.linspace(1, len(information_array[0]), len(information_array[0]))

        plt.subplot(2, 1, 1)
        for j in range(agentnumber):
            plt.plot(x, information_array[j])
        plt.scatter(x, average, label='average of all initial estimate')
      #  plt.plot(x, groundtruth, '--', label=legend1[legend_number])

        plt.xlabel('iteration')
        # plt.ylabel('estimated $t_0$')
        plt.ylabel(y_label[legend_number])
        plt.title(title_legend[legend_number], pad=15)

        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    return consensus_value, average, information_array


# function to perform average consensus on the estimated parameters.
# input args:
#      time_step_average_consensus: time step for running average consensus algorithm
#      receiver_number: number of receivers
#      t0_estimate: individual estimate of t0
#      accessible_sensor: available sensors for each sensor
#      layer_n: number of layers
#      noise_flag: flag to indicate it is noisy link or not
#      recover_flag: flag to use regularized-leat square to recover estimated parameters
#      SNR_noisy_link: SNR for noisy link
#      receiver_number: number of receivers
#      t0_estimate: initial estimate t0 for all layers at all receivers
#      para_estimate: initial estimate m0 for all layers at all receivers
#      accessible_sensor: number of neighbors

#output args:
#      consensus_t0: final consensused estimated t0
#      consenus_para: final consensused estimated m0
#      local_information: estimated parameter t0 at all iterations for all receivers
#      local_information1: estimated parameter m0 at all iterations for all receivers
def t0_average_consensus(time_step_average_consensus,t0_truth, m0_truth, layer_n, noise_flag, recover_flag, SNR_noisy_link, receiver_number,
                         t0_estimate, para_estimate, accessible_sensor):
    consensus_t0 = []
    consenus_para = []
    # update value for each layer
    update = []
    average_val = []

    # set iteration for running consensus averaging for randomly chosen neighbours
    local_information = []
    local_information1 = []
    # loop over layers
    for j in range(layer_n):
        # apply consensus averaging

        sensor_indice = 3
        # average consensus on parameters t0

        consenus_val, average, finallocal_t0 = random_gossip(time_step_average_consensus,recover_flag, sensor_indice, noise_flag, SNR_noisy_link,
                                                             receiver_number, t0_estimate.T[j], accessible_sensor,
                                                             t0_truth, m0_truth, j)

        # average consensus on quadratic coefficient m0
        consenus_para_estimate, average, finallocal_t01 = random_gossip(time_step_average_consensus,recover_flag, sensor_indice, noise_flag, SNR_noisy_link,
                                                                        receiver_number, para_estimate[j],
                                                                      accessible_sensor, t0_truth, m0_truth, j)
        # append for different layers
        consensus_t0.append(consenus_val)
        consenus_para.append(consenus_para_estimate)
        local_information.append(finallocal_t0)
        local_information1.append(finallocal_t01)

    # return t0 estimate after consensus for each layer
    return consensus_t0, consenus_para, local_information, local_information1


