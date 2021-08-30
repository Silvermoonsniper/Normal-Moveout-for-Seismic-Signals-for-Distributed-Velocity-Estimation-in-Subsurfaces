# here we try to calculate ground truth of arrival time of reflected wave
# the distance between source and receiver (unit:m)

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
#input args:
    # receiver: the receiver offset with respect to source
    # layer_n: the number of layers
    # layer_velocity: velocity of each layer
    # depth: layer depth
#output args:
#   travel_time: ground truth arrival time of reflected wave to a single receiver at certain layer
def desired_output_(layer_n, layer_velocity, depth, receiver):
    # determine angle of incidence for multiple layer

    x = []
    x_offset = 0

    # number of angle samples for test
    number = 20000
    # calculate angle for reflection at first interface

    first_angle = np.arctan(depth[0] / (0.5 * receiver))
    # angle testing range
    angle_test = np.linspace(first_angle, np.pi / 2, num=number)
    # acceptable maximum error
    threshold = 0.01
    # final calculated incident angel
    final_angle = 0
    traveltime = 0
    # flagfor only taking valid angle once
    check_flag = 0
    angle_array = np.zeros([len(layer_velocity)])
    # calculate position of reflected ray hit on ground surface
    if layer_n > 1:
        for i in angle_test:

            x_offset = 0

            angle_array[0] = i
            for j in range(1, layer_n):
                if abs((layer_velocity[j] / layer_velocity[j - 1]) * np.cos(angle_array[j - 1])) <= 1:

                    angle_array[j] = np.arccos((layer_velocity[j] / layer_velocity[j - 1]) * np.cos(angle_array[j - 1]))
                    # print((layer_velocity[j]/layer_velocity[j-1])*np.cos(angle_array[j-1]))
                    for k in range(layer_n):
                        x_offset = x_offset + 2 * depth[k] / np.tan(angle_array[k])

            if abs(x_offset - receiver) <= threshold and check_flag == 0:
                final_angle = angle_array

                # calculate reflected wave traveltime
                for i in range(layer_n):
                    traveltime = traveltime + 2 * depth[i] / (layer_velocity[i] * np.sin(final_angle[i]))

                check_flag = 1

    # if it's reflection from first layer
    if layer_n == 1:
        angle_array[0] = first_angle
        traveltime = np.sqrt((0.5 * receiver) ** 2 + depth[0] ** 2) * 2 / layer_velocity[0]

    return traveltime

# function to get ground truth arrival time for multiple receivers for different layers
# input args:
        # layer_n: number of layers
        #layer_veloicty: wave propagation velocity in each layer
        # depth: depth of each layer
        # receiver: receiver offset for whole receiver array
        # delta_t: time resolution to plot ground truth series
        # time_array: discrete time series for plot
    # output args:
        # desired:sparse ground truth series which has a peak at ground truth time instant
        #indexarray: index of ground truth time at time series
        # time_truth: ground truth arriavl time for multiple receivers
def real_signal(layer_n, layer_velocity, depth, receiver, delta_t, time_array):
    # travel time array for reflected wave from different interfaces
    #amplitude for plot Diracs at ground truth arrival time point
    peak = [1, 0.7, 0.5, 0.3]
    key = 0
    # array to store desired signal
    desired = []
    # array to store desired arrival time of reflected wave from interface
    time_truth = []
    for i in receiver:
        amplitude = 1
        real_signal = []
        time = []
        o = 0
        indexarray = []

        for j in range(1, layer_n + 1):
            time.append(desired_output_(j, layer_velocity, depth, i))

        final_time = []
        # ground truth
        time_truth = np.append(np.array(time_truth), np.array(time).flatten())
        for k in time:
            final_time.append(int(k / delta_t) * delta_t)

        for l in time_array:
            if l in final_time:

                for j in range(len(time_array)):
                    if time_array[j] == l:
                        index = j
                        indexarray.append(index)

                amplitude = peak[o]
                o += 1
                real_signal.append(amplitude)
            else:
                real_signal.append(0)
        key += 1
        plot_flag = 0
        desired.append(real_signal)

        # flag for check frequecny
        freq_flag = 1
        if key == 1 and plot_flag == 1:
            plt.plot(time_array, real_signal)
            plt.xlabel('time (s)')
            plt.ylabel('$s_n$')
            plt.title('Desired Seismic Measurement Sequence at Single Receiver')
            time_truth = np.array(time_truth).flatten()
        elif freq_flag == 0:
            return desired, indexarray, time_truth
    # ground truth arrival time
    time_truth = np.array(time_truth).flatten()
    return desired, indexarray, time_truth

