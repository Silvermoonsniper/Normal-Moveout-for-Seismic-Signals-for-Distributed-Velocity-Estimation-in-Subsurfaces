# Simulate seismic wave propagation in 2D
# use different acoustic wave velocities.Consider a source-receiver framework
# select a given region and discretize over it, we only consider seismic body waves
# in isotropic, homogeneous elastic media.
#  Fdtd is implemented in two ways:
#       1. scalar codes to compute wave ampltitude for all cells, typically slow,
#       2. vectorized codes to accelerate computation, very fast.
#   in both cases, wave simulation is almost identical, but further data processing  (deconvolution,picking) is done
#   with scalar codes. For vectorized codes, seismic measurement differs from scalar codes shows.


import math
import numpy as np

import matplotlib.pyplot as plt
from Ricker_wavelt import *



# dicretize sampling time
# this function aims to calculate seismic_wave velocity_generator

# define stability check function of 2D wave propagagtion
def numerical_stability(vel, delta_t, delta_h, delta_x):
    flag = 1 / (vel * ((1 / delta_h ) + (1 / delta_x )))

    if delta_t <= flag:
        return True
    else:
        return False
    # this function aims to get initial shape of seismic wave, here we could test for different initial


# wave displacement shape

def initial(b, h, L):
    A = 2
    # all constant
    constant_flag = 1
    layer_case = 0
    sine_flag = 0
    # depth from ground for each layer
    depth = np.linspace(0, 3e3, num=5)
    # flag for switching between test mode and real mode
    source_testflag = 0
    if constant_flag == 1:
        initial_value = 0
    if layer_case == 1:
        displacement = [0.7e0, 2.3e0, 3e0, 6e0, 7.8e0]
        k = 0
        while k + 1 < len(depth):
            if depth[k] <= h <= depth[k + 1]:
                initial_value = displacement[k]
                return initial_value
            else:
                k += 1

    if sine_flag == 1:
        initial_value = A * np.sin(math.pi * b / L)

    # if we try test

    if source_testflag == 1:
        initial_value = manual_solution(b, h, 0)

    return initial_value


# input parameters:
# recall stress-strain realtionship
nambda = 3e10
mu = 3e10
# linear density of geological material (unit:kg/m^2)
rho = 3000


# implement a source function
def source(x, h, t):
    # recall stress-strain realtionship
    nambda = 3e10
    mu = 3e10
    # linear density of geological material (unit:kg/m^2)
    rho = 3000
    # acoustic wave phase velocity
    wave_velocity = 1 / (np.sqrt((nambda + 2 * mu) / rho))
    source_value = -(2 * x * (1 - x) * h / wave_velocity ** 2) - 2 * t ** 2 * h
    return source_value


# construct an arbitrary solution of wave displacement for software verification
def manual_solution(x, h, t):
    solutionvalue = x * (1 - x) * (t) ** 2 * h

    return solutionvalue


# varying velocity function, assign velocity for each individual mesh point according to layer-cake model
# input parameters: n,J:distance index NUMBER and depth index NUMBER
# depth: the depth of each layer in layered-earth model
# delta_h: depth disrectization length
# initialize depth and velocity of each layer


def vary_velocity(depth, n, J, delta_h, layer_velocity):
    delta_h = depth[-1] / J
    # loop over all mesh points
    velocity = np.zeros([n, J])
    # initial displacement
    k = 0
    for i in range(n):
        for j in range(J):
            # check current point depth
            current_depth = j * delta_h

            while k + 1 < len(depth):

                if depth[k] <= current_depth <= depth[k + 1]:
                    velocity[i][j] = layer_velocity[k]

                k += 1
            k = 0
    return velocity


# return index of element in a given array
#input args:
#      interface_coordinate: depth indice corresponds to an interface
#      k: depth indice  to check if it is an interface
#output args:
#      i: valid interface coordinate
def index_search(interface_coordinate, k):
    for i in range(len(interface_coordinate)):
        if interface_coordinate[i] == k:
            return i


# fdtd solver
# input args:
#####
# n: total number of spatial grids in x-direction
# J: total number of spatial grids in depth
# time_rounds: total time discretization number
# delta_t: time interval for a sample
# nambda,mu: first and second Lame coefficients
# rho:linear density of geological material
# distance: length of computation region
# fm: frequency of ricker wavelet
# receiver_spaceindex: index of receiver cell
# source_location: index of source cell
# define a space index for receiver locations
# output args:
#         seismic_trace_x: measurement date at receiver
#         time_grid_pressure: wave amplitude in whole computation domain for all iterations
#         time_stamps: time array






def Arrivaltime_estimation(test_depth, layer_velocity, delta_x, delta_h, time_rounds, delta_t, distance, fm, receiver_spaceindex,
                           source_coordinate_x):

    i, j = 1, 1
    #indice of source cell
    source_location=int(source_coordinate_x/delta_x)
    # number of time rounds for simulation

    time_rounds = int(time_rounds)
    #calculate number of cells in horizontal x diretion
    n = distance / delta_x
    # calculate number of cells in vertical y direction
    J = test_depth.max() / np.array(delta_h)
    n = int(n)
    J = int(J)

    interface_coordinate = np.array(test_depth[1:] / delta_h - 1).flatten()
    for a in range(len(interface_coordinate)):
        interface_coordinate[a] = int(interface_coordinate[a])

    # initial sound pressure array, with dimension of n J
    grid_pressure = np.zeros([n, J, 2])
    grid_pressure_1 = np.zeros([n, J, 2])
    grid_pressure_2 = np.zeros([n, J, 2])
    # acquire velocity for each mesh point

    velocity = vary_velocity(test_depth, n, J, delta_h, layer_velocity)

    # calculate reflected time

    time_grid_pressure = []

    # check courant number condition
    for i in velocity.flatten():
        flag = numerical_stability(np.array(i), delta_t, delta_h, delta_x)
        if flag == False:
            raise TypeError("courant number condition is not fulfilled")
    velocity = vary_velocity(test_depth, n, J, delta_h, layer_velocity) ** 2
    # establish new velocity matrix for vectorized computation
    newvelocity = np.zeros([n, J, 2])
    for i in range(n):
        for j in range(J):
            newvelocity[i][j] = [velocity[i][j], velocity[i][j]]

    # flag for switching between test mode and real mode
    source_testflag = 0

    # receiver distance
    receiver_distance = receiver_spaceindex * np.array(delta_x)
    # time shift to let Ricker wavelet starts at almost zero
    time_shift = 5 * np.sqrt(2) / (2 * np.pi * fm)
    # peak index of source wavelet
    peak_source = int(time_shift / delta_t)
    # compute true arrival time
    t1 = np.sqrt((receiver_distance / 2) ** 2 + test_depth[1] ** 2) * 2 / layer_velocity[0]

    # initial wave amplitude at time instant 1 and 2 for each grid
    for k in range(n):
        for j in range(J):

            # initialization of wave displacement in x-h plane
            wave_vector = np.array(
                [initial(k * delta_x, j * delta_h, distance), initial(k * delta_x, j * delta_h, test_depth.max())])

            grid_pressure_1[k][j] = wave_vector
            if k == source_location and j == 0:
                grid_pressure_1[k][j] = [Ricker_response(fm, delta_t * 0 - time_shift),
                                         Ricker_response(fm, delta_t * 0 - time_shift)]

    # for anisotropical media
    vary_velocityflag = 1
    # STACK TO GET grid pressure for different time steps
    time_grid_pressure.append(np.array(grid_pressure_1))

    # calculate wave amplitude at second time interval for each grid
    for p in range(1, n - 1):
        for o in range(1, J - 1):
            pre = delta_t / (delta_x)
            pre1 = delta_t / (delta_h)
            # discretized velocity at three consecutive time instants
            vel_i = velocity[p][o]
            vel_next = velocity[p + 1][o]
            vel_pre = velocity[p - 1][o]
            vel_i1 = velocity[p][o]
            vel_next1 = velocity[p][o + 1]
            vel_pre1 = velocity[p][o - 1]
            if source_testflag == 1:
                grid_pressure[p][o] = grid_pressure_1[p][o] + 0.5 * (pre ** 2 * (
                            0.5 * (vel_i + vel_next) * (grid_pressure_1[p + 1][o] - grid_pressure_1[p][o]) - 0.5 * (
                                vel_i + vel_pre) * (grid_pressure_1[p][o] - grid_pressure_1[p - 1][o])) + pre1 ** 2 * (
                                                                                 0.5 * (vel_i1 + vel_next1) * (
                                                                                     grid_pressure_1[p][o + 1] -
                                                                                     grid_pressure_1[p][o]) - 0.5 * (
                                                                                             vel_i1 + vel_pre1) * (
                                                                                             grid_pressure_1[p][o] -
                                                                                             grid_pressure_1[p][
                                                                                                 o - 1])))
            else:
                grid_pressure[p][o] = grid_pressure_1[p][o] + 0.5 * (pre ** 2 * (
                            0.5 * (vel_i + vel_next) * (grid_pressure_1[p + 1][o] - grid_pressure_1[p][o]) - 0.5 * (
                                vel_i + vel_pre) * (grid_pressure_1[p][o] - grid_pressure_1[p - 1][o])) + pre1 ** 2 * (
                                                                                 0.5 * (vel_i1 + vel_next1) * (
                                                                                     grid_pressure_1[p][o + 1] -
                                                                                     grid_pressure_1[p][o]) - 0.5 * (
                                                                                             vel_i1 + vel_pre1) * (
                                                                                             grid_pressure_1[p][o] -
                                                                                             grid_pressure_1[p][
                                                                                                 o - 1])))

    # calculate u for ground boundary point and bottom boundary point at time instant 1
    # restore all wave displacement for same x location
    pre = delta_t / (delta_x)
    pre1 = delta_t / (delta_h)
    for j in range(n - 1):
        # discretized velocity at three consecutive time instants
        v_i = velocity[j][0]

        v_next = velocity[j + 1][0]

        v_pre = velocity[j - 1][0]
        v_i1 = velocity[j][0]
        v_next1 = velocity[j][1]
        v_pre1 = velocity[j][1]

        # calculate displacement at receivers
        mur_flag = 1
        if mur_flag == 1:
            if j != source_location:
                wb = (np.sqrt(velocity[j][0]) * delta_t - delta_h) / (np.sqrt(velocity[j][0]) * delta_t + delta_h)
                grid_pressure[j][0] = grid_pressure_1[j][1] + wb * grid_pressure[j][1] - wb * grid_pressure_1[j][0]
            else:
                grid_pressure[j][0] = grid_pressure_1[j][1] + wb * grid_pressure[j][1] - wb * grid_pressure_1[j][
                    0] - delta_t ** 2 * np.array([Ricker_response(fm, (i + 1) * delta_t - time_shift),
                                                  Ricker_response(fm, (i + 1) * delta_t - time_shift)])

        else:
            if j != source_location:
                grid_pressure[j][0] = pre ** 2 * (0.5 * (v_i + v_next) * (
                            grid_pressure_1[j + 1][0] - 2 * grid_pressure_1[j][0] + grid_pressure_1[j - 1][0])) + 2 * \
                                      grid_pressure_1[j][0] - grid_pressure_2[j][0] + pre1 ** 2 * (
                                                  0.5 * (v_i1 + v_next1) * (
                                                      grid_pressure_1[j][1] - 2 * grid_pressure_1[j][0] +
                                                      grid_pressure_1[j][1]))

        # wave displacement in the bottom boundary points
        v_ib = velocity[j][J - 1]
        v_nextb = velocity[j + 1][J - 1]
        v_preb = velocity[j - 1][J - 1]
        v_i1b = velocity[j][J - 2]
        v_next1b = velocity[j][J - 1]
        v_pre1b = velocity[j][J - 2]
        grid_pressure[j][J - 1] = 0.5 * (pre ** 2 * (
                    0.5 * (v_ib + v_nextb) * (grid_pressure_1[j + 1][J - 1] - grid_pressure_1[j][J - 1]) - 0.5 * (
                        v_ib + v_preb) * (grid_pressure_1[j][J - 1] - grid_pressure_1[j - 1][J - 1])) + pre1 ** 2 * (
                                                     0.5 * (v_i1b + v_next1b) * (
                                                         grid_pressure_1[j][J - 2] - grid_pressure_1[j][
                                                     J - 1]) - 0.5 * (v_i1b + v_pre1b) * (
                                                                 grid_pressure_1[j][J - 1] - grid_pressure_1[j][
                                                             J - 2]))) + 2 * grid_pressure_1[j][J - 1] - \
                                  grid_pressure_2[j][J - 1]
    # for left and right boundary
    for f in range(1, J - 1):
        # apply nonlinear weighting
        w = (np.sqrt(velocity[0][f]) * delta_t - delta_x) / (np.sqrt(velocity[0][f]) * delta_t + delta_x)
        grid_pressure[0][f] = grid_pressure_1[1][f] + w * grid_pressure[1][f] - w * grid_pressure_1[0][f]
        w1 = (np.sqrt(velocity[n - 1][f]) * delta_t - delta_x) / (np.sqrt(velocity[n - 1][f]) * delta_t + delta_x)
        grid_pressure[n - 1][f] = grid_pressure_1[n - 2][f] + w1 * grid_pressure[n - 2][f] - w1 * \
                                  grid_pressure_1[n - 1][f]

        # STACK TO GET grid pressure for different time steps
    time_grid_pressure.append(np.array(grid_pressure))

    # switch variables for next time step
    grid_pressure_2[:][:], grid_pressure_1[:][:] = grid_pressure_1, grid_pressure

    time_stamps = np.zeros([time_rounds, 1])

    # initialize array

    receiver_value = np.zeros([n, time_rounds])
    # propagate pressure in a given grid with grid pressure at previous time instants
    # LOOP over a given simulation time limit
    seismic_trace = []
    receiver_data = []
    time_tracker = []

    # i is time index, where j is position index
    for i in range(1, time_rounds):
        # record simulation time points
        time_stamps[i] = i * delta_t

        # inject a ricker wavelet source
        # set repeated interval for ricker wavelet exciation
        grid_pressure_1[source_location][0] = np.array(
            [Ricker_response(fm, (i) * delta_t - time_shift), Ricker_response(fm, (i) * delta_t - time_shift)])
        grid_pressure[source_location][0] = np.array(
            [Ricker_response(fm, (i + 1) * delta_t - time_shift), Ricker_response(fm, (i + 1) * delta_t - time_shift)])

        # do vectorized computation to save time
        vectorzied_flag = 1
        if vectorzied_flag == 1 and vary_velocityflag == 1:

            pre = delta_t / (delta_x)
            pre1 = delta_t / (delta_h)
            # discretized velocity at three consecutive time instants

            # for layer 1
            layer1u_xx = grid_pressure_1[:-2, 1:int(interface_coordinate[0])] - grid_pressure_1[1:-1,
                                                                                1:int(interface_coordinate[0])]
            layer1u_xx1 = grid_pressure_1[1:-1, 1:int(interface_coordinate[0])] - grid_pressure_1[2:,
                                                                                  1:int(interface_coordinate[0])]
            layer1u_yy = grid_pressure_1[1:-1, :int(interface_coordinate[0]) - 1] - grid_pressure_1[1:-1,
                                                                                    1:int(interface_coordinate[0])]
            layer1u_yy1 = grid_pressure_1[1:-1, 1:int(interface_coordinate[0])] - grid_pressure_1[1:-1,
                                                                                  2:int(interface_coordinate[0]) + 1]
            grid_pressure[1:-1, 1:int(interface_coordinate[0])] = 2 * grid_pressure_1[1:-1,
                                                                      1:int(interface_coordinate[0])] - grid_pressure_2[
                                                                                                        1:-1, 1:int(
                interface_coordinate[0])] + layer_velocity[0] ** 2 * pre ** 2 * (layer1u_xx - layer1u_xx1) + \
                                                                  layer_velocity[0] ** 2 * pre1 ** 2 * (
                                                                              layer1u_yy - layer1u_yy1)
            # for layers in between
            for s in range(1,len(layer_velocity)-1):

                layer2u_xx = grid_pressure_1[:-2,
                         int(interface_coordinate[s-1]) - 1:int(interface_coordinate[s])] - grid_pressure_1[1:-1, int(
                         interface_coordinate[s-1]) - 1:int(interface_coordinate[s])]
                layer2u_xx1 = grid_pressure_1[1:-1,
                          int(interface_coordinate[s-1] - 1):int(interface_coordinate[s])] - grid_pressure_1[2:, int(
                          interface_coordinate[s-1]) - 1:int(interface_coordinate[s])]
                layer2u_yy = grid_pressure_1[1:-1,
                         int(interface_coordinate[s-1]) - 2:int(interface_coordinate[s]) - 1] - grid_pressure_1[1:-1, int(
                         interface_coordinate[s-1]) - 1:int(interface_coordinate[s])]
                layer2u_yy1 = grid_pressure_1[1:-1,
                          int(interface_coordinate[s-1]) - 1:int(interface_coordinate[s])] - grid_pressure_1[1:-1, int(
                         interface_coordinate[s-1]):int(interface_coordinate[s]) + 1]
                grid_pressure[1:-1, int(interface_coordinate[s-1]) - 1:int(interface_coordinate[s])] = 2 * grid_pressure_1[
                                                                                                     1:-1, int(interface_coordinate[s-1]) - 1:int(interface_coordinate[s])] - grid_pressure_2[1:-1,
                                                                             int(interface_coordinate[s-1]) - 1:int(
                                                                                 interface_coordinate[s])] + \
                                                                                                 layer_velocity[
                                                                                                     s] ** 2 * pre ** 2 * (
                                                                                                             layer2u_xx - layer2u_xx1) + \
                                                                                                 layer_velocity[
                                                                                                     s] ** 2 * pre1 ** 2 * (
                                                                                                             layer2u_yy - layer2u_yy1)
            # for last layer :
            layer3u_xx = grid_pressure_1[:-2, int(interface_coordinate[len(layer_velocity)-2]) - 1:-1] - grid_pressure_1[1:-1, int(
                interface_coordinate[len(layer_velocity)-2]) - 1:-1]
            layer3u_xx1 = grid_pressure_1[1:-1, int(interface_coordinate[len(layer_velocity)-2]) - 1:-1] - grid_pressure_1[2:, int(
                interface_coordinate[len(layer_velocity)-2]) - 1:-1]
            layer3u_yy = grid_pressure_1[1:-1, int(interface_coordinate[len(layer_velocity)-2]) - 2:-2] - grid_pressure_1[1:-1, int(
                interface_coordinate[len(layer_velocity)-2]) - 1:-1]
            layer3u_yy1 = grid_pressure_1[1:-1, int(interface_coordinate[len(layer_velocity)-2]) - 1:-1] - grid_pressure_1[1:-1,
                                                                                       int(interface_coordinate[len(layer_velocity)-2]):]
            grid_pressure[1:-1, int(interface_coordinate[len(layer_velocity)-2]) - 1:-1] = 2 * grid_pressure_1[1:-1, int(
                interface_coordinate[len(layer_velocity)-2]) - 1:-1] - grid_pressure_2[1:-1, int(interface_coordinate[len(layer_velocity)-2]) - 1:-1] + \
                                                                       layer_velocity[len(layer_velocity)-1] ** 2 * pre ** 2 * (
                                                                                   layer3u_xx - layer3u_xx1) + \
                                                                       layer_velocity[len(layer_velocity)-1] ** 2 * pre1 ** 2 * (
                                                                                   layer3u_yy - layer3u_yy1)


        # scalar codes with for loops
        else:

            for k in range(1, J - 1):
                for j in range(1, n - 1):

                    # model varying velocity FDTD
                    if vary_velocityflag == 1:
                        pre = delta_t / (delta_x)
                        pre1 = delta_t / (delta_h)
                        # discretized velocity at three consecutive time instants
                        vel_i = velocity[j][k]
                        vel_next = velocity[j + 1][k]
                        vel_pre = velocity[j - 1][k]
                        vel_i1 = velocity[j][k]
                        vel_next1 = velocity[j][k + 1]
                        vel_pre1 = velocity[j][k - 1]

                        grid_pressure[j][k] = pre ** 2 * (0.5 * (vel_i + vel_next) * (
                                    grid_pressure_1[j + 1][k] - grid_pressure_1[j][k]) - 0.5 * (vel_i + vel_pre) * (
                                                                      grid_pressure_1[j][k] - grid_pressure_1[j - 1][
                                                                  k])) + 2 * grid_pressure_1[j][k] - grid_pressure_2[j][
                                                  k] + pre1 ** 2 * (0.5 * (vel_i1 + vel_next1) * (
                                    grid_pressure_1[j][k + 1] - grid_pressure_1[j][k]) - 0.5 * (vel_i1 + vel_pre1) * (
                                                                                grid_pressure_1[j][k] -
                                                                                grid_pressure_1[j][k - 1]))

        # implement Dirchlet boundary condition
        # here  a mur second order absorbing boundary condition is implemented to avoid reflection of wave at artificial
        # vertical boundary

        mur_flag = 1
        neumann = 1
        for f in range(J):
            # CHECK if it is ground  boundary points
            # mur second order absorbing boundary condition
            # pre constant w
            if mur_flag == 1:
                if f != J - 1:
                    # apply nonlinear weighting
                    w = (np.sqrt(velocity[0][f]) * delta_t - delta_x) / (np.sqrt(velocity[0][f]) * delta_t + delta_x)
                    grid_pressure[0][f] = grid_pressure_1[1][f] + w * grid_pressure[1][f] - w * grid_pressure_1[0][f]
                    w1 = (np.sqrt(velocity[n - 1][f]) * delta_t - delta_x) / (
                                np.sqrt(velocity[n - 1][f]) * delta_t + delta_x)
                    grid_pressure[n - 1][f] = grid_pressure_1[n - 2][f] + w1 * grid_pressure[n - 2][f] - w1 * \
                                              grid_pressure_1[n - 1][f]

                    # or implement Dirchelet condition
            elif neumann == 0:
                if f != J - 1:
                    grid_pressure_1[0][f] = 0
                    grid_pressure[0][f] = 0
                    grid_pressure_2[0][f] = 0

                    grid_pressure_1[n - 1][f] = 0
                    grid_pressure[n - 1][f] = 0
                    grid_pressure_2[n - 1][f] = 0
                    # implement Neuman condition to right and left boundary
            if neumann == 1 and mur_flag == 0 and f != 0 and f != J - 1:
                vel_i = velocity[0][f]

                vel_next = velocity[1][f]

                vel_pre = velocity[1][f]
                vel_i1 = velocity[0][f + 1]
                vel_next1 = velocity[0][f]
                vel_pre1 = velocity[0][f - 1]
                grid_pressure[0][f] = pre ** 2 * (
                            0.5 * (vel_i + vel_next) * (grid_pressure_1[1][f] - grid_pressure_1[0][f]) - 0.5 * (
                                vel_i + vel_pre) * (grid_pressure_1[0][f] - grid_pressure_1[1][f])) + 2 * \
                                      grid_pressure_1[0][f] - grid_pressure_2[0][f] + pre1 ** 2 * (
                                                  0.5 * (vel_i1 + vel_next1) * (
                                                      grid_pressure_1[0][f + 1] - grid_pressure_1[0][f]) - 0.5 * (
                                                              vel_i1 + vel_pre1) * (
                                                              grid_pressure_1[0][f] - grid_pressure_1[0][f - 1]))
                grid_pressure[n - 1][f] = pre ** 2 * (
                            0.5 * (vel_i + vel_next) * (grid_pressure_1[n - 2][f] - grid_pressure_1[n - 1][f]) - 0.5 * (
                                vel_i + vel_pre) * (grid_pressure_1[n - 1][f] - grid_pressure_1[n - 2][f])) + 2 * \
                                          grid_pressure_1[n - 1][f] - grid_pressure_2[n - 1][f] + pre1 ** 2 * (
                                                      0.5 * (vel_i1 + vel_next1) * (
                                                          grid_pressure_1[n - 1][f + 1] - grid_pressure_1[n - 1][
                                                      f]) - 0.5 * (vel_i1 + vel_pre1) * (
                                                                  grid_pressure_1[n - 1][f] - grid_pressure_1[n - 1][
                                                              f - 1]))

            ##wave displacement in the bottom boundary points with neumann condition
            v_ib = velocity[j][J - 1]

            v_nextb = velocity[j + 1][J - 1]

            v_preb = velocity[j - 1][J - 1]
            v_i1b = velocity[j][J - 2]
            v_next1b = velocity[j][J - 1]
            v_pre1b = velocity[j][J - 2]

            grid_pressure[j][J - 1] = 0.5 * (pre ** 2 * (
                        0.5 * (v_ib + v_nextb) * (grid_pressure_1[j + 1][J - 1] - grid_pressure_1[j][J - 1]) - 0.5 * (
                            v_ib + v_preb) * (
                                    grid_pressure_1[j][J - 1] - grid_pressure_1[j - 1][J - 1])) + pre1 ** 2 * (
                                                         0.5 * (v_i1b + v_next1b) * (
                                                             grid_pressure_1[j][J - 2] - grid_pressure_1[j][
                                                         J - 1]) - 0.5 * (v_i1b + v_pre1b) * (
                                                                     grid_pressure_1[j][J - 1] - grid_pressure_1[j][
                                                                 J - 2]))) + grid_pressure_1[j][J - 1]

        # restore all wave displacement for same x location-delta_t**2*np.array([Ricker_response(fm,(i)*delta_t),Ricker_response(fm,(i)*delta_t)])+
        for j in range(n - 1):

            neumann_flag = 0
            if neumann_flag == 1:
                # inject a ricker wavelet source(1/wave_velocity)**2*+2*grid_pressure_1[0][0]+grid_pressure_2[0][0]-delta_t**2*
                # calculate displacement at receivers
                # discretized velocity at three consecutive time instants
                v_i = velocity[j][0]

                v_next = velocity[j + 1][0]

                v_pre = velocity[j - 1][0]
                v_i1 = velocity[j][0]
                v_next1 = velocity[j][1]
                v_pre1 = velocity[j][1]
                grid_pressure[j][0] = 2 * grid_pressure_1[j][0] - grid_pressure_2[j][0] + 0.5 * (pre ** 2 * (
                            0.5 * (v_i + v_next) * (grid_pressure_1[j + 1][0] - grid_pressure_1[j][0]) - 0.5 * (
                                v_i + v_pre) * (grid_pressure_1[j][0] - grid_pressure_1[j - 1][0])) + pre1 ** 2 * (
                                                                                                             0.5 * (
                                                                                                                 v_i1 + v_next1) * (
                                                                                                                         grid_pressure_1[
                                                                                                                             j][
                                                                                                                             1] -
                                                                                                                         grid_pressure_1[
                                                                                                                             j][
                                                                                                                             0]) - 0.5 * (
                                                                                                                         v_i1 + v_pre1) * (
                                                                                                                         grid_pressure_1[
                                                                                                                             j][
                                                                                                                             0] -
                                                                                                                         grid_pressure_1[
                                                                                                                             j][
                                                                                                                             1])))

            # calculate maximum at right boundary
            max_val = np.array(grid_pressure[n - 1][:]).max()

            # Mur ABC for horizontal boundary solve for measurement at receiver and bottom surface\
            if neumann_flag == 0:
                wb = (np.sqrt(velocity[j][0]) * delta_t - delta_h) / (np.sqrt(velocity[j][0]) * delta_t + delta_h)
                if j != source_location:
                    grid_pressure[j][0] = grid_pressure_1[j][1] + wb * grid_pressure[j][1] - wb * grid_pressure_1[j][0]

            # record receiver measurement for single wave component
            if j > source_location:
                seismic_data = [np.array(grid_pressure[j][0][0]), np.array(grid_pressure[source_location][0][0]),
                                (i + 1) * delta_t]
                seismic_trace.append(seismic_data)

        # STACK TO GET grid pressure for different time steps
        time_grid_pressure.append(np.array(grid_pressure))
        # switch variables for next time step
        grid_pressure_2[:][:], grid_pressure_1[:][:] = grid_pressure_1[:][:], grid_pressure[:][:]
    return seismic_trace, grid_pressure, time_grid_pressure, time_stamps,interface_coordinate



