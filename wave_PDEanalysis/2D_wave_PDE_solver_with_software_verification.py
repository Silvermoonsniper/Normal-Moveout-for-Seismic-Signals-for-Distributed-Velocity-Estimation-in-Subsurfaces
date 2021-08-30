# solve seismic wave propagation in 2D and investigate how time resolution and space resolution influence numerical solution of wave
# PDE from FDTD
# use different acoustic wave velocities.Consider a source-receiver framework
# select a given region and discretize over it, we only consider seismic body waves
# in isotropic, homogeneous elastic media.
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# discretize the distance between source and receiver
# the number of grids
n = 40
J = 40
# the distance between source and receiver (unit:m)
distance = 2e2
delta_x = distance / n
x = np.linspace(0, distance, num=n)
# depth of one layer
depth = 100
# time rounds for simulation
time_rounds = 180
# total time for simulation (unit:s)
time = 0.3
# dicretize sampling time
delta_t = time / time_rounds
# peak frequency
fm = 0.2


# this function aims to calculate seismic_wave velocity_generator

# here we calculate amplitude of ricker wavelet at given time t
def Ricker_response(fm, t):
    # convert frequency into radians
    fm_radian = 2 * np.pi * fm
    amplitude_value = (1 - 0.5 * fm_radian ** 2 * t ** 2) * (np.exp((-0.25 * fm_radian ** 2 * t ** 2)))
    return amplitude_value


# this function aims to get initial shape of seismic wave, here we could test for different initial
# wave displacement shape

def initial(b, h, L):
    A = 2
    # all constant
    constant_flag = 0
    layer_case = 0
    sine_flag = 1
    # depth from ground for each layer
    depth = np.linspace(0, 3e3, num=5)
    # flag for switching between test mode and real mode
    # test mode: use manual constructed ground truth and source function to test numerical approximation quality
    # real mode: wave motion with given initila wavefield
    source_testflag = 1
    if constant_flag == 1:
        initial_value = A
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


# function is solving for numerical approximation of sound pressure
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
    distance1 = 2e2
    largest_depth = 3000
    # acoustic wave phase velocity
    wave_velocity = 1 / (np.sqrt((nambda + 2 * mu) / rho))

    # solve for source term
    source_value = -(2 * x * (1 - x) * (h) / wave_velocity ** 2) - 2 * t ** 2 * h
    return source_value


# construct an arbitrary solution of wave displacement for software verification
def manual_solution(x, h, t):
    distance1 = 2e2
    largest_depth = 3000
    # solutionvalue=x*(np.array((distance1)-x))*h*(np.array((largest_depth)-h))*(0.5*t+1)
    solutionvalue = x * (1 - x) * (t) ** 2 * (h)
    return solutionvalue


# varying velocity function, assign velocity for each individual mesh point according to layer-cake model
# input parameters: n,J:distance index NUMBER and depth index NUMBER
# depth: the depth of each layer in layered-earth model
# delta_h: depth disrectization length
# initialize depth and velocity of each layer


layer_velocity = np.array([0.7e3, 0.7e3, 0.7e3, 0.7e3, 0.7e3])


# layer_velocity=np.array([0.7e3, 2.3e3,3e3,6e3,7.8e3])
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


# numerical stability analysis
# define stability check function of 2D wave propagagtion
def numerical_stability(vel, delta_t, delta_h, delat_x):
    flag = 1 / (vel * np.sqrt((1 / delta_h ** 2) + (1 / delta_x ** 2)))

    if delta_t <= flag:
        return True
    else:
        return False

    # fdtd solver, input parameters:
    # n: total number of spatial grids in x-direction
    # J: total number of spatial grids in depth of one layer
    # time_rounds: total time discretization number
    # delta_t: time interval for a sample
    # nambda,mu: first and second Lame coefficients
    # rho:linear density of geological material
    # distance: distance between source and receiver
    # depth_max: maximum depth for simulation


# output args:
# seismic_trace_x: measurement date at receiver
# time_grid_pressure: wave amplitude in whole computation domain for all iterations
# time_stamps: time array
# error_array: averaged numerical approximation error for each iteration
# set up an initial wave field region
initial_wavefield_len = np.arange(0, 10)
initial_wavefield_wid = np.arange(0, 10)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=(6, 6))
ax = fig.gca(projection='3d')


def FEM_acoustic_wave(n, J, time_rounds, delta_t, distance, nambda, mu, rho):
    i, j = 1, 1
    # grid interval
    n = int(n)
    J = int(J)
    time_rounds = int(time_rounds)

    delta_x = distance / (n)
    # depth for each layer
    test_depth = np.linspace(0, 3e3, num=5)
    delta_h = test_depth.max() / int(J)

    # initial sound pressure array, with dimension of n J
    grid_pressure = np.zeros([n + 1, J + 1, 2])
    grid_pressure_1 = np.zeros([n + 1, J + 1, 2])
    grid_pressure_2 = np.zeros([n + 1, J + 1, 2])
    # acquire velocity for each mesh point
    # define a space index for receiver locations
    receiver_spaceindex = [n - 10, n - 12, n - 20, n - 27]
    velocity = vary_velocity(test_depth, n, J, delta_h, layer_velocity)

    # check numerical stability
    for i in velocity.flatten():
        stability_result = numerical_stability(np.array(i), delta_t, delta_h, delta_x)
        if stability_result == False:
            print(stability_result)
    velocity = velocity ** 2
    # flag for switching between test mode and real mode
    source_testflag = 1
    # flag for using initial wave zone
    initial_wave_flag = 0
    # initial energy array
    energy_array = []
    # initial sound pressure at time instant 1 and 2 for each grid
    for k in range(n):
        for j in range(J):

            # initialization of wave displacement in x-h plane
            if initial_wave_flag == 0:
                wave_vector = np.array(
                    [initial(k * delta_x, j * delta_h, distance), initial(k * delta_x, j * delta_h, test_depth.max())])
            if initial_wave_flag == 1:
                if k in initial_wavefield_len and j in initial_wavefield_wid:
                    # initialization of wave displacement in x-h plane
                    wave_vector = np.array([initial(k * delta_x, j * delta_h, distance),
                                            initial(k * delta_x, j * delta_h, test_depth.max())])
                else:

                    wave_vector = [0, 0]

            grid_pressure_1[k][j] = wave_vector

            # calculate total energy of initial wavefield

            if source_testflag == 0:
                ax.scatter3D(k * delta_x, j * delta_h, np.linalg.norm(grid_pressure_1[k][j]))
                ax.set_xlabel('x-coorinate:$x_m$')
                ax.set_ylabel('h-coordinate: $h_j$')
                ax.set_zlabel('Wave Displacement u(x,h,t)')
    # plt.title('The Initial Wave Displacement Field')
    energy = 0

    for q in grid_pressure_1:
        energy += np.linalg.norm(q) ** 2
    energy_array.append(energy)
    # for anisotropical media
    vary_velocityflag = 0

    # acoustic wave phase velocity
    wave_velocity = np.sqrt((nambda + 2 * mu) / rho)
    # calcualte coefficient r1,r2
    r1 = (delta_t) / (wave_velocity * delta_x)
    r2 = (delta_t) / (wave_velocity * delta_h)

    # calculate pressure at second time interval for each grid
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
                grid_pressure[p][o] = grid_pressure_1[p][o] + 0.5 * r1 ** 2 * (
                            grid_pressure_1[p + 1][o] - 2 * grid_pressure_1[p][o] + grid_pressure_1[p - 1][
                        o]) + 0.5 * r2 ** 2 * (grid_pressure_1[p][o + 1] - 2 * grid_pressure_1[p][o] +
                                               grid_pressure_1[p][o - 1]) - (
                                                  0.5 / wave_velocity) ** 2 * delta_t ** 2 * source(p * delta_x,
                                                                                                  o * delta_h, delta_t)

            else:
                grid_pressure[p][o] = grid_pressure_1[p][o] + 0.5 * r1 ** 2 * (
                            grid_pressure_1[p + 1][o] - 2 * grid_pressure_1[p][o] + grid_pressure_1[p - 1][
                        o]) + 0.5 * r2 ** 2 * (grid_pressure_1[p][o + 1] - 2 * grid_pressure_1[p][o] +
                                               grid_pressure_1[p][o - 1])

    # switch variables for next time step
    grid_pressure_2[:][:], grid_pressure_1[:][:] = grid_pressure_1, grid_pressure

    ##set Dirchlet boundray condition
    for i in range(n):
        grid_pressure_1[i][J] = 0

        grid_pressure_1[i][0] = 0
    for j in range(J):
        grid_pressure_1[0][j] = 0
        grid_pressure_1[n][j] = 0
    time_grid_pressure = []
    time_stamps = np.zeros([time_rounds, 1])

    # initialize approximation error array
    error_array = []

    error = np.zeros([n + 1, J + 1])
    # initialzie velocity field array
    velo = np.zeros([n, J])
    # grid_pressure[0][0]=[Ricker_response(fm,delta_t*1),Ricker_response(fm,delta_t*1)]
    # propagate pressure in a given grid with grid pressure at previous time instants
    # LOOP over a given simulation time limit
    seismic_trace = []
    # i is time index, where j is position index
    for i in range(2, time_rounds):
        # record simulation time points
        time_stamps[i] = i * delta_t
        energy = 0
        for j in range(1, n):
            for k in range(1, J):

                pre = delta_t / (delta_x)
                pre1 = delta_t / (delta_h)
                # model varying velocity FDTD
                if vary_velocityflag == 1:

                    # discretized velocity at three consecutive time instants
                    vel_i = velocity[j][k]
                    vel_next = velocity[j + 1][k]
                    vel_pre = velocity[j - 1][k]
                    vel_i1 = velocity[j][k]
                    vel_next1 = velocity[j][k + 1]
                    vel_pre1 = velocity[j][k - 1]
                    # calculate displacement
                    grid_pressure[j][k] = pre ** 2 * (
                                0.5 * (vel_i + vel_next) * (grid_pressure_1[j + 1][k] - grid_pressure_1[j][k]) - 0.5 * (
                                    vel_i + vel_pre) * (grid_pressure_1[j][k] - grid_pressure_1[j - 1][k])) + 2 * \
                                          grid_pressure_1[j][k] - grid_pressure_2[j][k] + pre1 ** 2 * (
                                                      0.5 * (vel_i1 + vel_next1) * (
                                                          grid_pressure_1[j][k + 1] - grid_pressure_1[j][k]) - 0.5 * (
                                                                  vel_i1 + vel_pre1) * (
                                                                  grid_pressure_1[j][k] - grid_pressure_1[j][k - 1]))

                    # calculate energy of wavefield
                    energy += np.linalg.norm(grid_pressure[j][k])

                    # calculate velocity at time n

                    velo[j][k] = np.linalg.norm((grid_pressure[j][k] - grid_pressure_1[j][k]) / delta_t)
                    # restore all wave displacement for same x location
                    if j == int(n / 2):
                        # append data for animations
                        seismic_data = [np.array(j * delta_x), np.array(k * delta_h), np.array(grid_pressure[j][k]),
                                        time_stamps[i]]
                        seismic_trace.append(seismic_data)

                # calculate numerical approximation of wave displacement without source term
                if source_testflag == 0 and vary_velocityflag == 0:

                    # recursive formulation to calculate wave form

                    grid_pressure[j][k] = r1 ** 2 * (
                                grid_pressure_1[j + 1][k] - 2 * grid_pressure_1[j][k] + grid_pressure_1[j - 1][
                            k]) + r2 ** 2 * (grid_pressure_1[j][k + 1] - 2 * grid_pressure_1[j][k] + grid_pressure_1[j][
                        k - 1]) + 2 * grid_pressure_1[j][k] - grid_pressure_2[j][k]

                    # restore all wave displacement for same x location
                    if j == int((n / 2)):
                        seismic_data = [np.array(j * delta_x), np.array(k * delta_h), np.array(grid_pressure[j][k]),
                                        time_stamps[i]]
                        seismic_trace.append(seismic_data)

                # calculate numerical approximation of wave displacement with a source term
                elif vary_velocityflag == 0:

                    grid_pressure[j][k] = r1 ** 2 * (
                                grid_pressure_1[j + 1][k] - 2 * grid_pressure_1[j][k] + grid_pressure_1[j - 1][
                            k]) + r2 ** 2 * (grid_pressure_1[j][k + 1] - 2 * grid_pressure_1[j][k] + grid_pressure_1[j][
                        k - 1]) + 2 * grid_pressure_1[j][k] - grid_pressure_2[j][k] - (
                                                      1 / wave_velocity) ** 2 * delta_t ** 2 * source(j * delta_x,
                                                                                                      k * delta_h,
                                                                                                      i * delta_t)

                    # calculate constructed ground truth solution

                    true = [manual_solution((j) * delta_x, (k) * delta_h, i * delta_t),
                            manual_solution((j) * delta_x, (k) * delta_h, i * delta_t)]

                    # calculate error of numerical approximation with FDTD
                    if np.linalg.norm(true) != 0:
                        error[j][k] = (np.linalg.norm((true - grid_pressure[j][k])) / np.linalg.norm(true)) ** 2


                    else:

                        error[j][k] = (np.linalg.norm((true - grid_pressure[j][k]))) ** 2

        error_array.append(error)

        # append total wave energy at different time
        energy_array.append(energy)

        # implement Neumann boundary condition
        for j in range(1, n):
            # wave displacement in the bottom boundary points
            v_ib = velocity[j][J - 1]
            if j + 1 < n:
                v_nextb = velocity[j + 1][J - 1]
            else:
                v_nextb = velocity[j - 2][J - 1]
            v_preb = velocity[j - 1][J - 1]
            v_i1b = velocity[j][J - 2]
            v_next1b = velocity[j][J - 1]
            v_pre1b = velocity[j][J - 2]
            grid_pressure[j][J - 1] = (r1 ** 2 * (
                        grid_pressure_1[j + 1][J - 1] - 2 * grid_pressure_1[j][J - 1] + grid_pressure_1[j - 1][
                    J - 1])) + r2 ** 2 * (grid_pressure_1[j][J - 2] - 2 * grid_pressure_1[j][J - 1] +
                                          grid_pressure_1[j][J - 2]) + grid_pressure_1[j][J - 1]

        # set Dirchlet boundray condition
        for p in range(n + 1):
            grid_pressure[p][J] = p*delta_x * (1 - p*delta_x) * (i*delta_t) ** 2 * (100)

            grid_pressure[p][0] = 0

        for j in range(J + 1):
            grid_pressure[0][j] = 0
            grid_pressure[n][j] = 200 * (1 - 200) * (i*delta_t) ** 2 * (j*delta_h)

        # switch variables for next time step
        grid_pressure_2[:][:], grid_pressure_1[:][:] = grid_pressure_1, grid_pressure
        # STACK TO GET grid pressure for different time steps
        time_grid_pressure.append(np.array(grid_pressure).flatten())

    return seismic_trace, time_grid_pressure, time_stamps, error_array

seismic_trace, time_grid_pressure, time_stamps, error_array=FEM_acoustic_wave(n, J, time_rounds, delta_t, distance,  nambda, mu, rho)
print(np.array(error_array[10].mean()))
#this part we visulaize numerical approximation error via FDTD for different time and space discretizations
#grid number points array
def software_verification_visualization(time,distance,depth,nambda,mu,rho):
    number=np.linspace(80,120,num=5)
    #samples for time level
    number_time=np.linspace(120,160,num=5)
    time=0.3
    average_error=[]
    #the distance between source and receiver (unit:m)
    distance=2e2
    delta_x=distance/n
    for i in number:
    #dicretize sampling time
        for j in number_time:
             #calculate time resolution
            delta_t=time/j
             #solve Wave PDE with different time and space resolution
            seismic_trace_x,time_grid_pressure,time_stamps,error_array=FEM_acoustic_wave(i,i,j,delta_t,distance,depth,nambda,mu,rho)

#append for average normalized approximation error with different time and space resolution
            average_error.append(np.array(error_array[1]).mean())
#reshape the average error array
    average_errorfinal=np.array(average_error).flatten().reshape([len(number),len(number)])

    return average_errorfinal



