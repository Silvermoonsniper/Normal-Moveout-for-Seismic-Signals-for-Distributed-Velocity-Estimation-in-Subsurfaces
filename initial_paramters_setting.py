#initial parameters as dictionary
import numpy as np
#function to assign receiver position
def receiver_position_assignment(distance,receiver_number_oneside):
    receiver_space= 0.5*distance/receiver_number_oneside
    #calculate receiver coordinates for all receivers on one side based on
    # inter-receiver distance
    receiver_coordinate= np.arange(0,0.5*distance,receiver_space)

    return receiver_coordinate[1:]
def initial_parameters_setting():
    intial_args = {
        ######
              #  The following arguments define the whole simulation set up for
               # velocity and depth estimator via NMO with both centralized and distributed case
        ######
    # number of cells in horizontal direction.
        "cell_number": 120,
    # ground truth of propagation velocity and depth of each layer
    # here depth refers to depth calculated from ground surface
    # these subsurfaces parameters could be chosen for a general n layer model
        "layer_velocity": np.array([0.667e3, 1.7e3, 2.2e3]),
    # note: the first element here should always be 0 which is depth at ground surface
    # depth from ground surface for each layer
        "test_depth" : np.array([0, 750, 2200, 2800]) * 2e-2,
    # number of accessible receivers (neighbors in distributed settings)
    # this should always be an even number and should be at least 4
    # due to consideration of communication radius
        "accessible_number" : 4,
    # visualize estimation profile at certain iterations
        "iteration" : [0, 5, 13, 20, 135],
    # SNR for the noisy seismic measurement (dB)
        "SNR" : 5,
    # total time rounds for simulation, unit:(s)
        "delta_t" : 0.25/5000,
    # total simulation time (s)
        "time" : 0.25,
        # peak frequency of ricker wavelet, unit:Hz
        "fm" : 200,

    # length of simulation region (m)
        "distance": 28,


    # number of receivers on one side plus 1, For example, if we place 12 receivers on one side, we assign it to 13. unit:m
        "receiver_number_oneside" : 13,
        # factor to choose new time level by analyzing ground truth arrival time at receivers
        "alpha" : 1,
    # if this flag sets to 'line_topology':
    # set line topology for distributed network
    # if this flag sets to 'random_topology':
    # set random topology for distributed network
        "line_topology_flag": 'line_topology',
    # flag to choose different picking time method
    # 'STA_LTA': we use STA/LTA picker to directly process noisy seismic measurement
    # 'picking_deconvolve_measurement': pick travel time from deconvolved profile
        "picking_travel_time_method" : 'picking_deconvolve_measurement',
        #array for different number of neighbors
        # the minimum in this array should be at least 4, and maximum should
        # be the length of "source_to_receiver_distance" minus 2, due to consideration of
        # line topology
        "neighbor_number" : np.array([4,6,8]),
        ######
            # arguments specify for distributed average conensus algorithm used for NMO
        #####
        # FLAG for noisy link
        "noisy_link": 1,

        # time step for running distributed average consensus
        "DNMO_iteration": 44,
        ######
           # arguments setting for adapt-then-combine framework, if you want to investigate this distributed
           # scheme, you could modify the following args
        ######
    # learning rate
        "yita" : 4e-12,

    # window size (the number of data points in the window) that used in STA/LTA picking method

        "windowsize" : 40,

    # SNR for noisy link (dB)
        "SNR_noisy_link" : 30,
    ######
          #  arguments specify for centralized Normal-moevout without recalculating time picking
    ######

    # flag to display wave propagation video
        # flag sets to "display_video": display video
        # flag sets to "display_wave_pattern": display wave propagation snapshot at certain iteration
        "video_flag": "display_wave_pattern",
    # iteration of wave pattern that you want to plot
        "time_instant": 280,
    # specify which receiver that we wanna to plot for seismic measurement
    # and reflectivity series, for example, 3 means 3rd receiver
        "single_plot_indice": 3,


       
        # flag to only plot seismic measurement for single receiver
        # set both flags to 1 to plot single receiver measurement
        # if plot_flag==1 and single_receiver = 0: plot multiple receiver response
        # else both flags are 0: don't plot
        "single_receiver" : 1,
        # flag to plot single seismic measurement
        "plot_flag": 1,


        ########
            # flags that define different kinds of plotting
        ########
                      # flag for plots for different analysis in NMO
                      # flag documentations:
                      # vel_flag: 1: plot centralized Normal moveout estimation result
                      # vel_flag1: 1：plot distributed normal moveout estimation results
                      #
                      # the following two flags should be operated after documented the code at the recalculating
                      # picking time in function: "NMO_estimation_performance_distributed.vel_depth_estimator()"

                      # pattern_analzye_flag: plot root-mean square velocity estimation curve analysis
                      # osicillation_pattern: plot how picking time influence the estimation in classic normal moveout
                      # without recalculating picking time
        "vel_flag":0,
        "vel_flag1":0,
        "pattern_analzye_flag":1,
        "osicillation_pattern":1

    }
    return intial_args
