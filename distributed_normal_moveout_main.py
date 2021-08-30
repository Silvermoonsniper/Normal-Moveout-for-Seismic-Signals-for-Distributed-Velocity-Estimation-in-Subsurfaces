


"""""""""
main method entrance, here is program entrance that we are able to switch between 
different algorithms.
"""""""""
from Class_Centralized_NMO import Centralized_NMO_estimator
from Class_DNMO import Distributed_Normal_moveout_estimator
from distributed_nmo_real_time_estimation_error_plot import plot_scheme_comparison, plot_topology_comparison, \
    plot_sensornumber_comparison
from initial_paramters_setting import initial_parameters_setting
from wave2D import numerical_stability
import numpy as np
if __name__ == '__main__':
        #parameters setting
        initial_args = initial_parameters_setting()


        #here user could modify the parameters from dictionary "initial_args" to
        # test for different subsurface models
        # example, write command as:
        # initial_args["source_to_receiver_distance"]=np.array([2,4,6,7,8])
        # this will set distance of receivers on one side to the shot source as
        # [2,4,6,7,8] meters respectively
        #  initial_args["SNR"] = 5 will set SNR for noisy seismic trace as 5dB
        #Call the class of distributed NMO and Centralized NMO
        R=Distributed_Normal_moveout_estimator(initial_args)

        C=Centralized_NMO_estimator(initial_args)
        # first perform a checking on Courant number condition, if it is not fulfilled, automacally choose a
        # new time resolution level which fulfills condition
        status = numerical_stability(initial_args["layer_velocity"].max(), initial_args["delta_t"],
                                     R.delta_x, R.delta_x)

        if status == False:
            # new time resolution
            initial_args["delta_t"] = 0.9 / (initial_args["layer_velocity"].max() * (
                        (1 / R.delta_x) + (1 / R.delta_x)))

        # only to one kind of simulation once,
        # therefore: when NMO_algorithm_flag is chosen, set "compare_analysis_flag" to 0
        # and another way around
        #Different STRINGS to switch between different algorithms
        # 'centralized_nmo': apply centralized NMO
        # 'distributed_nmo_average_consensus': apply distributed NMO with average_consensusand also compare with centralized nmo
        # 'distributed_ATC': apply distributed NMO with Adapt-then-combine
        NMO_algorithm_flag = 'centralized_nmo'
        #string to perform comparing analysis about
        # different topology for distributed NMO with average consensus
        # or different distributed NMO schemes
        # 'topology_flag': set this to compare different topology in distributed average consensus
        # 'scheme_flag': set this to compare two distributed NMO schems with
        #  same travel time picking
        compare_analysis_flag = 0

        if  compare_analysis_flag == "sensor_number":
            # this line should not be changed

            travel_time, t0_estimate, para_estimate = [], [], []
            peak, finaltime, v_error_array, d_error_array, v_error_array_cen, d_error_array_cen = R.distributed_nmo_main(
                compare_analysis_flag, travel_time, t0_estimate, para_estimate)
            initial_args["receiver_number_oneside"] = 7
            R = Distributed_Normal_moveout_estimator(initial_args)

            # this line should not be changed
            travel_time, t0_estimate, para_estimate = [], [], []
            peak, finaltime, v_error_array1, d_error_array1, v_error_array_cen1, d_error_array_cen1= R.distributed_nmo_main(
                compare_analysis_flag, travel_time, t0_estimate, para_estimate)
            plot_sensornumber_comparison(v_error_array, d_error_array, v_error_array_cen, d_error_array_cen,
                                         v_error_array1, d_error_array1, v_error_array_cen1, d_error_array_cen1)
        # call main function for distributed ATC-framework to estimate depth and layer velocity
        # this step is done to compare real-time estimation performance of two distributed NMO schemes
        if NMO_algorithm_flag == 'distributed_ATC'  or compare_analysis_flag == 'scheme_flag':
            # Set total time rounds for this scheme
            R.time_rounds = 5000
            # set time resolution for distributed scheme
            R.delta_t = R.time / R.time_rounds

            t0_array, m0_array, travel_time, local_error, local_velocity,t0_estimate, para_estimate = R.distributed_gradient_descent()
        # plot picking travel time and ground truth travel time for single layer
        # picking_plot_comparison(travel_time, R.receiver_distance, finaltime)

        #Call main function of distributed Normal-moevout with aveage consensus to estimate
        #layer velocity and depth, also centralzied normal-moveout scheme with
        # recalculation of picking time is also incorporated here

        elif NMO_algorithm_flag == 'distributed_nmo_average_consensus':

           #  R.iteration= [0, 5, 13, 36]
             #this line should not be changed
             travel_time,t0_estimate, para_estimate=[],[],[]
             peak,finaltime,v_error_array, d_error_array,v_error_array_cen,d_error_array_cen=R.distributed_nmo_main(compare_analysis_flag,travel_time,t0_estimate, para_estimate)

        #call main function to apply centralized Normal-moveout without recalculating travel time
        elif NMO_algorithm_flag == 'centralized_nmo':
             C.centralized_nmo_main_function()
        #plot comparison of estimation error with different normal-moevout schemes
        if compare_analysis_flag == 'scheme_flag':
            R.iteration = [0, 5, 13, 20,40,50,60]
            #distributed NMO with average consensus
            peak, finaltime, v_error_array, d_error_array, v_error_array_cen, d_error_array_cen = R.distributed_nmo_main(
                compare_analysis_flag, travel_time,t0_estimate, para_estimate)
            #plot comparison of estimation performance under same picking with STA/LTA picker
            plot_scheme_comparison(v_error_array, d_error_array, v_error_array_cen, d_error_array_cen, local_error,
                               local_velocity)

        #plot comparison of distributed NMO with average consensus under different neighbor topology
        elif compare_analysis_flag == 'topology_flag':
            # this line should not be changed
            travel_time=[]
            #for line topology
            R.line_topology_flag= 'line_topology'
            peak, finaltime, v_error_array, d_error_array, v_error_array_cen, d_error_array_cen = R.distributed_nmo_main(compare_analysis_flag,travel_time)
            #for random topology
            R.line_topology_flag = 'random_topology'
            peak, finaltime, v_error_array_1, d_error_array_1, v_error_array_cen, d_error_array_cen = R.distributed_nmo_main(compare_analysis_flag,travel_time)
            #plot comparison
            plot_topology_comparison(v_error_array, d_error_array, v_error_array_cen, d_error_array_cen, d_error_array_1,
                                     v_error_array_1)
