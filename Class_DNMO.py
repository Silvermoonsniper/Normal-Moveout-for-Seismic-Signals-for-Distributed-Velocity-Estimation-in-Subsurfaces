import numpy as np
import matplotlib.pyplot as plt
import STA_LTA_picking_traveltime


from Hessian_determinant_plot import Hessian_determinant_plot
from STA_picking_plot import picking_plot_comparison
from distributed_nmo_real_time_estimation_error_plot import plot_ATC_estimation_performance
from estimated_parameters_distributed_ATC import estimated_parameters_plot

from estimated_parameters_variation import estimated_parameters_variation
from initial_paramters_setting import receiver_position_assignment

from main_algorithms import centralized_NMO
from main_algorithms.distributed_gradient_descent_ATC import neighbours_generator, initial_estimate_parameters,adapt_then_combine_optimizer
from main_algorithms.velocity_depthestiamtor_with_ATC_framework import vel_depth_solver_ATC_iterations
from neighbour_number_Influence import plot_neighbour_number_Influence

from synthetic_seismic_measurement_data_generation import ground_truth_arrival_time_generator
from synthetic_seismic_measurement_data_generation import wave2D
from synthetic_seismic_measurement_data_generation import seismic_trace_extraction
from seismic_measurement_postprocessing import noisy_measurement_generator

from seismic_measurement_postprocessing import time_resolution
from seismic_measurement_postprocessing import seismic_trace_deconvolution
from seismic_measurement_postprocessing import travel_time_picking
from main_algorithms import distributed_average_consensus
from main_algorithms import  distributed_nonlinea_least_square_fitting
from plot_functions import distributed_nmo_real_time_estimation_error_plot
from plot_functions import local_estimation_performance_visualization_distributedNMO
from plot_functions import NMO_estimation_performance_distributed
from plot_functions import distributed_normal_moveout_estimation_error

### initialize parameters setting



class Distributed_Normal_moveout_estimator():
    #initialization of parameters
    def __init__(self,initial_args):
        for key in initial_args:
            setattr(self, key, initial_args[key])
        #calculate receiver position on one side
        self.source_to_receiver_distance=receiver_position_assignment(initial_args["distance"], initial_args["receiver_number_oneside"])
        # calculate thickness of each layer
        self.depth = initial_args["test_depth"][1:] - initial_args["test_depth"][0:len(initial_args["test_depth"]) - 1]
        # number of layers
        self.layer_n = len(initial_args["layer_velocity"])
        # total time rounds for simulation
        self.time_rounds = int(initial_args["time"] / initial_args["delta_t"])
        # coordinate of source on the ground surface (m), always located at center of ground surface
        self.source_coordinate_x = 0.5*initial_args["distance"]
        #size of cell in FDTD
        self.delta_x=initial_args["distance"]/initial_args["cell_number"]
        # receiver constellation array
        self.n = initial_args["distance"] / self.delta_x
        #indice of receivers on one side
        self.trace_number=np.array(self.source_to_receiver_distance/self.delta_x).astype(int)
        # the discretization level in y vertical direction (unit:m)
        self.delta_h= self.delta_x
        # NUMBER of receivers on one side
        self.receiver_number = len(self.trace_number)
        # the distance between each receiver to the shot source position
        self.receiver_distance = (self.trace_number + 1) * np.array(self.delta_x)
        # total time for plot ground truth series (unit:s)
        # this argument is usually chosen as same as simulation time of wave visualization
        self.time1 = initial_args["time"]
        # time rounds for simulation to plot ground truth arrival time series
        # ground truth time series is sparse series which only has nonzero entries on ground
        # truth time point, else 0. The choice of this parameter depends on how accurate you
        # want to located discretized ground truth arrival time in plot, usually 100000 is sufficient.
        self.time_rounds1 = 1e5
        # time resolution in ground truth series
        self.delta_t1 = self.time1 / self.time_rounds1
        # time array for plotting ground truth sparse series
        self.time_array1 = np.linspace(1, self.time_rounds1, num=self.time_rounds1) * self.delta_t1
        '''''''''''
                following two ground truth parameters would not influence code 
                running, but for comparison with estimated parameters
        '''''''''''
        # ground truth t0
        self.t0_truth = centralized_NMO.t0_solver(initial_args["test_depth"], initial_args["layer_velocity"])

        # calculate ground truth of parameter slowness m0 for each layer
        self.v_rms, self.oneway = centralized_NMO.rms_velocity(self.t0_truth, initial_args["layer_velocity"])
        self.m0_truth = 1 / np.array(self.v_rms) ** 2
        # indice of source cell
        self.source_indice = int(self.source_coordinate_x / self.delta_x)

        # receiver offset for all cells on one side in the ground surface
        self.receiver_spaceindex = np.arange(self.source_indice + 1, self.n - 1) - self.source_indice
        self.receiver = self.receiver_spaceindex * np.array(self.delta_x)
        # indice of cells on one side of computational region

        self.allreceiver_number = self.source_indice - 2
        print(self.trace_number)
    """""""""""
    Noisy seismic measurement data generation
    """""""""""
    # function to generate noisy seismic trace
    #output args:
    #          true: clean seismic trace
    #          time_grid_pressure: array contains wave amplitude in x-direction at all iterations
    #          standard_deviation_array: stanard deviation of AWGN for each seismic measurement at receivers
    #          noisy_final: noisy seismic measurement at receivers
    #          filter_trace: filtered seismic measurement via wiener filter at receivers
    #          post_processing: noiseless synthetic seismic measurement after using matched filter
    #          post_processing1: noiseless synthetic seismic measurement
    #          time_array: time series for plotting
    #          ricker_amp: ricker wavelet source
    #          newdesired: shifted ground truth series
    #          finaltime: ground truth arrival time at receivers
    #          newsynthetic_arriavl: ground truth arrival time for all ground cells on one side of source
    #          newsynthetic_arriavl_1: reshaped ground truth arrival time for all ground cells on one side of source


    def noisy_seismic_measurement_generation(self):

        # generate ground truth arrival time data from multiple receivers for each layer
        desired, indexarray, newsynthetic_arriavl = ground_truth_arrival_time_generator.real_signal(self.layer_n,
                                                                                                    self.layer_velocity,
                                                                                                    self.depth, self.receiver,
                                                                                                    self.delta_t1,
                                                                                                    self.time_array1)

        # generate synthetic measurement by solving wave equation
        # solve wave equation with ricker wavelet source with FDTD method
        seismic_trace, grid_pressure, time_grid_pressure, time_stamps,interface_coordinate = wave2D.Arrivaltime_estimation(self.test_depth,
                                                                                                      self.layer_velocity,
                                                                                                      self.delta_x,
                                                                                                      self.delta_h,
                                                                                                      self.time_rounds,
                                                                                                      self.delta_t,
                                                                                                      self.distance, self.fm,
                                                                                                      self.receiver_spaceindex,
                                                                                                      self.source_coordinate_x)

        # proceed to extract synthetic seismic measurement at receivers
        post_processing, post_processing1, cross, time_array, ricker_amp, newdesired = seismic_trace_extraction.plot_response(
            self.plot_flag,
            seismic_trace, self.time,
            self.time_rounds,
            self.time_rounds1, self.time1,
            desired,
            self.single_receiver)
        # extract noiseless seismic measurement at different receiver denoted by trace_number
        target = noisy_measurement_generator.target_trace(post_processing1, self.trace_number)

        # add AWGN to measurement
        noisy_final, filter_trace, true,standard_deviation_array = noisy_measurement_generator.multiple_noisy_trace(target,
                                                                                                               self.delta_t,
                                                                                                               self.SNR)

        # capture ground truth arrival time and receiver offset in sensor network

        finaltime, newsynthetic_arriavl_1 = time_resolution.ground_truth_selector_certainreceivers(self.trace_number,
                                                                                                   newsynthetic_arriavl,
                                                                                                   self.allreceiver_number,
                                                                                                   self.layer_n)
        # return variables
        return true,time_grid_pressure,standard_deviation_array, noisy_final, filter_trace, post_processing, post_processing1, time_array, ricker_amp, newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1
    """""""""""
    Picking travel time of reflection wave at receivers    
    """""""""""
    #input args:
    #          true: clean seismic trace
    #          time_grid_pressure: array contains wave amplitude in x-direction at all iterations
    #          standard_deviation_array: stanard deviation of AWGN for each seismic measurement at receivers
    #          noisy_final: noisy seismic measurement at receivers
    #          filter_trace: filtered seismic measurement via wiener filter at receivers
    #          post_processing: noiseless synthetic seismic measurement after using matched filter
    #          post_processing1: noiseless synthetic seismic measurement
    #          time_array: time series for plotting
    #          ricker_amp: ricker wavelet source
    #          newdesired: shifted ground truth series
    #          finaltime: ground truth arrival time at receivers
    #          newsynthetic_arriavl: ground truth arrival time for all ground cells on one side of source
    #          newsynthetic_arriavl_1: reshaped ground truth arrival time for all ground cells on one side of source

    #output args:
    #          peak: picking travel time at receivers
    #          reflectivity: reflectivity series at receivers
    #          peak_source: shifted interval of reflectivity series

    def travel_time_picking_main(self,standard_deviation_array, noisy_final, post_processing1, time_array, ricker_amp,
                                 newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1):

        # choose the picking method used to get measured travel time of reflections at receivers
        if self.picking_travel_time_method == 'STA_LTA':
            peak = STA_LTA_picking_traveltime.picking_time_noisy_trace_STALTA(self.time, self.time_rounds, self.windowsize,
                                                                              self.trace_number, newsynthetic_arriavl,
                                                                              newdesired, post_processing1, self.SNR,
                                                                              time_array,
                                                                              self.time_array1, self.layer_n)
            # transpose measured travel time for further processing
            peak = np.array(peak).T
            reflectivity=0,
            peak_source=0
        # if we pick travel time from deconvolved profile
        elif self.picking_travel_time_method == 'picking_deconvolve_measurement':
            # choose new time level
            new_delta_t, new_time_array = time_resolution.time_resolution_generator(self.alpha,
                                                                                    time_array[
                                                                                    0:self.time_rounds - 1000].max(),
                                                                                    time_array[0], finaltime)
            # perform deconvolution for noisy trace to denoisying and release burden on picking travel time
            reflectivity, peak_source = seismic_trace_deconvolution.multiple_reflectivity_retriever(self.time_rounds,
                                                                                                    noisy_final,
                                                                                                    self.fm,
                                                                                                    ricker_amp,
                                                                                                    post_processing1,
                                                                                                    self.trace_number,
                                                                                                    standard_deviation_array,
                                                                                                    self.delta_t1)
            # interpolate to get new deconvolved reflectivity series, this step is done for the case if time resolution is
            # quite small, and may not have sufficient memory to allocate numpy array in the deconvolution stage

            reflectivity_inter, new_delta_t = time_resolution.interpolation(new_time_array, time_array, reflectivity)
            # if no sufficient memory to allocate numpy array in the deconvolution stage use following codes and document next code command on picking travel time
            peak = travel_time_picking.arrival_time_picking(reflectivity_inter, self.trace_number, new_delta_t,
                                                            new_time_array,
                                                            newsynthetic_arriavl_1)

            # pick arrival time data for NMO by distinguishing local maxima peaks in the deconvolved profile
           # peak = travel_time_picking.arrival_time_picking(reflectivity, self.trace_number, self.delta_t,
            #                                                time_array,
             #                                               newsynthetic_arriavl_1)
            # return picking travel time, reflectivity series,
        return peak,reflectivity,peak_source
    """""""""""
    function to investigate how number of neighbors influence estimation in distributed NMO
    """""""""""
    #input args:
    #          neighbor_number: array for different number of neighbors
    #          peak: picking travel time at recievers
    #          finaltime: ground truth arrival time at receivers
    def neighbor_number_influence(self,neighbor_number,peak,finaltime):
        #array to store estimation error for different number of neighbors
        estimation_error_depth,estimation_error_velocity=[],[]
        for i in neighbor_number:
            # gather travel time and sensor position from neighbours and perform nonlinear-least-square fitting
            t0_estimate, para_estimate, accessible_sensor = distributed_nonlinea_least_square_fitting.fitting_sensor_networking(
                self.trace_number, self.line_topology_flag, self.delta_x, self.receiver_number, self.layer_n,
                np.array(peak), i)
            # average consensus on parameters
            consensus_t0, consenus_para, local_information, local_information1 = distributed_average_consensus.t0_average_consensus(
                self.DNMO_iteration, self.t0_truth, self.m0_truth, self.layer_n, 0, 0, self.SNR_noisy_link,
                self.receiver_number, t0_estimate, para_estimate, accessible_sensor)
            # if we don't apply dirls algorithm
            consensus_t01, consenus_para1, local_information11, local_information111 = distributed_average_consensus.t0_average_consensus(
                self.DNMO_iteration, self.t0_truth, self.m0_truth, self.layer_n, 0, 0, self.SNR_noisy_link,
                self.receiver_number, t0_estimate, para_estimate, accessible_sensor)
            # implement normal moveout
            #set flags that don't repeat plots of NMO
            self.vel_flag, self.vel_flag1, self.pattern_analzye_flag, self.osicillation_pattern=0,0,0,0
            newpeak, optimal_time, ground_depth, v_layer, t0coff, t0coffop, ground_depth_dis, v_layer_dis = NMO_estimation_performance_distributed.normal_moveout(
                self.vel_flag, self.vel_flag1, self.pattern_analzye_flag, self.osicillation_pattern, finaltime,
                local_information,
                local_information1, local_information11, local_information111, self.receiver_distance, self.noisy_link,
                consensus_t0,
                consenus_para, peak, self.layer_velocity, self.test_depth, self.layer_n, self.receiver_number)
            # investigate averaged estimated parameters variation
            estimated_parameters_variation(local_information, local_information1, self.t0_truth, self.m0_truth,
                                           self.receiver_distance)
            # if we want to plot real time estimation curve,calculate normalized estimation
            # error for all iterations

            v_error_array, d_error_array, v_error_array_cen, d_error_array_cen, ims = distributed_normal_moveout_estimation_error.distributed_NMO_estimation(
                self.layer_n, self.test_depth, self.layer_velocity, ground_depth, v_layer, self.receiver_distance,
                finaltime,
                local_information, local_information1, self.receiver_number, peak, t0_estimate, para_estimate,
                accessible_sensor)
            estimation_error_depth.append(d_error_array)
            estimation_error_velocity.append(v_error_array)
        #plot estimation error curve for different number of neighbors
        plot_neighbour_number_Influence(self.neighbor_number,estimation_error_depth,estimation_error_velocity,v_error_array_cen, d_error_array_cen)
    '''''''''''
    # main function to implement distributed Normal moveout algorithm with average consensus
    and compare it with centralized Normal moveout algorithm 
    '''''''''''
    #input args:
    #         compare_analysis_flag: string to identify if we do comparison between two distributed
    #         NMO schemes.
    #         travel_time: picking travel time at receivers
    #output args:
    #         peak:  picking travel time at receivers
    #         finaltime: ground truth arrival time at receivers
    #         v_error_array: normalized average velocity estimation error for all iterations with distributed NMO
    #         d_error_array: normalized average depth estimation error for all iterations with distributed NMO
    #         v_error_array_cen: normalized average velocity estimation error for all iterations with centralized NMO
    #         d_error_array_cen: normalized average velocity estimation error for all iterations with centralized NMO


    def distributed_nmo_main(self,compare_analysis_flag,travel_time,t0_estimateAverage, para_estimateAverage):

        # we we compare different distributed NMO scheme, don't need to generate measurement
        # but only use same picking in the ATC framework to accelerate code
        if compare_analysis_flag != 'scheme_flag':
            #generate noisy synthetic seismic measurement at receivers
            true,time_grid_pressure,standard_deviation_array,noisy_final, filter_trace, post_processing, post_processing1, time_array, ricker_amp, newdesired, finaltime, newsynthetic_arriavl,newsynthetic_arriavl_1=self.noisy_seismic_measurement_generation()
        #picking travel time after processing seismic measurement with deconvolution or using STA/LTA picker
            peak,reflectivity_inter,peak_source=self.travel_time_picking_main(standard_deviation_array, noisy_final, post_processing1, time_array, ricker_amp,
                                 newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1)

           #plot picking travel time
            picking_plot_comparison(peak, self.receiver_distance, finaltime)
        #if we want to compare performance of two Distributed NMO, we need to use same picking
        if compare_analysis_flag == 'scheme_flag':
            # generate ground truth arrival time data from multiple receivers for each layer
            desired, indexarray, newsynthetic_arriavl = ground_truth_arrival_time_generator.real_signal(self.layer_n,
                                                                                                        self.layer_velocity,
                                                                                                        self.depth,
                                                                                                        self.receiver,
                                                                                                        self.delta_t1,
                                                                                                        self.time_array1)
            # capture ground truth arrival time and receiver offset in sensor network

            finaltime, newsynthetic_arriavl_1 = time_resolution.ground_truth_selector_certainreceivers(self.trace_number,
                                                                                                       newsynthetic_arriavl,
                                                                                                       self.allreceiver_number,
                                                                                                       self.layer_n)

            peak=travel_time
            # transpose measured travel time for further processing
            peak = np.array(peak).T

        # distributed t0 estimate of each sensor for noiseless link
        if self.noisy_link == 0:

            # gather travel time and sensor position from neighbours and perform nonlinear-least-square fitting
            t0_estimate, para_estimate, accessible_sensor = distributed_nonlinea_least_square_fitting.fitting_sensor_networking(
                self.trace_number,self.line_topology_flag,self.delta_x, self.receiver_number, self.layer_n,
                np.array(peak), self.accessible_number)
            if compare_analysis_flag == 'scheme_flag':
                t0_estimate, para_estimate = t0_estimateAverage, para_estimateAverage
            # average consensus on parameters
            consensus_t0, consenus_para, local_information, local_information1 = distributed_average_consensus.t0_average_consensus(self.DNMO_iteration,self.t0_truth, self.m0_truth,self.layer_n, 0, 0, self.SNR_noisy_link,self.receiver_number, t0_estimate,para_estimate,accessible_sensor)
            # if we don't apply dirls algorithm
            consensus_t01, consenus_para1, local_information11, local_information111 = distributed_average_consensus.t0_average_consensus(self.DNMO_iteration,self.t0_truth,self.m0_truth,self.layer_n, 0, 0,self.SNR_noisy_link,self.receiver_number,t0_estimate,para_estimate,accessible_sensor)
            # implement normal moveout
            newpeak, optimal_time, ground_depth, v_layer, t0coff, t0coffop, ground_depth_dis, v_layer_dis = NMO_estimation_performance_distributed.normal_moveout(
                self.vel_flag, self.vel_flag1, self.pattern_analzye_flag, self.osicillation_pattern, finaltime, local_information,
                local_information1, local_information11, local_information111, self.receiver_distance, self.noisy_link, consensus_t0,
                consenus_para, peak,self.layer_velocity, self.test_depth, self.layer_n, self.receiver_number)
            #investigate averaged estimated parameters variation
            estimated_parameters_variation(local_information, local_information1,self.t0_truth,self.m0_truth,self.receiver_distance)
            # visualize estimation performance of distributed NMO with average consensus at different iterations
            for j in self.iteration:
                local_estimation_performance_visualization_distributedNMO.plot_distributed_nmo(self.layer_n, self.vel_flag1,
                                                                                               self.layer_velocity,
                                                                                               self.test_depth, peak,
                                                                                               finaltime,
                                                                                               self.receiver_distance,
                                                                                               self.receiver_number,
                                                                                               local_information1,
                                                                                               local_information,
                                                                                               ground_depth, v_layer,
                                                                                               ground_depth_dis,
                                                                                               v_layer_dis, j)
            # if we want to plot real time estimation curve,calculate normalized estimation
            # error for all iterations

            v_error_array, d_error_array, v_error_array_cen, d_error_array_cen, ims = distributed_normal_moveout_estimation_error.distributed_NMO_estimation(
                self.layer_n, self.test_depth, self.layer_velocity, ground_depth, v_layer, self.receiver_distance, finaltime,
                local_information, local_information1, self.receiver_number, peak, t0_estimate, para_estimate,
                accessible_sensor)

            # compare performance of distributed normal moveout with average
            # consensus and ATC-framework
            # plot real time estimation curve with noiseless transmission link
            distributed_nmo_real_time_estimation_error_plot.plot_estimation_comparison(v_error_array, d_error_array, v_error_array_cen, d_error_array_cen)
            #investigate how the number of neighbors influence estimation
            self.neighbor_number_influence( self.neighbor_number,peak, finaltime)
        else:
            # for noisy link
            noisy_peak, std = NMO_estimation_performance_distributed.noisy_picking(peak, self.SNR_noisy_link, self.receiver_number)

            # average consensus on parameters


            t0_estimate, para_estimate, accessible_sensor = distributed_nonlinea_least_square_fitting.fitting_sensor_networking(self.trace_number,
                self.line_topology_flag,self.delta_x, self.receiver_number, self.layer_n,
                np.array(noisy_peak).T, self.accessible_number)
            # average consensus on parameters
            consensus_t0, consenus_para, local_information, local_information1 = distributed_average_consensus.t0_average_consensus(
                self.DNMO_iteration, self.t0_truth, self.m0_truth, self.layer_n, 1, 0, self.SNR_noisy_link,
                self.receiver_number, t0_estimate, para_estimate, accessible_sensor)
            # if we don't apply dirls algorithm
            consensus_t01, consenus_para1, local_information11, local_information111 = distributed_average_consensus.t0_average_consensus(
                self.DNMO_iteration, self.t0_truth, self.m0_truth, self.layer_n, 1, 0, self.SNR_noisy_link,
                self.receiver_number, t0_estimate, para_estimate, accessible_sensor)
            # implement normal moveout
            newpeak, optimal_time, ground_depth, v_layer, t0coff, t0coffop, ground_depth_dis, v_layer_dis = NMO_estimation_performance_distributed.normal_moveout(
                self.vel_flag, self.vel_flag1, self.pattern_analzye_flag, self.osicillation_pattern, finaltime,
                local_information,
                local_information1, local_information11, local_information111, self.receiver_distance, self.noisy_link,
                consensus_t0,
                consenus_para, peak, self.layer_velocity, self.test_depth, self.layer_n, self.receiver_number)
            # investigate averaged estimated parameters variation
            estimated_parameters_variation(local_information, local_information1, self.t0_truth, self.m0_truth,
                                           self.receiver_distance)
            # visualize estimation performance of distributed NMO with average consensus at different iterations
            for j in self.iteration:
                local_estimation_performance_visualization_distributedNMO.plot_distributed_nmo(self.layer_n,
                                                                                               self.vel_flag1,
                                                                                               self.layer_velocity,
                                                                                               self.test_depth, peak,
                                                                                               finaltime,
                                                                                               self.receiver_distance,
                                                                                               self.receiver_number,
                                                                                               local_information1,
                                                                                               local_information,
                                                                                               ground_depth, v_layer,
                                                                                               ground_depth_dis,
                                                                                               v_layer_dis, j)
            # if we want to plot real time estimation curve,calculate normalized estimation
            # error for all iterations

            v_error_array, d_error_array, v_error_array_cen, d_error_array_cen, ims = distributed_normal_moveout_estimation_error.distributed_NMO_estimation(
                self.layer_n, self.test_depth, self.layer_velocity, ground_depth, v_layer, self.receiver_distance,
                finaltime,
                local_information, local_information1, self.receiver_number, peak, t0_estimate, para_estimate,
                accessible_sensor)

            # compare performance of distributed normal moveout with average
            # consensus and ATC-framework
            # plot real time estimation curve with noiseless transmission link
            distributed_nmo_real_time_estimation_error_plot.plot_estimation_comparison(v_error_array, d_error_array,
                                                                                       v_error_array_cen,
                                                                                       d_error_array_cen)

        # return the normalized average estimation error of layer depth and
        # velocity
        return peak,finaltime,v_error_array, d_error_array,v_error_array_cen,d_error_array_cen
    """""""""
    main function to implement distributed normal-moveout with adapt-then-combine strategy
    """""""""
    #output args:
    #        t0_array: estimated t0 per receiver per iteration
    #        m0_array: estimated m0 per receiver per iteration
    #        travel_time: picking travel time at receivers
    #        local_error: normalized mean depth estimation error with ATC algorithm per iteration
    #        local_velocity: normalized mean velocity estimation error with ATC algorithm per iteration
    def distributed_gradient_descent(self):
        # generate noisy synthetic seismic measurement at receivers
        true,time_grid_pressure,standard_deviation_array, noisy_final, filter_trace, post_processing, post_processing1, time_array, ricker_amp, newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1 = self.noisy_seismic_measurement_generation()
        # picking travel time after processing seismic measurement with deconvolution or using STA/LTA picker
        travel_time,reflectivity_inter,peak_source = self.travel_time_picking_main(standard_deviation_array, noisy_final, post_processing1, time_array,ricker_amp,
                                          newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1)
        #transpose travel time picking for further processing
        travel_time = np.array(travel_time).T
        # get measured travel time and receiver offset from neighbours for each receiver
        final_local_traveltime, receiver_distance, neighbour_numberarray, neighbourindice_array = neighbours_generator(
        self.line_topology_flag,self.receiver_distance, travel_time,self.accessible_number)

    # get initial estimated parameters by nonlinear least-square fitting with travel time and
    # sensor positions from neighbours

        t0_estimate, para_estimate = initial_estimate_parameters(self.trace_number,self.receiver_number, np.array(travel_time).T,
                                                             neighbourindice_array,self.delta_x,self.layer_n)

    # apply adapt-then-combine framework for estimated parameters at each receiver for each layer
        t0_array,m0_array,determinant_array=adapt_then_combine_optimizer(self.layer_n, para_estimate, t0_estimate, self.SNR_noisy_link, self.noisy_link,
                                 neighbourindice_array, self.plot_flag,
                                 self.DNMO_iteration, travel_time, self.yita,
                                 self.receiver_distance)
        #solve layer velocity and depth directly from final estimated parameters from ATC
        #algorithm and calculate normalized mean estimation error for certain iterations
        local_error,local_velocity=vel_depth_solver_ATC_iterations(self.vel_flag1,self.DNMO_iteration,self.iteration, t0_array, m0_array, self.receiver_distance, self.layer_n, self.layer_velocity,self.test_depth)


    # plot real time estimation curve with noiseless transmission link
        plot_ATC_estimation_performance(local_error, local_velocity)
    #plot determiant of Hessian to observe convexity of cost function
        plt.subplot(2,1,1)
        Hessian_determinant_plot(self.DNMO_iteration,determinant_array)
        t0_plotflag=2
        estimated_parameters_plot(t0_plotflag, t0_array,m0_array, self.DNMO_iteration, self.t0_truth, self.m0_truth)
    # return arguments
        return t0_array, m0_array, travel_time,local_error,local_velocity,t0_estimate, para_estimate