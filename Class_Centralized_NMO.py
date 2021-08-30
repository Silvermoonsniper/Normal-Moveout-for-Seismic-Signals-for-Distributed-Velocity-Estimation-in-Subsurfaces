import STA_LTA_picking_traveltime
import ground_truth_arrival_time_generator
import noisy_measurement_generator
import seismic_trace_deconvolution
import seismic_trace_extraction
import time_resolution
import travel_time_picking
import wave2D
import wave_pattern_plot
import wave_propagation_video
from Normal_moveout_analysis.inter_receiver_distance_effect import distance_effect
from initial_paramters_setting import receiver_position_assignment
from main_algorithms import centralized_NMO
from noisy_seismic_trace_plot import noisy_trace_plot
from reflectivity_series_plot import reflectivity_plot, multiple_reflectivity_plot
from IPython.display import HTML
import numpy as np

from wiener_filter_results_plot import wiener_filter_performance_visualization


class Centralized_NMO_estimator():
    # initialization of parameters
    def __init__(self, initial_args):
        for key in initial_args:
            setattr(self, key, initial_args[key])
            # calculate receiver position on one side
        self.source_to_receiver_distance = receiver_position_assignment(initial_args["distance"],initial_args["receiver_number_oneside"])
        # calculate thickness of each layer
        self.depth = initial_args["test_depth"][1:] - initial_args["test_depth"][0:len(initial_args["test_depth"]) - 1]
        # number of layers
        self.layer_n = len(initial_args["layer_velocity"])
        # total time rounds for simulation
        self.time_rounds = int(initial_args["time"] / initial_args["delta_t"])
        # coordinate of source on the ground surface (m), always located at center of ground surface
        self.source_coordinate_x = 0.5 * initial_args["distance"]
        # size of cell in FDTD
        self.delta_x = initial_args["distance"] / initial_args["cell_number"]
        # receiver constellation array
        self.n = int(initial_args["distance"] /self.delta_x)
        # indice of receivers on one side
        self.trace_number = np.array(self.source_to_receiver_distance/ self.delta_x).astype(int)
        # the discretization level in y vertical direction (unit:m)
        self.delta_h = self.delta_x
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
    """""""""""
        Noisy seismic measurement data generation
    """""""""""

    def noisy_seismic_measurement_generation(self):

        # generate ground truth arrival time data from multiple receivers for each layer
        desired, indexarray, newsynthetic_arriavl = ground_truth_arrival_time_generator.real_signal(self.layer_n,
                                                                                                    self.layer_velocity,
                                                                                                    self.depth,
                                                                                                    self.receiver,
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
                                                                                                      self.distance,
                                                                                                      self.fm,
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
        return true,time_grid_pressure, standard_deviation_array, noisy_final, filter_trace, post_processing, post_processing1, time_array, ricker_amp, newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1,interface_coordinate

    """""""""""
        Picking travel time of reflection wave at receivers    
        """""""""""

    def travel_time_picking_main(self, standard_deviation_array, noisy_final, post_processing1, time_array, ricker_amp,
                                 newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1):

        # choose the picking method used to get measured travel time of reflections at receivers
        if self.picking_travel_time_method == 'STA_LTA':
            peak = STA_LTA_picking_traveltime.picking_time_noisy_trace_STALTA(self.time, self.time_rounds,
                                                                              self.windowsize,
                                                                              self.trace_number, newsynthetic_arriavl,
                                                                              newdesired, post_processing1, self.SNR,
                                                                              time_array,
                                                                              self.time_array1, self.layer_n)
            # transpose measured travel time for further processing
            peak = np.array(peak).T
            reflectivity = 0
            peak_source = 0
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
            # if no sufficient memory to allocate numpy array in the deconvolution stage use following codes
            #peak = travel_time_picking.arrival_time_picking(reflectivity_inter, self.trace_number, new_delta_t,
            #                                                new_time_array,
            #                                                newsynthetic_arriavl_1)

            # pick arrival time data for NMO by distinguishing local maxima peaks in the deconvolved profile
            peak = travel_time_picking.arrival_time_picking(reflectivity, self.trace_number, self.delta_t,
                                                            time_array,
                                                            newsynthetic_arriavl_1)
            # return picking travel time, reflectivity series,
        return peak, reflectivity, peak_source
    '''''''''''
        main function for classical centralized Normal-moveout without recalculating of picking time
    '''''''''''

    def centralized_nmo_main_function(self):
        # generate noisy synthetic seismic measurement at receivers
        true,time_grid_pressure, standard_deviation_array, noisy_final, filter_trace, post_processing, post_processing1, time_array, ricker_amp, newdesired, finaltime, newsynthetic_arriavl, newsynthetic_arriavl_1,interface_coordinate = self.noisy_seismic_measurement_generation()


        # picking travel time after processing seismic measurement with deconvolution or using STA/LTA picker
        final_arrival, reflectivity, peak_source = self.travel_time_picking_main(standard_deviation_array, noisy_final,
                                                                                 post_processing1, time_array,
                                                                                 ricker_amp,
                                                                                 newdesired, finaltime,
                                                                                 newsynthetic_arriavl,
                                                                                 newsynthetic_arriavl_1)

        # if we want to display video or wave pattern at given time step for wave propagation


        if self.video_flag == "display_video":
                ani = wave_propagation_video.video_wave_propagation(time_grid_pressure, self.distance, self.test_depth)
                HTML(ani.to_html5_video())
        elif self.video_flag == "display_wave_pattern":

                # plot and instantaneous snapshot of wave propagation at specific iteration
                wave_pattern_plot.local_wave_propagation_visualization(self.delta_x, self.delta_h,time_grid_pressure, self.distance,
                                                                       self.test_depth,
                                                                       self.time_instant,interface_coordinate)
        # not play
        #plot noisy reflectivity series
        multiple_reflectivity_plot(time_array, reflectivity, self.time_array1, newdesired, peak_source)
        #plot seismic trace at mulitple receivers
        seismic_trace_extraction.multiple_trace_plot(time_array, post_processing1, self.time_array1, newdesired, peak_source)
            # plot noisy seismic trace for single receiver
        noisy_trace_plot(ricker_amp, time_array, noisy_final[self.single_plot_indice],self.picking_travel_time_method)
            # plot filtered noisy seismic measurement with wiener filter
        wiener_filter_performance_visualization(time_array, filter_trace[self.single_plot_indice], true[self.single_plot_indice])
            # plot reflectivity series
        if self.picking_travel_time_method == 'picking_deconvolve_measurement':
             reflectivity_plot(reflectivity[self.single_plot_indice], time_array, self.time_array1, newdesired,
                              self.trace_number, self.single_plot_indice, peak_source)
            # receiver constellation array
        receiver_distance = (self.trace_number + 1) * np.array(self.delta_x)
            # implement normal moveout with estimated parameters from nonlinear least-square fitting
            # visulize final reconstruction results
        peak, optimal_time, ground_depth, v_layer, t0coff, t0coffop, delta_t = centralized_NMO.normal_moveout(
                finaltime, self.vel_flag, final_arrival, receiver_distance, self.layer_velocity, self.test_depth,
                self.layer_n, self.delta_x,
                self.delta_t)
