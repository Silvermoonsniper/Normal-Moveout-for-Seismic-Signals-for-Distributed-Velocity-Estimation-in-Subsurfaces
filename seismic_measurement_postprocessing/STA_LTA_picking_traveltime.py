# design STA/LTA algorithm

import numpy as np
import matplotlib.pyplot as plt

from STA_ratio_plot import STA_ratio_plot
from noisy_measurement_generator import target_trace, multiple_noisy_trace

# design STA/LTA algorithm to obtain measured travel time of reflection waves at receivers, which is standard method in
# seismology and earthquake and also small event of wave detection.
# input args:
#        seismogram：noisy seismic measurement at receiver
#        k: window size
# output args:
#       ratio: STA/LTA ratio profile for single receiver

def STA_LTA_picking(seismogram, k):



    # calculate averaged amplitude in long window
    # long window is whole seismogram
    # absoute value
    abs_val = np.absolute(seismogram)
    averaged = np.mean(abs_val)
    # short window size
    ratio = []

    for j in range(len(seismogram) - k):
        if j + k < len(seismogram):
            # construct window
            short_window = seismogram[j:j + k]
            abs_val1 = np.absolute(short_window)
            # mean of STA window
            averaged1 = np.mean(abs_val1)

            # calculate STA ratio
            STA_ratio = averaged1 / averaged

            ratio.append(STA_ratio)

    ratio = ratio / np.array(ratio).max()

    # choose peak in ratio curve

    return ratio


from scipy.signal import find_peaks

#function to get peaks in the waveform for all receivers
#input args:
#         input: waveform that we need to extract peaks from multiple receivers, could be relectivity series or
#raw measurement
#         peak_indice: indice array that identify peaks with given threshold for multiple waveforms
#         finalpeak_amplitude: amplitude of those peaks

def peak_finder(input):
    # initialize peak indice array
    peak_indice = []
    finalpeak_amplitude = []
    for j in input:

        peaks, _ = find_peaks(j, height=0)
        peak_indice.append(peaks)
        peak_amplitude = np.zeros([len(peaks), 1])
        # amplitude of peak
        b = 0
        for a in peaks:
            peak_amplitude[b] = (j[a])
            b += 1
        finalpeak_amplitude.append(peak_amplitude)
    return peak_indice, finalpeak_amplitude


# find cloeset element to k in a list
def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


# function to get multiple  STA/LTA ratio curves from different noisy seismic traces
# input args:
#   windowsize: the number of data points for short-term windown used in STA-LTA picker
#   trace_number: indice array for seismic measurement. trace_number=[0;5] means extract seismic traces for first 6 receivers
#   time_array: time sampling array for plotting
#   #delta_t: time resolution in FDTD
#   newsynthetic_arriavl: ground truth arrival time at receivers
#   newdesired: shifted ground truth series
#   time_array1: time interval series for plotting ground truth series
#   layer_n: the number of layers
#   standard_deviation: standard deviation in the noisy seismic measurement
#output args:
#   final_arrival: selected time estimate of reflections from interfaces for whole receiver array

def multiple_STALTA_ratio_curve_extractor(newdesired,post_processing1, windowsize, trace_number, time_array, time_array1,delta_t,
                                          newsynthetic_arriavl, layer_n,SNR):
    # get corresponding noiseless seismic measurement
    target = target_trace(post_processing1, trace_number)
    # add AWGN into traces to generate noisy seismic traces
    noisy_final, filter_trace,clean_seismic_trace,standard_deviation = multiple_noisy_trace(target, delta_t, SNR)

    # loop over all traces
    ratio_array = []
    for j in range(len(noisy_final)):

        # deconvolve noisy signal
        noisy_flag = 1
        r = trace_number[j]

        ratio = STA_LTA_picking(np.array(noisy_final[j]).flatten(), windowsize)
        # append different STA-LTA ratio curve for different receivers
        ratio_array.append(ratio)
    # plot STA/LTA_ratio for final receiver
    STA_ratio_plot(ratio_array[1], time_array, time_array1, newdesired[trace_number[1]])

    # find peaks in STA-LTA ratio curves
    peak_indice, finalpeak_amplitude = peak_finder(ratio_array)
    # convert into real time
    peaktime = np.array(delta_t) * peak_indice

    # ground truth data
    ground_truth = newsynthetic_arriavl.reshape([len(post_processing1), layer_n])

    # choose peak which is closest to ground truth
    final_arrival = []

    for j in range(len(peaktime)):

        final_maxima = []
        for k in range(layer_n):
            # append picking
            final_maxima.append(closest(peaktime[j], ground_truth[j][k]))

        final_arrival.append(final_maxima)
        # transpose final picking array to further processing
    final_arrival = np.array(final_arrival).T
    # return peak time
    return final_arrival


# main function to get picking arrival time for different layers at different receivers and use distributed SGD to learning parameters from measured picking arrival time

#input args:
#      time: simulation time
#      time_rounds: number of time samples
#      windowsize: windowsize uses in STA/LTA picker
#      trace_number: indice array of receiver array
#      newsynthetic_arriavl: ground truth arrival time array
#      newdesired:shifted ground truth sparse series
#      post_processing1: noiseless seismic measurement at receiver array
#      standard_deviation: standard deviation uses in generate noisy seismic trace
#      time_array: time array for plotting measurement
#      time_array1: time array for plotting ground truth series
#      layer_n: number of layers
#output args：
#      final_arrival：selected time estimate of reflections from interfaces for whole receiver array




def picking_time_noisy_trace_STALTA(time, time_rounds, windowsize, trace_number, newsynthetic_arriavl,newdesired,post_processing1,SNR,time_array,time_array1,layer_n):
    # time resolution for simulation

    delta_t = time / time_rounds

    final_arrival = multiple_STALTA_ratio_curve_extractor(newdesired,post_processing1, windowsize, trace_number, time_array,time_array1,
                                                          delta_t, newsynthetic_arriavl, layer_n,SNR)
    return final_arrival



# function to investigate how window size influences picking
# input args:
#        newsynthetic_arriavl: ground truth arrival time
#        layer_n: number of layers

def windwo_size_effect(newsynthetic_arriavl,layer_n):
    # window size for STA/LTA algorithm
    window = np.linspace(40, 800, num=20)

    time_rounds = 25000
    time = 0.25
    delta_t = time / time_rounds
    # GROUND truth
    newsynthetic_arriavl = np.array(newsynthetic_arriavl).reshape([58, layer_n])
    newsynthetic_arriavl = np.array(newsynthetic_arriavl[0:28]).T
    # window_length
    window_len = window * np.array(delta_t)
    # estimation error array
    error_array = []
    for p in range(len(window)):
        final_arrival = picking_time_noisy_trace_STALTA(int(window[p]))
        # calculate estimation error
        error = np.linalg.norm(np.array(final_arrival - newsynthetic_arriavl).flatten()) / np.linalg.norm(
            newsynthetic_arriavl.flatten())
        error_array.append(error)
    plt.plot(window_len, error_array)
    plt.xlabel('window duration')
    plt.ylabel('normalized estimation error')


