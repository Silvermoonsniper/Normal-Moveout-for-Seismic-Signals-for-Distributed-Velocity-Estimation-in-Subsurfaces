from scipy.signal import find_peaks
import numpy as np
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


# arrival time picking algorithm
# input args:
#       reflectivity:reflectivity series for multiple sensors
#       trace_number: indices for receiver array
#       new_delta_t: time resolution
#       newtime_array: time array for plotting
#       newsynthetic_arriavl: ground truth arrival time to assist picking
#output args:
#       final_arrival: picking time estimate for reflection wave arrivals at receivers

def arrival_time_picking(reflectivity, trace_number, new_delta_t, newtime_array, newsynthetic_arriavl):
    # find all peaks

    all_peak, peak_amplitude = peak_finder(reflectivity)
    # convert indice to time
    all_peak = np.array(all_peak) * new_delta_t

    # find local maxima in all reflectivity series and their indices

    maxes = []
    final_arrival = []

    b = 0
    for j in peak_amplitude:
        arrival_index = []
        final_maxima = []
        ground_truth = []
        for i in range(1, len(j) - 1):
            if (j[i - 1] < j[i] and j[i] > j[i + 1]):
                maxes.append(j[i])
                # arrival time correspond to local maxima
                arrival_index.append(all_peak[b][i])
        # choose local maxima nearest to ground truth
        # ground truth

        ground_truth = newsynthetic_arriavl[trace_number[b]]

        b += 1
        for k in range(len(ground_truth)):
            # convert to list

            arrival_index = list(arrival_index)
            # error
            difference = abs(closest(arrival_index, ground_truth[k]) - ground_truth[k])
            thereshold = 2 * abs(closest(newtime_array, ground_truth[k]) - ground_truth[k])
            if difference >= thereshold:
                # append picking
                final_maxima.append(closest(newtime_array, ground_truth[k]))
            else:
                # append picking
                final_maxima.append(closest(arrival_index, ground_truth[k]))
        final_arrival.append(final_maxima)

    return final_arrival
