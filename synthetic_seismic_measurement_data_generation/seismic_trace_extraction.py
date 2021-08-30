# define a function to get cross-correlation between source wavelet and seismic data measured at receiver
from scipy import signal
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import numpy as np
from Ricker_wavelt import *
# here we try to make a function to estimate arriaval time from output response at receiver
# input seismic trace is an array
def Peak_time(data_array, time_array):
    Max = []
    k = 0
    for k in range(len(data_array)):
        maximum = np.array(data_array).max()

        if maximum == data_array[k]:
            estimate_time = time_array[k]

    return estimate_time







def cross_correlation(x, y):
    # compute PCC
    cr = np.correlate(x, y) / len(x)

    return cr


# construct time-delayed version of source impulse and find arrival time correspond to maximum cross-correlation
# input parameter:
# source: ricker wavelet at source
# y: seismic data measuremed at receiver
def peak_time_extractor(source, y, delta_t, desired, r, time_array, peak_source, time_array1):
    cr_value = []
    shift_time = []
    newdesired = np.pad(desired[r], peak_source, 'constant')

    for i in range(1, len(source)):
        shifted_source = np.pad(source, i, 'constant')

        shifted_source_final = shifted_source[0:len(source)]

        # computer cross-correlation
        cr = np.array(cross_correlation(shifted_source_final, y)).mean()
        cr_value.append(cr)
        shift_time.append(i * delta_t)
    # find max
    plt.plot(shift_time[0:], cr_value[0:])
    plt.plot(time_array1[0:100000], 0.0045 * newdesired[0:100000])
    cr_max = np.array(cr_value).max()

    for j in range(len(cr_value)):
        if cr_value[j] == cr_max:
            max_index = j

    # computer arrival time
    arrival = time_array[max_index]

    return arrival, cr_value


from scipy.interpolate import interp1d

#downsample operation
def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled


# plot output response for all receivers




# function for extract receiver measurement and plot output response for all receivers and input impulse response at source
# input args:
#       plot_flag: flag for plot
#       seismic_trace_x: seismic measurement recorded at all cells on one side of ground surface
#       time: simulation time
#       time_rounds: total number of time samples
#       time_rounds1: total number of time samples for plotting ground truth curves
#       desired: ground truth arrival times series
#       single_receiver: flag to plot single receiver measurement
#output args:
#      post_processing: seismic measurement after using matched filter
#      post_processing1: raw seismic measurement at receiver
#      cross: cross-correlation between wavelet and measurement
#      time_array: time series for plot
#      ricker_amp: source wavelet
#      newdesired_array: shifted ground truth series

def plot_response(plot_flag, seismic_trace_x,time, time_rounds, time_rounds1, simulation_time, desired,
                  single_receiver):
    fig = plt.figure(figsize=(6, 6))

    delta_t = time / time_rounds
    # array to store seismic measurement
    x_part = np.zeros([time_rounds, 1])
    h_part = np.zeros([time_rounds, 1])

    time_array = np.zeros([time_rounds, 1])
    data_array = []
    r = 0
    receiver_number = len(desired)
    # variable for extract measurement in while loop
    j = 0
    k = 0
    legend = ['Receiver 1', 'Receiver 2', 'Receiver 3', 'Receiver 4', 'Receiver 5', 'Receiver 6']
    # initialize post-processing seismic traces
    post_processing = []
    post_processing1 = []
    cross = []

    # sampling interval
    delta_t1 = time / time_rounds1
      # array to store ground truth array
    newdesired_array = []
    # time series for plotting
    time_array1 = np.linspace(1, time_rounds1, num=time_rounds1) * delta_t1
    time_array = np.linspace(1, time_rounds, num=time_rounds) * delta_t

    # two while loops to extract seismic measurement for each receiver
    while r < (receiver_number):
        while k <= len(seismic_trace_x):
            if k + r < len(seismic_trace_x):
                data_array.append(seismic_trace_x[k + r])

            k += receiver_number

        # obtain array for seismic measurement and discrete time series
        for a in np.array(data_array):
            data = np.array(a)

            if j < time_rounds:
                a1 = data[0]
                x_part[j] = a1

                h_part[j] = data[1]
                time_array[j] = data[2]

                j += 1
        j = 0
        # flatten the seismic and source signal array for further processing
        x_part = np.array(x_part).flatten()
        h_part = np.array(h_part).flatten()

        time_array = np.array(time_array).flatten()

        # maximum simulation time unit:s
        max_time = time_array.max()

        fm = 200
        time_slot = np.linspace(-5 * np.sqrt(2) / (2 * np.pi * fm), max_time, num=time_rounds)
        ricker_amp = Ricker_wavelet(fm, time_slot)
        # normalize seismic measurement data to make it visible in pot
        norm2 = np.array(x_part).max()
        x_part_norm = x_part / norm2

        # shift ground truth as peak of ricker wavelet is shifted by amount of "time_shift"
        time_shift = 5 * np.sqrt(2) / (2 * np.pi * fm)
        # peak index of source wavelet and shift the ground truth series
        peak_source = int(time_shift / delta_t1)


        # specify flag for different plots

        # plot flag to plot ricker source wavelet, if it is 1: plot, else: don't plot
        source_flag = 0
        # convolve seismic trace with wavelet to maximize SNR
        impulse=ricker_amp[0:120]
        newxpart=np.convolve(impulse,x_part)
        # get shifted ground truth series
        newdesired = np.pad(desired[r], peak_source, 'constant')
        #plot for single receiver measurement
        if single_receiver == 1 and source_flag == 0 and plot_flag == 1:
            if r == 25:
                # shift ground truth dirac series
               # newdesired = np.pad(desired[r], peak_source, 'constant')
              #  plt.plot(time_array[0:5000], x_part_norm[0:5000], label='Source')

                # plt.plot(time_array[0:4300],newxpart[0:4300],label='Source')
             #   plt.plot(time_array1[0:50000], newdesired[0:50000])

                plt.plot(time_array[0:2600], np.array(ricker_amp[0:2600]), label='Source')

                plt.xlabel('time (s)')
                plt.ylabel('Amplitude (m)')
            #plt.ylabel('Wave amplitude $u_{m_d,0}^j$')
                plt.title('Source Wavelet')
            #plt.title('Seismic measurement at single receiver with Neumann condition', pad=20)
            #plt.legend(['$\mathbf{z}_7$', 'Ground truth arrival time', 'Ricker Wavelet'])
                plt.legend(['Ricker wavelet'])

#plot multiple receiver measurement
        elif single_receiver == 0 and r < len(desired)+1 and plot_flag == 1:
            #      plt.plot(time_array[0:len(time_array)-4500],ricker_amp[0:len(time_array)-4500],label='Source')
            plt.plot(time_array[0:4200], x_part_norm[0:4200], label='Source')

            # plt.plot(time_array[0:4300],newxpart[0:4300],label='Source')
            #  plt.plot(time_array[0:len(time_array)-2],cr_value[0:len(time_array)-2],label='Source')
            plt.xlabel('time (s)')
            plt.ylabel('Impulse Response X-component $u_{x,(j,0)}^n$')
            plt.title('Filtered Impulse Response at Multiple Receiver ', loc='right', pad=10)
            #  plt.title('Impulse Response at Multiple Receiver and Source Wavelet')
            plt.legend(legend)
        elif single_receiver == 1 and source_flag == 1 and plot_flag == 1:
            plt.plot(time_array[0:len(time_array) - 3], h_part[0:len(time_array) - 3], label='Source')
            plt.xlabel('time (s)')
            plt.ylabel('Impulse Response $u_{x_s,0}^j$')
            plt.title('Ricker Wavelet Impulse Response $u_{x_s,0}^n$ at Source')
            plt.legend(['$u_{x_s,0}^n$'])
            # store orignial impuse response time series and non-filtered series for each receiver

        post_processing1.append(x_part)
        post_processing.append(newxpart)
        # store ground truth data
        #if single_receiver == 0:
        newdesired_array.append(newdesired)
        # proceed with next sesimic measurement extraction process
        r += 1
        k = 0
        data_array = []
    plt.show()
    return post_processing, post_processing1, cross, time_array, ricker_amp, newdesired_array
def multiple_trace_plot(time_array,reflectivity,time_array1,newdesired,peak_source):
    # plot result
    plot_flag = 1
    if plot_flag == 1:
        f = plt.figure(figsize=(10, 10))

        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharey=True)
        ax1.plot(time_array[0:len(time_array)], reflectivity[5][0:len(time_array)]/np.array(reflectivity[5][0:len(time_array)]).max(), label='Source')

        ax2.plot(time_array[0:len(time_array)], reflectivity[15][0:len(time_array)]/np.array(reflectivity[15][0:len(time_array)]).max(), label='Source')
        ax3.plot(time_array[0:len(time_array)], reflectivity[25][0:len(time_array)]/np.array(reflectivity[25][0:len(time_array)]).max(), label='Source')
        ax4.plot(time_array[0:len(time_array)], reflectivity[35][0:len(time_array)]/np.array(reflectivity[35][0:len(time_array)]).max(), label='Source')
        ax5.plot(time_array[0:len(time_array)], reflectivity[45][0:len(time_array)]/np.array(reflectivity[45][0:len(time_array)]).max(), label='Source')
        ax6.plot(time_array[0:len(time_array)], reflectivity[55][0:len(time_array)]/np.array(reflectivity[55][0:len(time_array)]).max(), label='Source')
        # overlay wih ground truth
        ax1.plot(time_array1[0:50000], newdesired[5][0: 50000], label='Source')
        ax2.plot(time_array1[0:50000], newdesired[15][0: 50000], label='Source')
        ax3.plot(time_array1[0:50000], newdesired[25][0: 50000], label='Source')
        ax4.plot(time_array1[0:50000], newdesired[35][0: 50000], label='Source')
        ax5.plot(time_array1[0:50000], newdesired[45][0: 50000], label='Source')
        ax6.plot(time_array1[0:50000], newdesired[55][0: 50000], label='Source')
        ax1.set_title('Seismic measurement for multiple receivers')
        ax6.set_xlabel('time (s)')
        ax3.set_ylabel('normalized wave amplitude')
        plt.tight_layout()
        f.legend(['$z_d, d=1,3,5,7,9,11$', 'ground truth'],loc="upper right")
    plt.show()