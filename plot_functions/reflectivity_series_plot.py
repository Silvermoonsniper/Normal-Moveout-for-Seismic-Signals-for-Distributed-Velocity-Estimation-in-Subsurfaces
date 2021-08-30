import matplotlib.pyplot as plt
import numpy as np
#function to plot reflectivity series
#input args:
#         LS:reflectivity series
#         time_array: time series for plot
#         time_array1: time series for plotting ground truth series
#         newdesired: ground truth series
#         trace_number: indice of receivers
#         r: indice of receiver in receier array
#         peak_source: double shifted number of datapoints in the ricker wavelet
def reflectivity_plot(LS,time_array,time_array1,newdesired,trace_number,r,peak_source):
    plot_flag = 1
    if plot_flag == 1:
        plt.plot(time_array[0:len(time_array)], LS[0:len(time_array)])
        newdesired[r]=np.pad(newdesired[r], (int(0.5*peak_source), 0), 'constant')
        plt.plot(time_array1[0:50000], newdesired[r][0: 50000])
        # plt.plot(time_array[0:4000],original_trac[0:4000])
        # plt.plot(time_array[0:4000],post_processing[r][0:4000])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        # plt.legend(['original seismic trace'])
        plt.legend(['$\hat{\mathbf{\mu}}_d$', 'ground truth'])
        plt.title('The orignial seismic trace')
        plt.title('The Comparison between reflectivity series and ground truth reflection')
    plt.show()

#function to plot multiple reflectivity series
#input args:
#     time_array: time series for plotting
#     reflectivity: reflectivity series at multiple receivers
#     time_array1: time series for plotting ground truth series
#     newdesired: ground truth series
#     peak_source: double shifted number of datapoints in the ricker wavelet
def multiple_reflectivity_plot(time_array,reflectivity,time_array1,newdesired,peak_source):
    # plot result
    plot_flag = 1
    if plot_flag == 1:
        f = plt.figure(figsize=(10, 10))
        newdesired[5] = np.pad(newdesired[5], (int(0.5 * peak_source), 0), 'constant')
        newdesired[15] = np.pad(newdesired[15], (int(0.5 * peak_source), 0), 'constant')
        newdesired[25] = np.pad(newdesired[25], (int(0.5 * peak_source), 0), 'constant')
        newdesired[35] = np.pad(newdesired[35], (int(0.5 * peak_source), 0), 'constant')
        newdesired[45] = np.pad(newdesired[45], (int(0.5 * peak_source), 0), 'constant')
        newdesired[55] = np.pad(newdesired[55], (int(0.5 * peak_source), 0), 'constant')

        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharey=True)
        ax1.plot(time_array[0:len(time_array)], reflectivity[0][0:len(time_array)], label='Source')
        ax2.plot(time_array[0:len(time_array)], reflectivity[1][0:len(time_array)], label='Source')
        ax3.plot(time_array[0:len(time_array)], reflectivity[2][0:len(time_array)], label='Source')
        ax4.plot(time_array[0:len(time_array)], reflectivity[3][0:len(time_array)], label='Source')
        ax5.plot(time_array[0:len(time_array)], reflectivity[4][0:len(time_array)], label='Source')
        ax6.plot(time_array[0:len(time_array)], reflectivity[5][0:len(time_array)], label='Source')
        # overlay wih ground truth
        ax1.plot(time_array1[0:50000], newdesired[5][0: 50000], label='Source')
        ax2.plot(time_array1[0:50000], newdesired[15][0: 50000], label='Source')
        ax3.plot(time_array1[0:50000], newdesired[25][0: 50000], label='Source')
        ax4.plot(time_array1[0:50000], newdesired[35][0: 50000], label='Source')
        ax5.plot(time_array1[0:50000], newdesired[45][0: 50000], label='Source')
        ax6.plot(time_array1[0:50000], newdesired[55][0: 50000], label='Source')
        ax1.set_title('Reflectivity Series for multiple receivers')
        ax6.set_xlabel('time (s)')
        ax3.set_ylabel('normalized amplitude')
        plt.tight_layout()
        f.legend(['$\hat{\mathbf{\mu}}_d$', 'ground truth'])
    plt.show()