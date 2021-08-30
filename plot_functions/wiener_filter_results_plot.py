import  matplotlib.pyplot as plt
import  numpy as np
#input args:
#       time_array: time series for plotting
#       final_output: filtered seismic measurement after using wiener filter
#       true: original noisy seismic measurement
def wiener_filter_performance_visualization(time_array,final_output,true):
    plot_all = 1
    if plot_all == 1:
        plt.plot(time_array[0:len(time_array)-2], (np.array(final_output[0:len(time_array)-2])))
        plt.plot(time_array[0:len(time_array)-2], (np.array(true[0:len(time_array)-2])))
        # plt.plot(time_array[0:4800],wiener_coff[0:4800])
        plt.legend(['Estimated Signal $\mathbf{y}_d$','Desired Signal $\mathbf{z}_d$'])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Filtered Seismic Measurement via Wiener Filter')
    elif plot_all == 0:
        plt.plot(time_array[0:len(time_array) - 1800], (np.array(final_output[0:len(time_array) - 1800])))
        plt.legend(['Estimated Signal', 'Desired Signal'])
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Filtered Seismic Measurement at Receiver via Wiener Filter')
    plt.show()