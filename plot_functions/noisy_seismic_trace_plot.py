
import matplotlib.pyplot as plt
import  numpy as np
#plot noisy seismic trace
#input args:
#         ricker_amp: ricker wavelet impulse
#         time_array: time series for plot
#         noisy_trace: noisy seismic measurement
def noisy_trace_plot(ricker_amp,time_array,noisy_trace,picking_travel_time_method):
    #normalize noisy seismic trace
    max=np.array(noisy_trace).flatten().max()
    noisy_trace=noisy_trace/max
    if picking_travel_time_method=='STA_LTA':
        plt.plot(time_array[0:len(time_array)], noisy_trace[0][0:len(time_array)])
    else:
        plt.plot(time_array[0:len(time_array)], noisy_trace[0:len(time_array)])

    plt.plot(time_array[0:2000],ricker_amp[0:2000])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Noisy seismic measurement $\mathbf{z}_d^n$')
    plt.legend(['noisy seismic measurement $\mathbf{z}_d^n$','Ricker wavelet $\mathbf{r}_d$'])
    plt.show()