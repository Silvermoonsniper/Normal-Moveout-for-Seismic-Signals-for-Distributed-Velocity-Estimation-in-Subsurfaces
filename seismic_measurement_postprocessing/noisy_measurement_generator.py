# add noise to measured signal
from scipy.linalg import sqrtm
import math
import numpy as np
import matplotlib.pyplot as plt
from wiener_filter import *

# function to generate noisy sesimic trace with AWGN
#input args:
#       post_processing1: noiseless seismic trace
#       SNR: signal-to-noise ratio of noisy seismic profile
#output args:
#      noisy_trace: noisy seismic trace at single receiver
#      standard_deviation: corresponding standard deviation of AWGN calculated from SNR
def noisy_trace_generator(post_processing1, SNR):
    noisy_trace = []


    # add awgn
    #calculate standard deviation of noise
    standard_deviation=np.sqrt(np.var(post_processing1))/(10**(SNR/20))
  #  standard_deviation=5e-4
   # print(10*math.log(standard_deviation**2/np.var(post_processing1)))
    AWGN = standard_deviation * np.random.randn(1, len(post_processing1))
    noisy_trace.append(np.add(post_processing1[0:len(post_processing1)], AWGN.flatten()[0:len(post_processing1)]))



    return noisy_trace,standard_deviation



# function to gather targeting traces from a receiver array
#input args:
#        post_processing1:noiseless seismic trace
#        trace_number: indice array of receiver array
#output args:
#        target: seismic measurement for a given receiver array

def target_trace(post_processing1, trace_number):
    target = []
    for j in range(len(post_processing1)):
        if j in trace_number:
            target.append(post_processing1[j])
    return target

#function to get multiple noisy seismic measurment at receiver array
#input args:
#       post_processing1:noiseless seismic trace
#       delta_t: time resolution
#       SNR: signal-to-noise ratio of noisy seismic profile
#output args:
#      noisy_final:multiple noisy seismic trace
#      filter_trace: multiple filtered noisy seismic trace
#      standard_deviation_array: standard deviation of AWGN for each noisy profile
def multiple_noisy_trace(post_processing1, delta_t, SNR):
    noisy_final = []
    filter_trace = []
    standard_deviation_array=[]
    clean_seismic_trace=[]
    for j in post_processing1:
        noisy_trace,standard_deviation = noisy_trace_generator(j, SNR)
        # apply wiener filter for denoising
        intnoisy_trace = np.array(noisy_trace).flatten()
        if len(j)>5e3:
            correlation, final_output, true = wiener_filtering(intnoisy_trace[0:5000], j[0:5000], delta_t)
        else:
            correlation, final_output,true = wiener_filtering(intnoisy_trace[0:len(j)], j[0:len(j)], delta_t)
        # append noisy sesimic trace
        noisy_final.append(noisy_trace)
        #append clean seismic trace
        clean_seismic_trace.append(true)
        # append filtered noisy trace
        filter_trace.append(final_output)
        #append for standard deviation of noisy measurement
        standard_deviation_array.append(standard_deviation)

    return noisy_final, filter_trace,clean_seismic_trace,standard_deviation_array

