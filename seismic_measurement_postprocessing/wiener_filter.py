from scipy import signal
def shift(seq, n):
    a = n % len(seq)
    return seq[-a:] + seq[:-a]


from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
# implement a wiener filter
#input args:
#      data: signal we wanna to filter
#      input_signal: signal we wanna to approximate
#      delta_t: time resolution
#output args:
#      correlation: pearson correlation between signal want to approxiamte and filtered signal
#      final_output: filtered signal by wiener filter
#      true：original noisy seismic trace
def wiener_filtering(data, input_signal, delta_t):
    # construct shift raw seismic sequence
    # initialzie matrix to store autocorrelation values
    auto_value = np.zeros([len(data), len(data)])
    column = []

    # origial trace data
    original_data = data
    for k in range(len(data)):
        lag_signal = np.pad(data, k, 'constant')
        # calculate autocorrelation
        a = np.correlate(data, lag_signal[0:len(data)])

        # fill autocorrelation matrxi
        column.append(a)
    column_val = np.array(column).flatten()

    # shift column to construct final autocorrelation matrix
    for j in range(len(column_val)):
        val = np.array(shift(list(column_val), n=j))
        auto_value[j][:] = val
    auto_value=np.eye(len(data))
    # calculate cross correlation between desired sequence and raw sequence
    cr_value = []

    # for i in range(peak_source):
    newinput_signal = np.pad(input_signal, 0, 'constant')
    cr = signal.correlate(original_data, newinput_signal[0:len(original_data)])
    for k in range(len(cr)):
        cr_value.append(cr[k])

    # reshape array
    transform = np.array(auto_value).flatten()

    # calculate wiener filter coeffiiceinets
    intermedaite = np.linalg.inv(np.array(auto_value))

    wiener_coff = np.matmul(np.array(intermedaite), np.array(cr_value[0:len(original_data)]))
    wiener_coff = wiener_coff[::-1]
    # calculate output sequence after filter

    inputvalue = np.array(data)

    output = np.convolve(wiener_coff, data)

    final_output = output
    # true filtered signalnp.convolve(ricker_amp[0:220],output)

    # normalize input signal and estimated signal
    maximum = np.array(final_output).max()
    final_output = final_output / maximum
    true = input_signal

    maxim = np.array(true).max()
    true = true / maxim
    # calculate pearson correlation between originial signal and estimated signal
    correlation = np.dot(final_output[0:len(data)-1], true[0:len(data)-1]) / (
                np.linalg.norm(final_output[0:len(data)-1]) * np.linalg.norm(true[0:len(data)-1]))

    # return correlation and filtered noisy seismic trace

    return correlation, final_output,true
# input signal

# correlation,final_output=wiener_filtering(noisy_trace[0][0:5000],post_processing1[50][0:5000],delta_t)
