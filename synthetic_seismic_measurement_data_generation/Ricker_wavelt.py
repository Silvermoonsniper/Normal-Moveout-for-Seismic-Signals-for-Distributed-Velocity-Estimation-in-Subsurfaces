# this program is aiming to implement Ricker Wavelet and oberve its shape and parametric influecne regarding to shape
# input args:
# 1.fm:the frquency of Ricker wavelet
# 2,time_slot: discretized time array
# output args:
# 1. ricker_amp=ampltide of Ricker wavelet in time-domain

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# maximum simulation time unit:s
max_time = 0.3
# peak frequency (Hz)
fm = 200
# time shift to let Ricker wavelet starts at zero
time_shift = np.sqrt(2) / (2 * np.pi * fm)
time_slot = np.linspace(-time_shift, max_time, num=2e3)

# function to plot ricker wavelet and its spectrum
# input args:
#     fm: peak frequency
#     time_slot: time interavl for plot the ricker wavelet
#output args:
   #  ricker_amp: Ricker wavelet waveform
def Ricker_wavelet(fm, time_slot):
    ricker_amp = []
    spectrumVal = []
    # convert frequency into radians
    fm_radian = 2 * np.pi * fm

    # calculate amplitude
    for i in time_slot:
        amplitude_value = (1 - 0.5 * fm_radian ** 2 * i ** 2) * (np.exp((-0.25 * fm_radian ** 2 * i ** 2)))
        ricker_amp.append(amplitude_value)
    # calculate spectrum

    spectrum = np.fft.fft(ricker_amp)
    # frequency range to plot spectrum
    frq = np.linspace(-0 * np.pi * fm, 6 * np.pi * fm)
    for j in frq:
        spectrumVal.append((2 * j ** 2) / (np.sqrt(np.pi) * fm_radian ** 3) * np.exp(-j ** 2 / fm_radian ** 2))
    # plot for spectrum
    spectrum_flag = 0
    if spectrum_flag == 1:
        plt.plot(frq / (2 * np.pi), spectrumVal)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$R(\omega)$')
        plt.title('Spectrum of Ricker Wavelet')

    return ricker_amp





# here we calculate amplitude of ricker wavelet at given time t
# try to combine Ricker wavelet as input impulse for shot source and estimate arrival time at receiver array

# input args:
#      fm: peak frequncy of ricker wavelet
#      t: time point
def Ricker_response(fm, t):
    # convert frequency into radians
    fm_radian = 2 * np.pi * fm
    # shift the ricker wavelet and let it start at zero
    amplitude_value = (1 - 0.5 * fm_radian ** 2 * t ** 2) * (np.exp((-0.25 * fm_radian ** 2 * t ** 2)))
    return amplitude_value

