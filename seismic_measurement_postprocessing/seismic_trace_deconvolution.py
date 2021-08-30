import numpy as np
import matplotlib.pyplot as plt
#compute convolutional matrix of wavelet
#input args:
#      len_ref: length of reflectivity series applied for deconvolution
#      wavelet_data: ricker wavelet impulse
#output args:
#     convolution_matrix: convolutional matrix of wavelet

def convolution_matrix_gen(len_ref, wavelet_data):
    # length of ricker wavelet
    len_wavelet = len((wavelet_data[0:100]))
    # ricker wavelet that is used for constructing convolution matrix of
    #wavelet
    wavelet = wavelet_data[0:100]
    pad_wavelet = []
    # number of rows in convolution matrix
    number_row = len_ref + len_wavelet
    # pad for wavelet sequence

    pad_wavelet = np.pad(wavelet, len_ref, 'constant')

    # flip wavelet
    pad_wavelet = pad_wavelet

    convolution_matrix = np.zeros([number_row, len_ref], dtype='float32')
    for j in range(number_row):
        data = pad_wavelet[j:j + len_ref]
        #  print(len(data[::-1]),len(convolution_matrix[j]))
        convolution_matrix[j] = (data[::-1])
    return convolution_matrix


# deconvolve seismic trace to get reflectivity funtion of seismic trace, then we could select maximas correspond
# arrival of reflection wave






# deconvolve seismic trace to get reflectivity funtion of seismic trace, then we could select maximas correspond
# arrival of reflection wave





#function to implement a deconvolution operator to noisy seismic trace to recover reflectivity
#input args:
        #noisy_flag: flag to idenity noisy measurement
        # ricker_amp: ricker wavelet data
        # post_processing: noisy measurement at multiple receivers
        # r: receiver indice
        # standard_deviation:
        # delta_t1: time resolution to plot ground truth series
#output args:
#       LS: reflectivity series
#       peak_source: shifted interval, as peak in source wavelt is not at zero time point

def regularized_LS_estimator(noisy_flag,fm, time_rounds,ricker_amp, post_processing, r, standard_deviation, delta_t1):
    len_refl = time_rounds
    # length of source wavelet
    lengeth_wavelet = 100
    #calculate standard deviation with SNR


    # caluclate regularization coefficient
    rho_r = 1 / (standard_deviation ** 3)

    # calculate convolution matrix of wavelet
    convolution_matrix = convolution_matrix_gen(len_refl, ricker_amp[0:lengeth_wavelet])
    # estimate of reflectivity
    w_wt = np.matmul(convolution_matrix.T, convolution_matrix)
    # shape

    inverse_term = (w_wt / (standard_deviation ** 2) + rho_r * np.eye(len(w_wt[0])))

    inverse = inverse_term.flatten().reshape(len_refl, len_refl)
    # second term
    # check if wwt is singular
    # calculate its determinant
    determinant = np.linalg.det(inverse)
    if determinant == 0:
        # regularization parameter

        inverse = inverse + rho_r * np.eye(len(w_wt[0]))

    # normalize seismic trace
    # Wtd_j term in the regularized formula


    if noisy_flag==1:
        post_processing[r]=np.array(post_processing[r]).flatten()
        second = np.matmul(convolution_matrix, post_processing[r][0:len_refl + lengeth_wavelet])
    else:
        second = np.matmul(convolution_matrix.T, post_processing[r][0:len_refl + lengeth_wavelet])
    # estimate reflectivity at time step k+1
    LS = np.matmul(np.linalg.inv(inverse), second[0:len_refl])

    # normalize
    LS = LS / (np.array(LS).max()-np.array(LS).min())

    time_shift = 5 * np.sqrt(2) / (2 * np.pi * fm)

    # peak index of source wavelet and shift the ground truth series
    peak_source = int(1 * time_shift / delta_t1)
    # convolve with source wavelet
    original_trac = np.convolve(LS, ricker_amp[0:lengeth_wavelet])
    original_trac = original_trac / original_trac.max()
    return LS, peak_source




# function to get reflectivity series for multiple receivers
# input args:
#          trace_number: seismic indice number array
#          filter_trace: filtered seismic trace
#          ricker_amp: ricker wavelet
#          post_processing1: noiseless seismic trace
#          trace_number: indice array  of receivers
#          standard_deviation_array: standard deviation of AWGN for each noisy profile
#          time_array1: time array for plotting in the ground truth series
#          delta_t1: time resolution in time_array1
#output args:
#        reflectivity: multiple reflectiviy series at multiple receivers
#        peak_source； shifted time interavl


def multiple_reflectivity_retriever(time_rounds,noisy_final, fm,ricker_amp, post_processing1, trace_number, standard_deviation_array,
                                    delta_t1):
    # array to store multiple reflectivity series after denoising
    reflectivity = []
    noisy_flag = 1
    a=0
    if noisy_flag == 0:
        for j in trace_number:
            ref, peak_source = regularized_LS_estimator(noisy_flag, fm,time_rounds,ricker_amp, post_processing1, j, standard_deviation_array[a],
                                                        delta_t1)
            a+=1
            reflectivity.append(ref)
    else:
        for j in range(len(trace_number)):
            ref, peak_source = regularized_LS_estimator(noisy_flag,fm,time_rounds, ricker_amp, noisy_final, j, standard_deviation_array[a],
                                                        delta_t1)
            reflectivity.append(ref)
            a+=1
    return reflectivity,peak_source

