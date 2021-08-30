from scipy import interpolate
import numpy as np

# extract ground truth arrival time of certain receiver array
#input args:
#        trace_number: indice array of receivers
#        newsynthetic_arriavl: ground truth arriavl time of reflection waves at receiver
#        receiver_number:number of receiver in receiver array
#        layer_n:number of layers
#output args:
#        finaltime：selected ground truth arrival time of certain receiver array
#        newsynthetic_arriavl_1: transposed array of ground truth arrival time for all cells on one side

def ground_truth_selector_certainreceivers(trace_number, newsynthetic_arriavl, receiver_number, layer_n):
    # ground truth arrival time for all receivers
    newsynthetic_arriavl_1 = newsynthetic_arriavl.reshape([receiver_number, layer_n])
    time = []

    for i in range(len(newsynthetic_arriavl_1)):
        if i in trace_number:

            time.append(newsynthetic_arriavl_1[i])

    finaltime = np.array(time).reshape([len(trace_number), layer_n]).T
    return finaltime, newsynthetic_arriavl_1


# function to choose time resolution according ground truth
#input args:
#         alpha: factor to control time resolution level
#         max_time:maximum plot time with certain time level defined by alpha
#         min_time: minimum plot time with certain time level defined by alpha
#         synthetic_arriavl: ground truth arrival time
#output args:
#        delta_t: time level
#        new_time_array: time array with this time level for plotting


def time_resolution_generator(alpha, max_time, min_time, synthetic_arriavl):
    # rechoose time and space discretization level
    diff = []
    synthetic_arriavl = sorted(np.array(synthetic_arriavl).flatten())
    for l in range(len(synthetic_arriavl) - 1):
        diff.append(synthetic_arriavl[l + 1] - synthetic_arriavl[l])

    delta_t = alpha * np.array(diff).min()

    # calculate sampling time with maximum and minimum of simulation time
    time_rounds_max = int(max_time / np.array(delta_t))
    time_rounds_min = int(min_time / np.array(delta_t)) + 1
    new_time_array = np.arange(time_rounds_min, time_rounds_max) * np.array(delta_t)
    return delta_t, new_time_array



# construct new deconvolved seismic measurement,
# as memory error may occur in performing matrix inversion
# (unable to allocate very high dimension matrix)
def interpolation(xnew, time_array, post_processing):
    new_post = []
    for j in post_processing:
        f = interpolate.interp1d(time_array[0:len(time_array)-1000], j[0:len(time_array)-1000])

        # caclculate new time resolution
        new_delta_t = xnew[11] - xnew[10]
        fnew = f(xnew)
        new_post.append(fnew)

    newtime_array = xnew
    post_processing = new_post
    return post_processing, new_delta_t

