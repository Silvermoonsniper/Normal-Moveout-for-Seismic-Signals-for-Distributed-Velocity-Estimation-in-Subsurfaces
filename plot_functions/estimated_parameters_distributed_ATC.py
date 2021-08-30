# plot estimated parameters
import matplotlib.pyplot as plt
import numpy as np
#input args:
#     t0_plotflag: flag to plot estimated t0 variation along different time steps
#     t0：estimated t0 array for all iterations
#     m0： estimated m0 array for all iterations
#     time_step：total time step for running SGD
#     ground_truth_t0：ground truth value of parameter t0 for all layers
#     ground_truth_m0：ground truth value of parameter m0 for all layers

def estimated_parameters_plot(t0_plotflag, t0, m0, time_step, ground_truth_t0, ground_truth_m0):
    # legend for plotting
    legend_m0 = ['ground truth $u_{1,j}$', 'ground truth $u_{2,j}$', 'ground truth $u_{3,j}$']
    legend_t0 = ['ground truth $t_{0,d,1}$', 'ground truth $t_{0,d,2}$', 'ground truth $t_{0,d,3}$']
    label_t0=['$estimated {t}_{0,d,1}$','$estimated {t}_{0,d,2}$','$estimated {t}_{0,d,3}$']
    title=['estimated ${t}_{0,d,1}$ at different iteration','estimated ${t}_{0,d,2}$ at different iteration','estimated ${t}_{0,d,3}$ at different iteration']
    label_m0 = ['$estimated {u}_{d,1}$', '$estimated {u}_{d,2}$', '$estimated {u}_{d,3}$']
    titlem0 = ['estimated ${u}_{d,1}$ at different iteration', 'estimated ${u}_{d,2}$ at different iteration',
             'estimated ${u}_{d,3}$ at different iteration']

    for k in range(len(t0)):
        if t0_plotflag == 1:
            plt.subplot(3, 1, k + 1)

            for j in range(len(t0[k].T)):
                plt.plot(np.arange(time_step), t0[k].T[j])

                plt.xlabel('Iteration')
                plt.ylabel(label_t0[k])
               # if k == 0:
                plt.title(title[k])
            plt.scatter(np.arange(time_step), np.ones(time_step) * ground_truth_t0[k], label=legend_t0[k])
            plt.legend()
            plt.tight_layout()

        elif t0_plotflag == 2:
            plt.subplot(3, 1, k + 1)
            for j in range(len(t0[k].T) - 1):
                plt.plot(np.arange(time_step), m0[k].T[j])

                plt.xlabel('Iteration')

                plt.ylabel(label_m0[k])
                plt.title(titlem0[k])
            plt.scatter(np.arange(time_step), np.ones(time_step) * ground_truth_m0[k], label=legend_m0[k])
            plt.legend()
            plt.tight_layout()
    plt.show()
# function to retieval estimate at given iteration
# input args:
            #    iteration: iteration we want to visualize
            #    t0_array: estimated t0 for all iterations
            #    m0_array； estimated m0 for all iterations
#output args:
#          local_t0: estimated parameter two-way vertical travel time t0 at particular time step
#          local_m0: estimated parameter slowness m0 at particular time step

def local_estimation(iteration, t0_array, m0_array):
    local_t0 = []
    local_m0 = []
    for j in t0_array:
            a = 0
            for l in j:
                if a == iteration:
                            local_t0.append(l)
                a += 1
    for k in m0_array:
            b = 0
            for p in k:
                if b == iteration:
                            local_m0.append(p)

                b += 1
    return local_t0, local_m0