import numpy as np
import matplotlib.pyplot as plt
def estimated_parameters_variation(local_information, local_information1 ,t0_truth,m0_truth,receiver_distance):
    a=0
    legend_m0 = [" mean estimated $m_{0}$", " mean estimated $m_{1}$", " mean estimated $m_{2}$"]

    legend=[" mean estimated $t_{0,0}$"," mean estimated $t_{0,1}$"," mean estimated $t_{0,2}$"]

        #plot estimated parameters variation
    plt.subplot(2,1,1)
    plt.plot(receiver_distance,np.array(local_information[2]).T[0])
    plt.scatter(receiver_distance, np.array(t0_truth[2])*np.ones(len(receiver_distance)))
    plt.title(" The comparison of $\hat{t}_{0,d,2}$")
    plt.legend(['estimated $\hat{t}_{0,d,2}$', 'ground truth'])
    plt.xlabel("receiver distance (m)")
    plt.ylabel(" estimated parameter $t_0$")


    plt.tight_layout()
    plt.subplot(2, 1,2)
    plt.plot(receiver_distance,np.array(local_information1[2]).T[0] )
    plt.scatter(receiver_distance, np.array(m0_truth[2]) * np.ones(len(receiver_distance)))
    plt.xlabel("receiver distance (m)")
    plt.ylabel(" estimated parameter $u$")
    plt.legend(['estimated $\hat{u}_{d,2}$','ground truth'])
    plt.title(" The comparison of $\hat{u}_{d,2}$")
    plt.tight_layout()
    plt.show()