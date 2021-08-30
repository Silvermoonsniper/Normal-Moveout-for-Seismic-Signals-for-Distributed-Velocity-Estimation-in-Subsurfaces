from Class_DNMO import Distributed_Normal_moveout_estimator
from initial_paramters_setting import initial_parameters_setting


a=[[0.04501005, 0.04511477, 0.04524343, 0.04546485, 0.04575508, 0.04603035,
  0.04643429, 0.04680231 ,0.04731695, 0.04788844, 0.04838812, 0.04905835],
 [0.07910196, 0.07911692 ,0.07914085, 0.07917975, 0.07922762, 0.0792755,
  0.07934432, 0.07941014, 0.0794999,  0.07960164, 0.07968841, 0.07981108],
 [0.09000514, 0.09000813, 0.09001113, 0.0900201,  0.09002609, 0.09003506,
  0.09004703, 0.090059,   0.09007396, 0.09009191, 0.09010687, 0.09012782]]
import matplotlib.pyplot as plt
initial_args = initial_parameters_setting()


        #here user could modify the parameters from dictionary "initial_args" to
        # test for different subsurface models
        # example, write command as:
        # initial_args["source_to_receiver_distance"]=np.array([2,4,6,7,8])
        # this will set distance of receivers on one side to the shot source as
        # [2,4,6,7,8] meters respectively
        #  initial_args["SNR"] = 5 will set SNR for noisy seismic trace as 10dB
        #Call the class of distributed NMO and Centralized NMO
import numpy as np
receiver=np.array([ 1.16666667,  2.33333333,  3.26666667,  4.43333333,  5.6,         6.53333333,
  7.7,         8.63333333,  9.8,        10.96666667, 11.9,        13.06666667])
R=Distributed_Normal_moveout_estimator(initial_args)
print(R.receiver_distance)


for j in np.array(a):
    l=[j[::-1],j]

    distance=np.array([-receiver[::-1],receiver]).flatten()
   # plt.subplot(3,1,t)
    plt.scatter(distance,np.array(l).flatten())
    plt.xlabel('receiver distance $x_d$')
    plt.ylabel('picking time $\{t}_{d,f}$')
    plt.tight_layout()

    plt.title('picking travel time at receivers')
plt.legend(['layer 1','layer 2','layer 3'])
plt.show()
import math

