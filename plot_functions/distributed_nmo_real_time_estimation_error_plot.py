#function to plot distributed NMO estimation error with different iteration and centralized one
#plot local estimation error curve
import matplotlib.pyplot as plt
import numpy as np
#input args:
#       v_error_array: normalized average velocity estimation error for all iterations with distributed NMO
#       d_error_array: normalized average depth estimation error for all iterations with distributed NMO
#       v_error_array_cen: normalized average velocity estimation error for all iterations with centralized NMO
#       d_error_array_cen: normalized average velocity estimation error for all iterations with centralized NMO
def plot_estimation_comparison(v_error_array,d_error_array,v_error_array_cen,d_error_array_cen):


    # time step for plot
    time_step = np.arange(0, len(v_error_array))
   # time_step_ATC = np.arange(1, 28)
    plt.subplot(2, 1, 2)

    plt.plot(time_step, v_error_array, label='distributed velocity estimation error $E_v$ with average consensus')
  #  plt.plot(time_step_ATC, local_velocity, label='distributed velocity estimation error $E_v$ with ATC')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')

    plt.plot(time_step, v_error_array_cen, label='centralized velocity estimation error $E_v^c$', color='red')

    plt.legend()
    plt.title(' Velocity estimation performance ')

    plt.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(time_step, d_error_array_cen, label='centralized depth estimation error $E_d^c$', color='red')
    plt.plot(time_step, d_error_array, label='distributed depth estimation error with average consensus $E_d$')
  #  plt.plot(time_step_ATC, local_error, label='distributed depth estimation error $E_d$ with ATC')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')
    plt.legend(loc="center right",        # Position of the legend
           borderaxespad=0.1         # Add little spacing around the legend box
         )
    plt.title(' Depth estimation performance ')

    plt.tight_layout()
    plt.show()

def plot_sensornumber_comparison(v_error_array,d_error_array,v_error_array_cen,d_error_array_cen,v_error_array1,d_error_array1,v_error_array_cen1,d_error_array_cen1):


    # time step for plot
    time_step = np.arange(0, len(v_error_array))
   # time_step_ATC = np.arange(1, 28)
    plt.subplot(2, 1, 2)

    plt.plot(time_step, v_error_array, label='D=24')
  #  plt.plot(time_step_ATC, local_velocity, label='distributed velocity estimation error $E_v$ with ATC')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')

    plt.plot(time_step, v_error_array_cen1, label='D=12', color='red')
    plt.plot(time_step, v_error_array1, label='D=12')
    #  plt.plot(time_step_ATC, local_velocity, label='distributed velocity estimation error $E_v$ with ATC')



    plt.plot(time_step, v_error_array_cen, label='D=24')

    plt.legend()
    plt.title(' Velocity estimation performance ')

    plt.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(time_step, d_error_array, label=' D=24')
    #  plt.plot(time_step_ATC, local_velocity, label='distributed velocity estimation error $E_v$ with ATC')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')

    plt.plot(time_step, d_error_array_cen1, label='D=12', color='red')
    plt.plot(time_step, d_error_array1, label='D=12')
    #  plt.plot(time_step_ATC, local_velocity, label='distributed velocity estimation error $E_v$ with ATC')

    plt.plot(time_step, d_error_array_cen, label='D=24')
    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')
    plt.legend(loc="center right",        # Position of the legend
           borderaxespad=0.1         # Add little spacing around the legend box
         )
    plt.title(' Depth estimation performance ')

    plt.tight_layout()
    plt.show()
#plot function to plot comparison of estimation performance of these three schemes: centralized and two distributed NMO
def plot_scheme_comparison(v_error_array,d_error_array,v_error_array_cen,d_error_array_cen,local_error, local_velocity):
    #time step for plot
    time_step=np.arange(0,len(v_error_array))
    time_step_ATC=np.arange(0,len(local_error))
    plt.subplot(2,1,2)

    plt.plot(time_step,v_error_array,label=' $E_v$ ')
    plt.plot(time_step_ATC, local_velocity, label=' $E_v^{ATC}$')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')

    plt.plot(time_step,v_error_array_cen,label=' $E_v^c$',color='red')



    plt.legend()
    plt.title('Velocity estimation performance ')

    plt.tight_layout()

    plt.subplot(2,1,1)
    plt.plot(time_step,d_error_array_cen,label='$E_d^c$',color='red')
    plt.plot(time_step,d_error_array,label='$E_d$')
    plt.plot(time_step_ATC, local_error, label=' $E_d^{ATC}$')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')
    plt.legend(loc="center right",        # Position of the legend
           borderaxespad=0.1         # Add little spacing around the legend box
           )
    plt.title('Depth estimation performance ')

    plt.tight_layout()
    plt.show()
#plot function to plot estimation performance of ATC distributed NMO estimation
def plot_ATC_estimation_performance(local_error, local_velocity):
    #time step for plot

    time_step_ATC=np.arange(0,len(local_error))
    plt.subplot(2,1,2)


    plt.plot(time_step_ATC, local_velocity, label=' $E_v^{ATC}$')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')




    plt.legend()
    plt.title('Velocity estimation performance ')

    plt.tight_layout()

    plt.subplot(2,1,1)

    plt.plot(time_step_ATC, local_error, label=' $E_d^{ATC}$')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')
    plt.legend(loc="center right",        # Position of the legend
           borderaxespad=0.1         # Add little spacing around the legend box
           )
    plt.title('Depth estimation performance ')

    plt.tight_layout()
    plt.show()
#plot function to plot comparison of estimation performance of different topology:
#   1.line topology
#   2. random topology
def plot_topology_comparison(v_error_array,d_error_array,v_error_array_cen,d_error_array_cen,local_error, local_velocity):
    #time step for plot
    time_step=np.arange(0,len(v_error_array))
    time_step_ATC=np.arange(0,len(local_error))
    plt.subplot(2,1,2)

    plt.plot(time_step,v_error_array,label=' $E_v$ with line topology')
    plt.plot(time_step_ATC, local_velocity, label='$E_v$ with random topology')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')

    plt.plot(time_step,v_error_array_cen,label='$E_v^c$',color='red')



    plt.legend(loc="best",        # Position of the legend
           borderaxespad=0.1         # Add little spacing around the legend box
           )
    plt.title('Velocity estimation performance ')

    plt.tight_layout()

    plt.subplot(2,1,1)
    plt.plot(time_step,d_error_array_cen,label='$E_d^c$',color='red')
    plt.plot(time_step,d_error_array,label='$E_d$ with line topology')
    plt.plot(time_step_ATC, local_error, label='$E_d$ with random topology')

    plt.xlabel('Iteration')
    plt.ylabel('Estimation error')
    plt.legend(loc="center right",        # Position of the legend
           borderaxespad=0.1         # Add little spacing around the legend box
           )
    plt.title(' Depth estimation performance ')

    plt.tight_layout()
    plt.show()
