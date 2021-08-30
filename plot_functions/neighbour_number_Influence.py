
import matplotlib.pyplot as plt
import  numpy as np
def plot_neighbour_number_Influence(neighbor_number,estimation_error_depth,estimation_error_velocity,v_error_array_cen, d_error_array_cen):

    for i in range(len(estimation_error_depth)):
        # time step for plot
        time_step = np.arange(0, len(np.array(estimation_error_depth[i])))
        # time_step_ATC = np.arange(1, 28)
        plt.subplot(2, 1, 2)

        plt.plot(time_step, estimation_error_velocity[i], label='L='+" " +str(int(neighbor_number[i])))
        #  plt.plot(time_step_ATC, local_velocity, label='distributed velocity estimation error $E_v$ with ATC')

        plt.xlabel('Iteration')
        plt.ylabel('Estimation error')
        if i==0:
            plt.plot(time_step, v_error_array_cen, linestyle='dashed',
                 label='centralized velocity estimation error $E_v^c$', color='red')

        plt.legend()
        plt.title(' Velocity estimation performance ')

        plt.tight_layout()

        plt.subplot(2, 1, 1)
        if i==0:
           plt.plot(time_step, d_error_array_cen, linestyle='dashed',label='centralized depth estimation error $E_d^c$', color='red')
        plt.plot(time_step, estimation_error_depth[i], label='L=' + " " + str(int(neighbor_number[i])))

        plt.xlabel('Iteration')
        plt.ylabel('Estimation error')
        plt.legend(loc="center right",  # Position of the legend
                   borderaxespad=0.1  # Add little spacing around the legend box
                   )
        plt.title(' Depth estimation performance ')

        plt.tight_layout()
    plt.show()
