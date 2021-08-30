#function to plot estimation performance of ATC-framework
import matplotlib.pyplot as plt
import numpy as np
#input args:
#          velocity_plot: flag to plot velocity estimation result
#          v_layer: estimated layer velocity for each layer
#          receiver_offset: receiver distance to source
#          layer_velocity: ground truth layer velocity
#          ground_depth: estimated layer depth
#          test_depth: ground truth layer depth
def estimation_ATC_performance(j,velocity_plot,v_layer,receiver_offset,layer_velocity,ground_depth,test_depth):
    # plot estimated depth and velocity from learning parameter and ground truth
    # legend for plotting

    if velocity_plot == 1:
        for i in range(len(v_layer)):
            distance = [-receiver_offset[::-1], receiver_offset]
            final_v=[v_layer[i][::-1], v_layer[i]]
            plt.plot(np.array(distance).flatten(), np.array(final_v).flatten(), label='estimated layer velocity')
            plt.scatter(distance, layer_velocity[i] * np.ones(2*len(v_layer[i])), label='ground truth')
            plt.xlabel('receiver offset (m)')
            plt.ylabel('layer velocity (m/s)')
            title='estimated layer velocity for distributed NMO with ATC at iteration'+" " +str(int(j)+1)
            plt.title(title)
            # plot estimated depth and velocity from learning parameter and ground truth
            if i == 0:
                plt.legend(bbox_to_anchor=(1, 0.5), loc='upper right')
        plt.show()
        for i in range(len(v_layer)):
            distance = [-receiver_offset[::-1], receiver_offset]
            depth = [np.array(ground_depth[i]).flatten()[::-1], np.array(ground_depth[i]).flatten()]
            plt.plot(np.array(distance).flatten(), -np.array(depth).flatten().T, label='estimated layer depth')
            plt.scatter(distance, -test_depth[i + 1] * np.ones(2*len(v_layer[i])).T, label='ground truth')
            plt.xlabel('receiver offset (m)')
            plt.ylabel('layer depth (m)')
            title='estimated Subsurfaces for distributed NMO with ATC at iteration'+" " +str(int(j)+1)
            plt.title(title)
            if i == 0:
                plt.legend(bbox_to_anchor=(1, 0.5), loc='upper right')
        plt.show()