
import matplotlib.pyplot as plt
def STA_ratio_plot(ratio,time_array,time_array1, newdesired):
    # plot ratio
    plot_flag = 1

    if plot_flag == 1:
        print(len(ratio))
        plt.plot(time_array[0:len(ratio) - 1], ratio[0:len(ratio) - 1], label='STA/LTA ratio')
        plt.plot(time_array1[0:60000], newdesired[0:60000], label='ground truth')
        plt.xlabel('time (s)')
        plt.ylabel('STA/LTA ratio')
        plt.title('STA/LTA ratio variation')
        plt.legend()
    plt.show()