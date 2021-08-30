# plot comparison of true arrival time and picking from STA
import numpy as np
import matplotlib.pyplot as plt
def picking_plot_comparison(final_arrival, receiver_offset,finaltime):




        #reshape for plotting
    final_arrival=np.array(final_arrival).T

    plt.figure(figsize=(6, 6))
    plt.plot(receiver_offset, final_arrival[0][0:len(receiver_offset)], label='picking time $\mathbf{t}_1$')
    plt.plot(receiver_offset, finaltime[0][0:len(receiver_offset)], label='ground truth $\mathbf{t}^r_1$')
    plt.xlabel('receiver offset (m)')
    plt.ylabel('travel time (s)')
    plt.title('The comparison of picking time and ground truth for first layer')
    plt.legend()
    plt.tight_layout()
    plt.show()
