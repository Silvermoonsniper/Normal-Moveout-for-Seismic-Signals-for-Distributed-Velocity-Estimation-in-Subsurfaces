
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from IPython.display import HTML
from IPython.display import display, Image


##animation for seismic wavelet motion in whole layered earth model
# input args:
#  time_grid_pressure: wave amplitude for all cells at different time instant
def video_wave_propagation(time_grid_pressure, distance, test_depth):
    # wave amplitude of whole computational domain in first 200 steps
    a = np.array(time_grid_pressure[0:200]).flatten()
    # maximum and minimum wave amplitude for display
    minimum = a.min()
    maximum = abs((a.max()))
    data_array = []
    time_array = []
    # the number of grids for display
    n = 120
    J = 120
    # the distance between source and receiver (unit:m)

    delta_x = distance / n

    x = np.linspace(0, distance, num=n)
    h = np.linspace(-distance, -test_depth.max(), num=J)
    h1 = np.linspace(0, -distance, num=J)
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()

    #ax.set_xlim((0, distance))
    #ax.set_ylim((-test_depth.max()), 0)


    # time rounds for simulation
    time_rounds = 6000
    # total time for simulation (unit:s)
    time = 1

    # dicretize sampling time
    delta_t = time / time_rounds
    X, Y = np.meshgrid(x, h)
    Z = np.zeros((n, J))
    X1, Y1 = np.meshgrid(x, h1)
    Z1 = np.zeros((n, J))
    # wave amplitude of whole model for given time instant range
    a = np.array(time_grid_pressure[0:300])
    # interval for display
    interval = 10
    intervalvalue = 50 * delta_t * 1e4
    # time point array
    time = np.arange(0, len(a), interval)
    # Set the colormap and norm to correspond to the data for which
    ims = []

    for k in time:

        local_amplitude = a[k]

        for i in range(n):
            for j in range(J):
                if np.linalg.norm((local_amplitude[j][i])) != 0:
                    Z[i, j] = np.linalg.norm((local_amplitude[j][i + 120]))
                else:
                    Z[i, j] = np.linalg.norm(local_amplitude[j][i])

        imge = plt.pcolormesh(X, Y, Z, vmin=0, vmax=0.05 * maximum, cmap='hot')
        for i in range(n):
            for j in range(J):
                if np.linalg.norm((local_amplitude[j][i])) != 0:
                    Z1[i, j] = np.linalg.norm((local_amplitude[j][i]))
                else:
                    Z1[i, j] = np.linalg.norm(local_amplitude[j][i])
        # vmax is maximum amplitude for displaying
        imge1 = plt.pcolormesh(X1, Y1, Z1, vmin=0, vmax=0.05 * maximum, cmap='hot')
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        plt.title('Ricker Wavelet Motion with Mur ABC')
        ims.append([imge, imge1])
    ani = animation.ArtistAnimation(fig, ims, interval=intervalvalue, blit=True,
                                    repeat_delay=1000)

    bar = plt.colorbar(imge)
    HTML(ani.to_html5_video())
    plt.show()
    return ani


