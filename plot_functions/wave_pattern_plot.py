import numpy as np
import matplotlib.pyplot as plt


# function to plot instantaneous wave propagation
#input args:
#        delta_x: space discretization level in x horizontal direction in FDTD
#        time_grid_pressure: wave amplitude matrix at all time instants
#        distance: whole length of computational region
#        test_depth: the depth calculated from ground surface fro each interface
#        time_instant: the time step of wave propagation pattern we want to display

def local_wave_propagation_visualization(delta_x,delta_h, time_grid_pressure, distance, test_depth, time_instant,interface_coordinate):
    # wave amplitude of whole computational domain in first 200 steps
    a = np.array(time_grid_pressure[0:200]).flatten()
    # maximum and minimum wave amplitude for display
    minimum = a.min()
    maximum = abs((a.max()))
    print(maximum)
    data_array = []
    time_array = []
    # the number of grids for plotting, n should be same of J to use pcolormesh
    n=int(distance/delta_x)
    J = n
    # number of cells in y direction
    Y_cell=int(test_depth.max()/delta_h)

    x = np.linspace(0, distance, num=n)
    h = np.linspace(-distance, -test_depth.max(), num=J)
    h1 = np.linspace(0, -distance, num=J)
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()



    # time rounds for display
    time_rounds = 10000
    # total time for display (unit:s)
    time = 0.25

    # dicretize sampling time
    delta_t = time / time_rounds
    X, Y = np.meshgrid(x, h)
    Z = np.zeros((n, J))
    X1, Y1 = np.meshgrid(x, h1)
    Z1 = np.zeros((n, J))
    # create background for video
    Z2 = np.zeros((n, J))

    Z3 = np.zeros((n, J))
    # wave amplitude of whole model at given time instant

    # interval for display
    interval = 10
    intervalvalue = 50 * delta_t * 1e4
    # time point array
    time = np.arange(0, len(time_grid_pressure), interval)
    ims = []

    # extract wave amplitude at given time instant
    local_amplitude = time_grid_pressure[time_instant]
    #if number of cells in y direction is twice in x direction
    if Y_cell == 2*n:
      for i in range(n):
        for j in range(J):
            if np.linalg.norm((local_amplitude[j][i])) != 0:
                Z[i, j] = np.linalg.norm((local_amplitude[j][i + n]))
            else:
                Z[i, j] = np.linalg.norm(local_amplitude[j][i])
            if i in interface_coordinate:

                Z2[i, j] = 1
            else:
                Z2[i, j] = 0
      for i in range(n):
        for j in range(J):
            if np.linalg.norm((local_amplitude[j][i])) != 0:
                Z1[i, j] = np.linalg.norm((local_amplitude[j][i]))
            else:
                Z1[i, j] = np.linalg.norm(local_amplitude[j][i])
            if i+n in interface_coordinate:
                Z3[i, j] = 1
            else:
                Z3[i, j] = 0
                # plot background to distinguish interface
      img2 = plt.pcolormesh(X1, Y1, Z2, vmin=0, vmax=0.05 * maximum, cmap='hot', animated=True)
      img3 = plt.pcolormesh(X, Y, Z3, vmin=0, vmax=0.05 * maximum, cmap='hot', animated=True)
      imge = plt.pcolormesh(X, Y, Z, vmin=0, vmax=0.05 * maximum, cmap='hot')
      imge1 = plt.pcolormesh(X1, Y1, Z1, vmin=0, vmax=0.05 * maximum, cmap='hot')

      plt.xlabel('Distance (m)')
      plt.ylabel('Depth (m)')
      plt.title('Ricker Wavelet Motion with Mur ABC')
      bar = plt.colorbar(imge)
      bar.set_label('transform amplitude', rotation=270)
    else:
      for i in range(n):
          for j in range(n):
                if np.linalg.norm((local_amplitude[j][i])) != 0:
                    Z1[i, j] = np.linalg.norm((local_amplitude[j][i]))
                else:
                    Z1[i, j] = np.linalg.norm(local_amplitude[j][i])


      imge1 = plt.pcolormesh(X1, Y1, Z1, vmin=0, vmax=0.05 * maximum, cmap='hot')
      plt.xlabel('Distance (m)')
      plt.ylabel('Depth (m)')
      plt.title('Ricker Wavelet Motion with Mur ABC')
      bar = plt.colorbar(imge1)
      bar.set_label('transform amplitude', rotation=270)

    plt.show()