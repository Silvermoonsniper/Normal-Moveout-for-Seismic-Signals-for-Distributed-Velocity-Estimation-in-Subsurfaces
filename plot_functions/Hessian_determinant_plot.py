#plot determinant of Hessian at different iterations
import matplotlib.pyplot as plt
import  numpy as np
#input args:
#    time_step: time step for SGD
#    determinant_array: determinant of Hessian at all iterations
def Hessian_determinant_plot(time_step,determinant_array):
    #time step array
    time_step_array=np.arange(0,time_step)
    plt.plot(time_step_array[0:time_step-3],determinant_array[0:time_step-3])
    plt.xlabel('Iteration')
    plt.ylabel('det($\mathbf{H}$)')
    plt.title('The determinant of Hessian matrix at different iterations')
    plt.show()