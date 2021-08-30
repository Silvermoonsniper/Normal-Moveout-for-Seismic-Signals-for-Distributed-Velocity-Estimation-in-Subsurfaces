import  numpy as np
import matplotlib.pyplot as plt
# a function to select different receiver array with different inter-receiver distance
import ground_truth_arrival_time_generator
from initial_paramters_setting import initial_parameters_setting
from main_algorithms.centralized_NMO import vel_depth_estimator, t0_solver
class distance_effect():
   def __init__(self,initial_args):
      for key in initial_args:
         setattr(self, key, initial_args[key])
#number of cells in horizontal direction
      self.n = 120
#length of computational region
      self.distance = 28
#number of layers
      self.layer_n = len(initial_args["layer_velocity"])
      # calculate thickness of each layer
      self.depth = initial_args["test_depth"][1:] - initial_args["test_depth"][0:len(initial_args["test_depth"]) - 1]
      # time resolution in ground truth series
      self.time_rounds1=1e5
      self.delta_t1 = self.time / self.time_rounds1
      # time array for plotting ground truth sparse series
      self.time_array1 = np.linspace(1, self.time_rounds1, num=self.time_rounds1) * self.delta_t1

      #space resolution
      self.delta_x=self.distance/self.n
      self.inter_receivernumber = np.arange(1, 10)
# source cell
      self.source_cell=60
# inter receiver distance
      self.inter_receiverdistance = np.arange(1, 10) * np.array(self.delta_x)
      self.receiver_spaceindex = np.arange(self.source_cell+1, self.n - 1) - self.source_cell
      self.receiver = self.receiver_spaceindex * np.array(self.delta_x)
      self.receiver_number = 6

#input args:
#        receiver_number: number of receivers
#        delta_x: size of cell in x direction in FDTD
#        inter_receivernumber: indice of receivers in receiver array
#        receiver_spaceindex: indice of all ground cells on one side of source, as source is
#        always located at center of ground line
#output args:
#       sort_receiver: distance of each receiver with respect to source
#       indice: indices of chosen receivers
   def receiver_array_selector(self,receiver_number, delta_x,inter_receivernumber,receiver_spaceindex ):
       sort_receiver = []
       indice = []
       for j in inter_receivernumber:
           single_array = np.zeros([receiver_number, 1])
           single_array_indice = np.zeros([receiver_number, 1])
           a = 0
           last_indice = 1
           while last_indice < receiver_spaceindex[-1] - 1 and a < receiver_number:
            # assign different reciever offset
               single_array[a] = (last_indice + 1) * np.array(delta_x)
               single_array_indice[a] = last_indice
               last_indice = last_indice + j
               a += 1

           sort_receiver.append(single_array)
        # indice
           indice.append(single_array_indice)
       return sort_receiver, indice





# select ground truth arrival time with different inter-receiver distance
# input args:
#           newsynthetic_arriavl：ground truth for all receivers
#           trace_number: indice of selected receiver
#output args:
#           finaltime: ground truth arrival time at receivers
   def ground_truth_selector(self,newsynthetic_arriavl, trace_number,layer_n):
       time = []
       for i in range(len(newsynthetic_arriavl)):
          if i in trace_number:
            time.append(newsynthetic_arriavl[i])
       print(time)
       finaltime = np.array(time).reshape([len(trace_number), layer_n]).T
       return finaltime





# investigate how the distance between receivers influence estimation error
# input args:
#          finaltime: ground truth arrival time of reflection wave at selected receivers
#          receiver_distance: distance of receivers to source
#          layer_velocity: ground truth velocity of each layer
#          test_depth: ground truth depth of each layer
   def inter_receiver_distance_effect(self,finaltime, receiver_distance, layer_velocity, test_depth):
    # receiver offset and arrival time
       synthetic_offset = np.array([receiver_distance, receiver_distance, receiver_distance]).flatten()
       synthetic_arriavl = sorted(np.array(finaltime).flatten())
    # rechoose time and space discretization level
       diff = []
       diff_offset = []
       alpha = 1
       for l in range(len(synthetic_arriavl) - 1):
           diff.append(synthetic_arriavl[l + 1] - synthetic_arriavl[l])

           diff_offset.append(abs(synthetic_offset[l + 1] - synthetic_offset[l]))

       delta_t = alpha * np.array(diff).min()

       delta_x = alpha * np.array(diff_offset).min()
    # receiver  ground truth arrival time
       synthetic_arriavl = sorted(np.array(finaltime).flatten())


       peak, offset = [], []
       for j in synthetic_arriavl:
           peak.append(int((j / delta_t)) * delta_t)
       for k in synthetic_offset:
           offset.append((k / delta_x) * delta_x)
    # reshape arrival time and offset array
       peak = np.array(peak).reshape(len(layer_velocity), len(np.array(receiver_distance)))
       offset = np.array(offset).reshape(len(layer_velocity), len(np.array(receiver_distance)))
       #ground truth t0

       t0d=t0_solver(test_depth, layer_velocity)
       optimal_flag=0
    # estimated velocity and depth from NMO
       ground_depth, v_layer, t0coff = vel_depth_estimator(offset, peak,layer_velocity, t0d, optimal_flag)
    # ground truth of arrival time
       synthetic_arriavl = np.array(synthetic_arriavl).reshape(len(layer_velocity), len(np.array(receiver_distance)))
    #ground truth t0
       t0d1=t0d
    # calcualte value of function f
       f = []
       f_new = []
       for j in range(len(peak)):
           term = np.sqrt(abs( (peak[j] **2- t0coff[j]**2)))
           ground_truth = (2 * (finaltime[j] - t0d1[j]) * t0d1[j])

        #    K=1/np.sqrt(synthetic_arriavl[j]**2-t0d1[j]**2)
           f_new.append(np.array(abs(t0coff[j] - t0d1[j]) / t0d1[j]))
           f.append(np.array((receiver_distance / term)).mean())
    # average f
       f_mean = np.array(f).flatten()

       f_new = np.array(f_new).flatten().mean()
    # calculate estimation error
       depth_error, velocity_error = [], []
       a = 1
       ground_depth = ground_depth[0:len(layer_velocity)]
       v_layer = v_layer[0:len(layer_velocity)]
       for j in ground_depth:
           depth_error.append(np.array(abs(j - test_depth[a])) / test_depth[a])
           a += 1
       b = 0
       for k in v_layer:
           velocity_error.append(np.array(abs(k - layer_velocity[b])) / layer_velocity[b])
           b += 1
    # average over all layers
       finaldepth_error = np.array(depth_error[0:len(layer_velocity)]).mean()
       finalvelocity_error = np.array(velocity_error[0:len(layer_velocity)]).mean()
       return finaldepth_error, finalvelocity_error, f_mean, f_new



# test for different receiver distance

   def distance_effect_simulation(self,inter_receiverdistance,inter_receivernumber,receiver_spaceindex,newsynthetic_arriavl,delta_x,receiver_number,layer_velocity, test_depth):
    # get different receiver array
       sort_receiver, indice = self.receiver_array_selector(receiver_number, delta_x,inter_receivernumber,receiver_spaceindex )
       depth, vel, function_f, function_f1 = [], [], [], []
       for j in range(len(indice)):
           finaltime = self.ground_truth_selector(newsynthetic_arriavl, indice[j],len(layer_velocity))

        # obtain estimation error
           finaldepth_error, finalvelocity_error, f, f_new = self.inter_receiver_distance_effect(finaltime, sort_receiver[j],
                                                                                         layer_velocity, test_depth)
           depth.append(finaldepth_error)
           vel.append(finalvelocity_error)
        # plot function f
           function_f.append(f)
           function_f1.append(f_new)
       function_f = np.array(function_f).T
       function_f1 = np.array(function_f1).T

       function_f[2][0] = 0
    # final_product=np.multiply(inter_receiverdistance[0:8],1/np.array(function_f[0:8]))
       plot_flag = 0
       if plot_flag == 0:
           plt.subplot(2, 1, 1)
        # plot results
           plt.plot(inter_receiverdistance[0:8], depth[0:8])
           plt.plot(inter_receiverdistance[0:8], vel[0:8])
           plt.title('The velocity and depth estimation error')
           plt.xlabel('inter-receiver distance (m)')
           plt.ylabel('normalized estimation error')
           plt.legend(['velocity estimation error $\hat{\epsilon}_v$', 'depth estimation error $\hat{\epsilon}_d$'])
           plt.tight_layout()
           plt.subplot(2, 1, 2)
           plt.plot(inter_receiverdistance[0:8], function_f[0][0:8])
           plt.plot(inter_receiverdistance[0:8], function_f[1][0:8])
           plt.plot(inter_receiverdistance[0:8], function_f[2][0:8])

       plt.title('f($t_{d,f},\hat{t}_{0,f},x_d$) with different inter-receiver distance')
       plt.xlabel('inter-receiver distance (m)')
       plt.ylabel('f($t_{d,f},\hat{t}_{0,f},x_d$)')
       plt.legend(['layer 1', 'layer 2', 'layer 3'])
       plt.tight_layout()

       plt.show()
        # plt.plot(inter_receiverdistance[0:8],function_f1[0][0:8])
        # plt.plot(inter_receiverdistance[0:8],function_f1[1][0:8])
       plt.plot(inter_receiverdistance[0:8], function_f1[0:8])
       plt.title('g(${\hat{t}_{0,f}}$) with different inter-receiver distance')
       plt.xlabel('inter-receiver distance (m)')
       plt.ylabel('g(${\hat{t}_{0,f}}$)')
       plt.legend(['g(${\hat{t}_{0,f}}$)'])
       plt.show()
#investigate how inter-receiver distance influence estimation error
if __name__ == '__main__':
 # generate ground truth arrival time data from multiple receivers for each layer
   initial_args = initial_parameters_setting()
 #call the class
   self=distance_effect(initial_args)
   desired, indexarray, newsynthetic_arriavl = ground_truth_arrival_time_generator.real_signal(self.layer_n,
                                                                                                    self.layer_velocity,
                                                                                                    self.depth,
                                                                                                    self.receiver,
                                                                                                    self.delta_t1,
                                                                                                    self.time_array1)
   newsynthetic_arriavl=np.array(newsynthetic_arriavl).reshape(self.source_cell-2,self.layer_n)

   self.distance_effect_simulation(self.inter_receiverdistance,self.inter_receivernumber,self.receiver_spaceindex, newsynthetic_arriavl, self.delta_x, self.receiver_number, self.layer_velocity,
                        self.test_depth)

