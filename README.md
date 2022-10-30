# Normal-Moveout-of-Seismic-Signals-for-Distributed-Velocity-Estimation-in-Subsurfaces
Master thesis at DLR, institute of Communications and Navigation.  
   1. Provide Synthetic seismic measurement data generation via wave PDE solver  
   2. Inversion algorithm design to estimate subsurface model parameters 
   3. Deconvoluton and Wiener Filter design to perform denosing
   4. Numerical optimization schemes with sensor fusion algorithm implementation via average consensus and adapt-then-combine to achieve autonomous subsurface exploration.

Problem statement <br/>
Seismic exploration is of great importance for future planetary missions: the planet’s subsurface structure can reveal a great deal of information about the planet’s past and present. We envision the use of a network of multiple agents – of a swarm – to autonomously and cooperatively reconstruct the subsurface structure. To compute an image of the subsurface typically several processing steps of seismic measurements are required. One of those steps is a so called normal moveout strategy that can be used to obtain velocity estimates of the subsurface. At present seismic exploration surveys such data processing is conducted in a centralized manner, where all seismic measurements are available at a single computing entity. However, for application in a network of multiple agents these computations need to be conducted in a distributed fashion.

Research objectives <br/>
The goal of project is to implement the normal moveout strategy in a distributed fashion for an estimation of the subsurface velocities and layer depths within a network of geophones. To this end, layered Earth models shall be simulated by solving a wave equation that generates synthetic measurements at the geophone locations. Based on the seismic measurements the objective is to reconstruct the velocities and depths of the layered Earth model in a distributed fashion. The developed scheme shall be tested via simulations with synthetic measurement.

Horizontal subsurface model   
<p align="center"
  <img src="https://user-images.githubusercontent.com/89796179/198896127-24b86b27-3c53-4a11-aa09-52163da99fd5.png" width="400" />
   
</p>

 Wave propagation in homogeneous medium with Mur absorbing boundary condition, the left and right side boundaries are set with Mur absorbing  boundary condition, while the bottom boundary is set to be reflective boundary with Neumann condition.

https://user-images.githubusercontent.com/89796179/198653747-844fc700-88d2-4bdc-b906-6a3b9f18ffa9.mp4



Seismic measurement acquisition and processing <br/>
Deconvolution on seismic measurement to obtain reflectivity series


<p float="left">
  <img src="https://user-images.githubusercontent.com/89796179/198648585-1eaf1978-55de-4b52-95b3-5c2769167015.png" width="400" />
  <img src="https://user-images.githubusercontent.com/89796179/198650218-a910f229-11ec-4a42-a2ad-a23424dbf68c.png" width="400" /> 
 
</p>


 The performance of distributed NMO(Normal Moveout) for seismic wave velocity estimation with 3 layer subsurface model with average consensus <br/>

<p float="left">
  <img src="https://user-images.githubusercontent.com/89796179/198657417-8543778e-0632-4255-9726-c42c50f8ae45.png" width="400" />
  <img src="https://user-images.githubusercontent.com/89796179/198657423-b5a43581-7d03-4924-a98c-cc8c8e6c166e.png" width="400" />
   <br/>
 <img src="https://user-images.githubusercontent.com/89796179/198657429-e3019871-d68c-4388-bece-b7d665698519.png" width="400" />
  <img src="https://user-images.githubusercontent.com/89796179/198657438-f050ae5b-b6fa-44f7-8de0-125e911cefe2.png" width="400" /> 
 
</p>

 The performance of distributed NMO(Normal Moveout) with adapt-then-combine for seismic wave velocity estimation with 3 layer subsurface model
<p float="left">
  <img src="https://user-images.githubusercontent.com/89796179/198895397-25893c1e-07c9-48b6-a153-cad80fb873eb.png" width="400" />
  <img src="https://user-images.githubusercontent.com/89796179/198895398-33003719-634a-4562-896a-ce9e8d9e244e.png" width="400" />
   <br/>
 <img src="https://user-images.githubusercontent.com/89796179/198895399-cf939f74-8f72-4228-9ef9-a1f375219fd7.png" width="400" />
  <img src="https://user-images.githubusercontent.com/89796179/198895400-86044ede-853b-4624-ac1d-e0290340baa3.png" width="400" /> 
 
</p>
