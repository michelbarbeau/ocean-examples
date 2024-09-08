# Copyright 2024 Michel Barbeau
############################
# 3D Target Tracking Example
# Author: Michel Barbeau, Carleton University
# Version: 2024/09/07
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DWAVE OCEAN Install
    python -m venv ocean
    . ocean/bin/activate
    SEE: https://docs.ocean.dwavesys.com/en/latest/overview/install.html
    
Usage: 
	python3 quantum-3D-target-tracking-with-obstacles.py
"""

from dimod import ConstrainedQuadraticModel, Integer, quicksum
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

####################################################################
### Test if a 3D coordinate triple included in a list of coordinates
def isIn(pos, poslist):
   # Get the number of elements in the array and
   # convert to a number of 3D coordinates
   n = int( poslist.shape[0] / 3 )
   # Scan the list of coordinates to test inclusion of "pos"
   for k in range(n):
       if pos[0]==poslist[k] and pos[1]==poslist[k + n] and pos[2]==poslist[k + 2*n]:
           return(True)
   return(False)  

# Custom error type
class Collision(Exception):
    pass

###############################
### 3D Target Tracking function
def TargetTracking(n, maxpos, m):
   ### parameters
   # n : number of instants
   # maxpos : index of maximum position
   # m : number of obstacles

   # print("\nNumber of instants: ", n)
   # create the variables, they represent
   # [x_0,x_1,\dots,x_{n-1},y_0,y_1,\dots,y_{n-1},z_0,z_1,\dots,z_{n-1}]
   x = [Integer(f'x_{i}', lower_bound=0, upper_bound=maxpos) for i in range(3*n)]
    
   # generate obstacles' random positions
   rng = np.random.default_rng()
   obs = rng.integers(low=0, high=maxpos, size=3*m)
   print("\nObstacles positions:\n", obs)

   # Generate target's random positions
   pos = np.zeros(3 * n, dtype=int)
   while True: 
      try:
         pos[0] = rng.integers(low=0, high=maxpos)
         pos[n] = rng.integers(low=0, high=maxpos)
         pos[2*n] = rng.integers(low=0, high=maxpos)   
         if isIn([pos[0], pos[n],pos[2 * n]], obs):
            raise Collision
         else:
            break
      except Collision:
         print("\nCollision of target position with obstacle:\n", pos)
         continue # Try again!
   # Generate all other target's random positions
   for i in range(n-1):
      while True: 
         try:
            delta = rng.integers(low=-1, high=2, size=3)  # x,y,z-displacements
            pos[i + 1] = np.min([np.max([pos[i] + delta[0], 0]), maxpos])
            pos[i + 1 + n] = np.min([np.max([pos[i + n] + delta[1], 0]), maxpos])
            pos[i + 1 + 2 * n] = np.min([np.max([pos[i + 2 * n] + delta[2], 0]), maxpos])
            if isIn([pos[i + 1], pos[i + 1 + n],pos[i + 1 + 2 * n]], obs):
               raise Collision
            else:
               break
         except Collision:
            print("\nCollision of target position with obstacle:\n", pos)
            continue # Try again!
                
   print("\nTarget positions:\n", pos)

   # initialize the CQM object
   cqm = ConstrainedQuadraticModel()

   # add the variables to the model
   for i in range(len(x)):
      cqm.add_variable('INTEGER', f'x_{i}', lower_bound=0, upper_bound=maxpos)

   # objective is to minimize the acceleration
   obj = quicksum([ (x[i+2]-2*x[i+1]+x[i])**2 + \
                    (x[i+2+n]-2*x[i+1+n]+x[i+n])**2 + \
                    (x[i+2+2*n]-2*x[i+1+2*n]+x[i+2*n])**2 \
                    for i in range(n-2) ])
   cqm.set_objective(obj)

   # add minimum an maximum chaser-target distance constraints
   smin = 1  # chaser-target minimum distance
   smax = 1  # chaser-target maximum distance
   for i in range(n):
      cqm.add_constraint( ( (x[i]-pos[i])**2+(x[i+n]-pos[i+n])**2+(x[i+2*n]-pos[i+2*n])**2 ) <= smax**2, label=f'xmax_{i}')
      cqm.add_constraint( ( (x[i]-pos[i])**2+(x[i+n]-pos[i+n])**2+(x[i+2*n]-pos[i+2*n])**2 ) >= smin**2, label=f'xmin_{i}')
 
   # add the obstacle avoidance constraints
   for i in range(n):
      for j in range(m):
         cqm.add_constraint( ( (x[i]-obs[j])**2+(x[i+n]-obs[j+m])**2+(x[i+2*n]-obs[j+2*m])**2 ) >=smin**2, label=f'obs_{i*m+j}')
    
   # Initialize the CQM solver
   # hybrid solver
   sampler = LeapHybridCQMSampler()
   # Submit for solution
   answer = sampler.sample_cqm(cqm)

   # Convert chaser's list of positions from dictionary to array
   chaserpositions = [ answer.first.sample.get(f'x_{i}') for i in range(3*n) ]
   # Print the solution (chaser's positions), 
   # in array form [x_0,x_1,\dots,x_{n-1},y_0,y_1,\dots,y_{n-1},z_0,z_1,\dots,z_{n-1}]
   print("\nChaser positions:\n",chaserpositions)

   # Validate solution
   smaxdx = smax**2
   smindx = smin**2
   Result = True
   for i in range(n):
      delta = (chaserpositions[i]-pos[i])**2+(chaserpositions[i+n]-pos[i+n])**2+(chaserpositions[i+2*n]-pos[i+2*n])**2
      if delta >= smindx and delta <= smaxdx:
         continue
      else:
        print("*** Invalid solution!\n")
        Result = False
        break
  
   # print the directory of information
   # print("\nInfo:", answer.info)
   # print run time
   # print("\nn: ", n, " Run time: ", answer.info['run_time'], " QPU Access Time: ", answer.info['qpu_access_time'])
   # print sample with lowest energy
   # print("\nSample:", answer.first.sample)
   # print([math.sqrt( (x[i]-pos[i])**2+(x[i+n]-pos[i+n])**2+(x[i+2*n]-pos[i+2*n])**2) for i in range(n)])
   # Convert chaser's list of positions from dictionary to array
   chaserpositions = [ answer.first.sample.get(f'x_{i}') for i in range(n) ]
   # Print the solution (chaser's positions), 
   # in array form [x_0, x_1,\ldost,n_{n-1}
   print(chaserpositions)
   print("Tracking score:", answer.first.energy)

   return(Result, answer.info['run_time'], answer.info['qpu_access_time'])

################
### Main program

maxpos = 10 # Size of cubic 3D space

numaverages = 1 # Number of averages
numsamples = 0 # Number of valid samples used to calculate an average
maxnumsamples = 10 # Maximum number of valid samples used to calculate an average

RTaverages = np.zeros(numaverages, dtype=float) # Hybrid Comp Run Time
QPUaverages = np.zeros(numaverages, dtype=float) # QPU Access Time

print("---")
for i in range(numaverages):
    T1 = 0 # Hybrid Comp Run Time
    T2 = 0 # QPU Access Time
    numberOfPoints = (i + 1) * 50 # number of points used to caclulate this comp time avaerage
    for j in range(maxnumsamples):
         # 1st param is number of points
         # 2nd param is index of maximum position
         # 3rd param number of obstacles
         Result, RT, QPU = TargetTracking(numberOfPoints, maxpos, 10*maxpos)
         if Result:
            numsamples = numsamples + 1    
            T1 = T1 + RT
            T2 = T2 + QPU
    if numsamples > 0: # Calculate averages!      
       RTaverages[i] = T1 / numsamples
       QPUaverages[i] = T2 / numsamples
       print("---\nNumber of points: ", numberOfPoints)
       print("Number of samples: ", numsamples)
       print("RTaverages[i]: ", RTaverages[i])
       print("RQPUAverages[i]: ", QPUaverages[i])
    else: # No sample available!
       print("---\nNumber of points: ", numberOfPoints, " - All solutions are invalid!\n")
    print("---")

# Print the arrays
print("RT Averages:", RTaverages)
print("QPU Averages:", QPUaverages)

exit()

