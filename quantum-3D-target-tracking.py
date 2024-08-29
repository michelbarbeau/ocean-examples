# Copyright 2024 Michel Barbeau
############################
# 3D Target Tracking Example
# Author: Michel Barbeau, Carleton University
# Version: 2024/08/28
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
Usage: 
	python3 quantum-3D-target-tracking.py
"""

from dimod import ConstrainedQuadraticModel, Integer, quicksum
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

###############################
### 3D Target Tracking function
def TargetTracking(n, maxpos):
   ### parameters
   # n : number of instants
   # maxpos : index of maximum position

   # print("\nNumber of instants: ", n)
   # create the variables, they represent
   # [x_0,x_1,\dots,x_{n-1},y_0,y_1,\dots,y_{n-1},z_0,z_1,\dots,z_{n-1}]
   x = [Integer(f'x_{i}', lower_bound=0, upper_bound=maxpos) for i in range(3*n)]

   # generate target's random positions
   rng = np.random.default_rng()
   pos = np.zeros(3 * n, dtype=int)
   pos[0:3] = rng.integers(low=0, high=maxpos, size=3)
   delta = rng.integers(low=-1, high=2, size=[3, n])  # x,y,z-displacements
   # print("\nDeltas:\n", delta)
   for i in range(n-1):
      pos[i + 1] = np.min([np.max([pos[i] + delta[0, i + 1], 0]), maxpos])
      pos[i + 1 + n] = np.min([np.max([pos[i + n] + delta[1, i + 1], 0]), maxpos])
      pos[i + 1 + 2 * n] = np.min([np.max([pos[i + 2 * n] + delta[2, i + 1], 5]), maxpos])
   #print("\nTarget positions:\n", pos)

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

   # Initialize the CQM solver
   # hybrid solver
   sampler = LeapHybridCQMSampler()
   # Submit for solution
   answer = sampler.sample_cqm(cqm)

   # Convert chaser's list of positions from dictionary to array
   chaserpositions = [ answer.first.sample.get(f'x_{i}') for i in range(3*n) ]
   # Print the solution (chaser's positions), 
   # in array form [x_0,x_1,\dots,x_{n-1},y_0,y_1,\dots,y_{n-1},z_0,z_1,\dots,z_{n-1}]
   print(chaserpositions)

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
   print("\nSample:", answer.first.sample)
   # print([math.sqrt( (x[i]-pos[i])**2+(x[i+n]-pos[i+n])**2+(x[i+2*n]-pos[i+2*n])**2) for i in range(n)])
   print("Tracking score:", answer.first.energy)

   return(Result, answer.info['run_time'], answer.info['qpu_access_time'])

################
### Main program

maxpos = 10 # Size of cubic 3D space

numaverages = 1 # Number of averages
numsamples = 0 # Number of valid samples used to calculate an average
maxnumsamples = 2 # Maximum number of valid samples used to calculate an average

RTaverages = np.zeros(numaverages, dtype=float) # Hybrid Comp Run Time
QPUaverages = np.zeros(numaverages, dtype=float) # QPU Access Time

print("---")
for i in range(numaverages):
    T1 = 0 # Hybrid Comp Run Time
    T2 = 0 # QPU Access Time
    numberOfPoints = (i + 1) * 10 # number of points used to caclulate this comp time avaerage
    for j in range(maxnumsamples):
         # 1st param is number of points
         # 2nd param is index of maximum position
         # 3rd param number of obstacles
         Result, RT, QPU = TargetTracking(numberOfPoints, maxpos)
         if Result:
            numsamples = numsamples + 1    
            T1 = T1 + RT
            T2 = T2 + QPU
    if numsamples > 0: # Calculate averages!      
       RTaverages[i] = T1 / numsamples
       QPUaverages[i] = T2 / numsamples
       print("Number of points: ", numberOfPoints)
       print("Number of samples: ", numsamples)
       print("RTaverages[i]: ", RTaverages[i])
       print("RQPUAverages[i]: ", QPUaverages[i])
    else: # No sample available!
       print("Number of points: ", numberOfPoints, " - All solutions are invalid!\n")
    print("---")

# Print the arrays
print("RT Averages:", RTaverages)
print("QPU Averages:", QPUaverages)

exit()

