# Copyright 2024 Michel Barbeau
############################
# 1D Target Tracking Example
# Author: Michel Barbeau, Carleton University
# Version: 2024/03/09
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
	python3 3D-target-tracking.py
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
   smax = 2  # chaser-target maximum distance
   for i in range(n):
      cqm.add_constraint( ( (x[i]-pos[i])**2+(x[i+n]-pos[i+n])**2+(x[i+2*n]-pos[i+2*n])**2 ) <= smax**2, label=f'xmax_{i}')
      cqm.add_constraint( ( (x[i]-pos[i])**2+(x[i+n]-pos[i+n])**2+(x[i+2*n]-pos[i+2*n])**2 ) >=smin**2, label=f'xmin_{i}')

   # Initialize the CQM solver
   # hybrid solver
   sampler = LeapHybridCQMSampler()
   # Submit for solution
   answer = sampler.sample_cqm(cqm)

   # print the directory of information
   # print("\nInfo:", answer.info)
   # print run time
   # print("\nn: ", n, " Run time: ", answer.info['run_time'], " QPU Access Time: ", answer.info['qpu_access_time'])
   # print sample with lowest energy
   # print("\nSample:", answer.first.sample)
   # print([math.sqrt( (x[i]-pos[i])**2+(x[i+n]-pos[i+n])**2+(x[i+2*n]-pos[i+2*n])**2) for i in range(n)])
   # print("Tracking score:", answer.first.energy)
   return(answer.info['run_time'], answer.info['qpu_access_time'])

################
### Main program

maxpos = 10
#n = 1000
#TargetTracking(n, maxpos)

numaverages = 10
numsamples = 10
RTaverages = np.zeros(numaverages, dtype=float)
QPUaverages = np.zeros(numaverages, dtype=float)
for i in range(numaverages):
    T1 = 0
    T2 = 0
    print("i: ", i, "\n")
    for j in range(numsamples):
         RT, QPU = TargetTracking((i + 1) * 100, maxpos)
         print(RT, QPU, " ")
         T1 = T1 + RT
         T2 = T2 + QPU
    print("\n")
    RTaverages[i] = T1 / numsamples
    QPUaverages[i] = T2 / numsamples
    print("RTAverage: ", RTaverages[i], "RQPUAverage: ", QPUaverages[i],"\n")

print("RT Averages:", RTaverages, "\n")
print("QPU Averages:", QPUaverages, "\n")

exit()

