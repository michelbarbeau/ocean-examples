# Copyright 2024 Michel Barbeau
############################
# 1D Target Tracking Example
# Author: Michel Barbeau, Carleton University
# Version: 2024/02/15
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
	python3 1D-target-tracking.py
"""

from dimod import ConstrainedQuadraticModel, Integer, quicksum
from dwave.system import LeapHybridCQMSampler
import numpy as np
import matplotlib.pyplot as plt

# number of instants
n = 10
maxpos=10
print("\nNumber of instants: ", n)
# create the variables, they represent
X = [Integer(f'x_{i}', lower_bound=0, upper_bound=maxpos) for i in range(n)] # x-axis positions,
Y = [Integer(f'y_{i}', lower_bound=0, upper_bound=maxpos) for i in range(n)] # y-axis positions,
Z = [Integer(f'z_{i}', lower_bound=0, upper_bound=maxpos) for i in range(n)] # z-axis positions,
Vx = [Integer(f'vx_{i}', lower_bound=-maxpos, upper_bound=maxpos) for i in range(n)] # x-axis speed,
Vy = [Integer(f'vy_{i}', lower_bound=-maxpos, upper_bound=maxpos) for i in range(n)] # y-axis speed, and
Vz = [Integer(f'vz_{i}', lower_bound=-maxpos, upper_bound=maxpos) for i in range(n)] # z-axis speed.

# generate target's random positions
rng = np.random.default_rng()
pos =  np.zeros((3,n), dtype=int)
pos[:,0] = rng.integers(low=0, high=maxpos,size=3)
delta = rng.integers(low=-1, high=2, size=[3,n]) # x,y,z-displacements
print("\nDeltas:\n", delta)
for i in range(n-1):
   pos[0,i+1] = np.min( [np.max( [pos[0,i] + delta[0,i+1], 0]), maxpos] )
   pos[1,i+1] = np.min( [np.max( [pos[1,i] + delta[1,i+1], 0]), maxpos] )
   pos[2,i+1] = np.min( [np.max( [pos[2,i] + delta[2,i+1], 5]), maxpos] )
print("\nTarget positions:\n", pos)

# initialize the CQM object
cqm = ConstrainedQuadraticModel()

# add the variables to the model
for i in range(len(X)):
   cqm.add_variable('INTEGER', f'x_{i}', lower_bound=0, upper_bound=maxpos)
   cqm.add_variable('INTEGER', f'vx_{i}', lower_bound=-maxpos, upper_bound=maxpos)
   cqm.add_variable('INTEGER', f'y_{i}', lower_bound=0, upper_bound=maxpos)
   cqm.add_variable('INTEGER', f'vy_{i}', lower_bound=-maxpos, upper_bound=maxpos)
   cqm.add_variable('INTEGER', f'z_{i}', lower_bound=0, upper_bound=maxpos)
   cqm.add_variable('INTEGER', f'vz_{i}', lower_bound=-maxpos, upper_bound=maxpos)

# initial acceleration
Vx[0]=0
Vx[1]=0
Vx[2]=0
# objective is to minimize the accelaration
obj = quicksum([ ((Vx[i+1]-Vx[i])**2+(Vy[i+1]-Vy[i])**2+(Vz[i+1]-Vz[i])**2) for i in range(n-1)])
cqm.set_objective(obj)

# add constraints
maxdx = 2
mindx = 1
for i in range(len(X)-1):
    cqm.add_constraint(X[i+1]-X[i]-Vx[i]==0, label=f'px_{i}')
    cqm.add_constraint(Y[i+1]-Y[i]-Vy[i]==0, label=f'py_{i}')
    cqm.add_constraint(Z[i+1]-Z[i]-Vz[i]==0, label=f'pz_{i}')
for i in range(len(X)):
    cqm.add_constraint( ( (X[i]-pos[0,i])**2+(Y[i]-pos[1,i])**2+(Z[i]-pos[2,i])**2 ) <= maxdx**2, label=f'xmax_{i}')
    cqm.add_constraint( ( (X[i]-pos[0,i])**2+(Y[i]-pos[1,i])**2+(Z[i]-pos[2,i])**2 ) >=mindx**2, label=f'xmin_{i}')

# Initialize the CQM solver
sampler = LeapHybridCQMSampler()

# Submit for solution
answer = sampler.sample_cqm(cqm)
print("\nInfo:", answer.info)
print("\nSample:", answer.first.sample)
print("Tracking score:", answer.first.energy)

exit()
