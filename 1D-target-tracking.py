# Copyright 2024 Michel Barbeau
############################
# 1D Target Tracking Example
# Author: Michel Barbeau, Carleton University
# Version: 2024/02/07
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
# create the variables
vars = [Integer(f'x_{i}') for i in range(n)]
print("\nNumber of instants: ", n)

# generate target's random positions
rng = np.random.default_rng()
maxpos=10
pos = rng.integers(low=0, high=maxpos, size=n)
print("\nTarget positions: ", pos)

# initialize the CQM object
cqm = ConstrainedQuadraticModel()

# maximum x-coordinate
maxxcoord = 2
print("\nMaximum number of coordinates: ", maxxcoord)
# add constraint to variables 
for i in range(len(vars)):
   # cqm.add_variable('INTEGER', f'x_{i}', lower_bound=0.0, upper_bound=maxpos)
   cqm.add_variable('INTEGER', f'x_{i}' )

# objective: minimize accelaration, in fact speed for now
obj = -quicksum([((vars[i+1]-vars[i])**2) for i in range(n-1)])
cqm.set_objective(obj)

# constraint: equal number of employees per shift
maxdx = 4
for i in range(len(vars)):
    cqm.add_constraint((vars[i]-pos[i])>=-maxdx, label=f'u_{i}')
    cqm.add_constraint((vars[i]-pos[i])<=maxdx, label=f'l_{i}')

# Initialize the CQM solver
sampler = LeapHybridCQMSampler()

# Submit for solution
answer = sampler.sample_cqm(cqm)
print("\nInfo:", answer.info)
print("\nSample:", answer.first.sample)
print("Tracking score:", answer.first.energy)

# plot results
dx = np.zeros(10)
for key, val in answer.first.sample.items():
   v = key.split("_")
   print("\n", key, val, v)
   dx[int(v[1])] = val-pos[int(v[1])]

# plot result
fig, ax = plt.subplots()
ax.bar(np.arange(n), dx)
ax.set(xlabel='Time', ylabel='Distance to target')
ax.set_title('1D Target Tracking')
fig.suptitle('Max dx is '+str(maxdx))
plt.grid()
plt.show()

