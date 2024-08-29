# Copyright 2024 Michel Barbeau
############################
# 1D Target Tracking Example
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
	python3 quantum-1D-target-tracking.py
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
X = [Integer(f'x_{i}', lower_bound=0, upper_bound=maxpos) for i in range(n)] # positions and
V = [Integer(f'v_{i}', lower_bound=-maxpos, upper_bound=maxpos) for i in range(n)] # speed

# generate target's random positions
rng = np.random.default_rng()
pos = rng.integers(low=0, high=maxpos, size=n)
print("\nTarget positions: ", pos)

# initialize the CQM object
cqm = ConstrainedQuadraticModel()

# add the variables to the model
for i in range(len(X)):
   cqm.add_variable('INTEGER', f'x_{i}', lower_bound=0, upper_bound=maxpos)
   cqm.add_variable('INTEGER', f'v_{i}', lower_bound=-maxpos, upper_bound=maxpos)

# objective is to minimize the accelaration
obj = quicksum([((V[i+1]-V[i])**2) for i in range(n-1)])
cqm.set_objective(obj)

# add constraints
maxdx = 2
mindx = 2
for i in range(len(X)-1):
    cqm.add_constraint(X[i+1]-X[i]-V[i]==0, label=f'p_{i}')
for i in range(len(X)):
    cqm.add_constraint((X[i]-pos[i])>=-maxdx, label=f'u_{i}')
    cqm.add_constraint((X[i]-pos[i])<=maxdx, label=f'l_{i}')
    cqm.add_constraint((X[i]-pos[i])**2>=mindx**2, label=f'min_{i}')

# Initialize the CQM solver
sampler = LeapHybridCQMSampler()

# Submit for solution
answer = sampler.sample_cqm(cqm)
# Print the solution, in dictionary form
print("\nInfo:", answer.info)
print("\nSample:", answer.first.sample)
print("Tracking score:", answer.first.energy)

# Convert chaser's list of positions from dictionary to array
chaserpositions = [ answer.first.sample.get(f'x_{i}') for i in range(n) ]
# Print the solution (chaser's positions), 
# in array form [x_0, x_1,\ldost,n_{n-1}
print(chaserpositions)

# Validate solution
smaxdx = maxdx**2
smindx = mindx**2
for i in range(n):
    delta = (chaserpositions[i]-pos[i])**2
    if delta >= smindx and delta <= smaxdx:
       continue
    else:
        print("*** Invalid solution!\n")
        exit()
print("*** Valid solution!\n")
exit()
