
#%%
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize


# Load matrices
dist_matrix = pd.read_csv('EMO/ciudad_ixtepec_distance_matrix.csv', index_col=0).values
time_matrix = pd.read_csv('EMO/ciudad_ixtepec_time_matrix.csv', index_col=0).values

n_points = dist_matrix.shape[0]
start_node = 0  # Central node

class VRPProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=n_points-1,  # Exclude start node from permutation
            n_obj=2,
            n_constr=0,
            xl=0,
            xu=n_points-2,
            type_var=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a permutation of nodes excluding the start node
        route = [start_node] + [i+1 for i in x] + [start_node]
        total_dist = 0
        total_time = 0
        for i in range(len(route)-1):
            total_dist += dist_matrix[route[i], route[i+1]]
            total_time += time_matrix[route[i], route[i+1]]
        out["F"] = [total_dist, total_time]

# Setup NSGA-II
algorithm = NSGA2(
    pop_size=50,
    sampling=PermutationRandomSampling(),
    crossover=OrderCrossover(),
    mutation=InversionMutation(),
    eliminate_duplicates=True
)

problem = VRPProblem()

res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    seed=1,
    verbose=True
)

# Print Pareto front
for i, sol in enumerate(res.F):
    print(f"Solution {i}: Total Distance = {sol[0]:.2f}, Total Time = {sol[1]:.2f}")