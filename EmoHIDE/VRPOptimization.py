
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
points_df = pd.read_csv('EMO/ciudad_ixtepec_points.csv')

n_points = dist_matrix.shape[0]
start_node = n_points-1  # Central node

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
        route = [start_node] + list(x) + [start_node]
        total_dist = 0
        total_time = 0
        for i in range(len(route)-2):
            total_dist += dist_matrix[route[i], route[i+1]]
            total_time += time_matrix[route[i], route[i+1]]
        out["F"] = [total_dist, total_time]

# Setup NSGA-II
algorithm = NSGA2(
    pop_size=100,
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
#%%
import osmnx as ox
import matplotlib.pyplot as plt
# Function to visualize the route
def plot_route(solution_idx, X, F):
    # Get the specific solution
    x = X[solution_idx]

    # Construct the route including start/end point
    route = [start_node] + list(x) + [start_node]

    # Load road network
    place = "Ciudad Ixtepec, Oaxaca, Mexico"
    G = ox.graph_from_place(place, network_type='drive')

    # Get lat/lon coordinates
    lats = points_df['lat'].tolist()
    lons = points_df['lon'].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # Plot road network
    ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0,
                  edge_color='gray', edge_linewidth=0.5)
    route_colors = [
        '#e6194B',  # Red
        '#3cb44b',  # Green
        '#4363d8',  # Blue
        '#f58231',  # Orange
        '#911eb4',  # Purple
        '#42d4f4',  # Cyan
        '#f032e6',  # Magenta
        '#bfef45',  # Lime
        '#fabed4',  # Pink
        '#469990',  # Teal
        '#dcbeff',  # Lavender
        '#9A6324',  # Brown
        '#800000',  # Maroon
        '#aaffc3',  # Mint
        '#000075',  # Navy
    ]
    for i in range(len(route) - 2):
        route = ox.shortest_path(G,points_df['node'][i], points_df['node'][i +1])
        ox.plot_graph_route(G, route, ax=ax,show=False, close=False, node_size=0,route_linewidth=1,route_color=route_colors[i])


    for i, row in points_df.iterrows():
        ax.text(row['lon'], row['lat'], str(i), fontsize=8, color='black', zorder=4)

    # # Plot points with sequence numbers
    # for i, node_idx in enumerate(route):
    #     marker_size = 100 if i == 0 or i == len(route) - 1 else 80
    #     marker_color = 'green' if i == 0 or i == len(route) - 1 else 'red'
    #     ax.scatter(lons[node_idx], lats[node_idx], s=marker_size, c=marker_color, zorder=5)
    #     ax.text(lons[node_idx], lats[node_idx], f"{i}:{node_idx + 1}", fontsize=10,
    #             ha='center', va='bottom', color='black', fontweight='bold')

    # Add title with metrics
    plt.title(
        f"TSP Route #{solution_idx + 1}: Distance = {F[solution_idx][0]:.1f}m, Time = {F[solution_idx][1]:.1f}s")
    plt.tight_layout()

    return fig, ax


# Print Pareto front and visualize top solutions
print("Pareto Front Solutions:")
for i, sol in enumerate(res.F):
    print(f"Solution {i + 1}: Total Distance = {sol[0]:.2f}m, Total Time = {sol[1]:.2f}s")

# Plot the first 3 solutions (or fewer if there are less than 3)
num_to_plot = min(3, len(res.F))
for i in range(num_to_plot):
    fig, ax = plot_route(i, res.X, res.F)
    plt.savefig(f"tsp_route_solution_{i + 1}.png", dpi=300)
    plt.show()