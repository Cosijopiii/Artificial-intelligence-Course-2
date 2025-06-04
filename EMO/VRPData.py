#%%
import osmnx as ox
import networkx as nx
import random
import pandas as pd

# Load the road network of Ciudad Ixtepec
place = "Ciudad Ixtepec, Oaxaca, Mexico"
G = ox.graph_from_place(place, network_type='drive')
#G = ox.simplify_graph(G)

# Get 10 random delivery points
nodes = list(G.nodes)
delivery_nodes = random.sample(nodes, 10)

# Save lat/lon of delivery points
delivery_points = []
for node in delivery_nodes:
    point = G.nodes[node]
    delivery_points.append({'node': node, 'lat': point['y'], 'lon': point['x']})

# Add travel time to edges (assume 40 km/h avg speed)

for u, v, data in G.edges(data=True):
    speed_kph = 35
    speed_mps = speed_kph * 1000 / 3600
    bias=1
    data['travel_time'] = data['length'] / speed_mps*bias  # seconds

def count_turns_in_path(G, path):
    """Count the number of significant turns in a path."""
    if len(path) < 3:
        return 0

    turns = 0
    for i in range(len(path)-2):
        # Get the bearings between consecutive segments
        u, v, w = path[i], path[i+1], path[i+2]

        # Get the bearings/angles - pass coordinates as separate arguments
        bearing1 = ox.bearing.calculate_bearing(
            G.nodes[u]['y'], G.nodes[u]['x'],  # lat1, lon1
            G.nodes[v]['y'], G.nodes[v]['x']  # lat2, lon2
        )

        bearing2 = ox.bearing.calculate_bearing(
            G.nodes[v]['y'], G.nodes[v]['x'],  # lat1, lon1
            G.nodes[w]['y'], G.nodes[w]['x']  # lat2, lon2
        )
        # Calculate turn angle
        turn_angle = abs((bearing2 - bearing1 + 180) % 360 - 180)

        # Count significant turns (more than 30 degrees)
        if turn_angle > 30:
            turns += 1

    return turns
# Build distance and time matrices
distance_matrix = []
time_matrix = []
turns_matrix = []
for i in range(len(delivery_points)):
    row_dist = []
    row_time = []
    row_turns = []
    for j in range(len(delivery_points)):
        orig = delivery_points[i]['node']
        dest = delivery_points[j]['node']
        if orig == dest:
            row_dist.append(0)
            row_time.append(0)
            row_turns.append(0)
        else:
            try:
                dist = nx.shortest_path_length(G, orig, dest, weight='length')
                time = nx.shortest_path_length(G, orig, dest, weight='travel_time')
                temp_path= nx.shortest_path(G, orig, dest, weight='length')

            except:
                dist = float('inf')
                time = float('inf')
              #  turns = float('inf')
            turns = count_turns_in_path(G, temp_path)
            row_dist.append(dist)
            row_time.append(time)
            row_turns.append(turns)

    distance_matrix.append(row_dist)
    time_matrix.append(row_time)
    turns_matrix.append(row_turns)


emission_rate = 0.192  # grams CO₂ per meter

co2_matrix = []
for i in range(len(distance_matrix)):
    row_co2 = []
    for j in range(len(distance_matrix)):
        co2 = distance_matrix[i][j] * emission_rate
        row_co2.append(co2)
    co2_matrix.append(row_co2)

# Export CO₂ matrix
pd.DataFrame(co2_matrix).to_csv("EMO/ciudad_ixtepec_co2_matrix.csv", index=False)
pd.DataFrame(delivery_points).to_csv("EMO/ciudad_ixtepec_points.csv", index=False)
pd.DataFrame(distance_matrix).to_csv("EMO/ciudad_ixtepec_distance_matrix.csv", index=False)
pd.DataFrame(time_matrix).to_csv("EMO/ciudad_ixtepec_time_matrix.csv", index=False)
pd.DataFrame(turns_matrix).to_csv("EMO/ciudad_ixtepec_turns_matrix.csv", index=False)

#%%
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

# Load road network
place = "Ciudad Ixtepec, Oaxaca, Mexico"
G = ox.graph_from_place(place, network_type='drive')

# Load delivery points
df_points = pd.read_csv("EMO/ciudad_ixtepec_points.csv")
lats = df_points['lat'].tolist()
lons = df_points['lon'].tolist()

# Set up a high-resolution figure
fig, ax = plt.subplots(figsize=(12, 12), dpi=300)  # You can increase figsize or dpi as needed

# Plot the road network
ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0, edge_color='gray', edge_linewidth=0.6)

# Plot delivery points
ax.scatter(lons, lats, c='red', s=30, marker='o', label='Delivery Points', zorder=3)

# Annotate each point with its ID
for i, row in df_points.iterrows():
    ax.text(row['lon'], row['lat'], str(i), fontsize=8, color='black', zorder=4)

# Final touches
plt.legend()
plt.title("Delivery Points in Ciudad Ixtepec", fontsize=14)
plt.tight_layout()

# Optional: Save to file
plt.savefig("ciudad_ixtepec_map_highres.png", dpi=300)
plt.show()

#%%
import numpy as np
import pandas as pd
import itertools

# Load matrices
dist_matrix = pd.read_csv('EMO/ciudad_ixtepec_distance_matrix.csv', index_col=0).values
time_matrix = pd.read_csv('EMO/ciudad_ixtepec_time_matrix.csv', index_col=0).values
co2_matrix = pd.read_csv('EMO/ciudad_ixtepec_co2_matrix.csv', index_col=0).values
turns_matrix= pd.read_csv('EMO/ciudad_ixtepec_turns_matrix.csv', index_col=0).values
n_points = dist_matrix.shape[0]
start_node = n_points - 1  # Central node

# All permutations of delivery nodes (excluding start node)
nodes = list(range(n_points - 1))
all_perms = list(itertools.permutations(nodes))

results = np.zeros((len(all_perms), 3))  # columns: distance, time, co2
TURN_PENALTY= 10000  # Penalty for turns in seconds
for idx, perm in enumerate(all_perms):
    route = [start_node] + list(perm) + [start_node]
    total_dist = 0
    total_time = 0
    total_co2 = 0
    total_turns = 0

    for i in range(len(route) - 2):
        total_dist += dist_matrix[route[i], route[i + 1]]
        total_time += time_matrix[route[i], route[i + 1]]
        total_co2 += co2_matrix[route[i], route[i + 1]]
        total_turns += turns_matrix[route[i], route[i + 1]]
    penalized_time = total_time + (TURN_PENALTY * total_turns)

    results[idx] = [total_dist, total_turns, total_co2]

# Save results to CSV
np.savetxt('EMO/all_permutations_objectives.csv', results, delimiter=',', header='distance,time,co2', comments='')


#%%
import matplotlib.pyplot as plt
from itertools import combinations

# Plot all pairwise combinations of the three columns: distance, time, co2
col_names = ['Total Distance', 'Total Turns', 'Total CO2']
plt.figure(figsize=(18, 5))

for idx, (i, j) in enumerate(combinations(range(3), 2)):
    plt.subplot(1, 3, idx + 1)
    plt.scatter(results[:, i], results[:, j], alpha=0.5, s=10)
    plt.xlabel(col_names[i])
    plt.ylabel(col_names[j])
    plt.title(f'{col_names[i]} vs {col_names[j]}')
    plt.grid(True)

plt.tight_layout()
plt.show()
#%%

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
co2_matrix = pd.read_csv('EMO/ciudad_ixtepec_co2_matrix.csv', index_col=0).values

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
        total_co2 = 0
        for i in range(len(route) - 2):
            total_dist += dist_matrix[route[i], route[i + 1]]
            total_time += time_matrix[route[i], route[i + 1]]
            total_co2 += co2_matrix[route[i], route[i + 1]]
        out["F"] = [total_dist, total_co2]
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
    ('n_gen', 1000),
    seed=1,
    verbose=True,
    save_history=True
)

# Print Pareto front
for i, sol in enumerate(res.F):
    print(f"Solution {i}: Total Distance = {sol[0]:.2f}, Total Time = {sol[1]:.2f}")
#%%
import osmnx as ox
import matplotlib.pyplot as plt
# Function to visualize the route{}}
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
        ox.plot_graph_route(G, route, ax=ax,show=False, close=False, node_size=0,route_linewidth=1)


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

