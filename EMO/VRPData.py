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
delivery_nodes = random.sample(nodes, 20)

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
        if i == j:
            row_dist.append(0)
            row_time.append(0)
            row_turns.append(0)
        else:
            try:
                dist = nx.shortest_path_length(G, orig, dest, weight='length')
                time = nx.shortest_path_length(G, orig, dest, weight='travel_time')
                temp_path= nx.shortest_path(G, orig, dest, weight='length')
                turns = count_turns_in_path(G, temp_path)
            except:
                dist = float('inf')
                time = float('inf')
                turns = float('inf')
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
dist_matrix = pd.read_csv('EMO/ciudad_ixtepec_distance_matrix.csv' ).values
time_matrix = pd.read_csv('EMO/ciudad_ixtepec_time_matrix.csv').values
co2_matrix = pd.read_csv('EMO/ciudad_ixtepec_co2_matrix.csv').values
turns_matrix= pd.read_csv('EMO/ciudad_ixtepec_turns_matrix.csv').values
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
dist_matrix = pd.read_csv('EMO/ciudad_ixtepec_distance_matrix.csv').values
time_matrix = pd.read_csv('EMO/ciudad_ixtepec_time_matrix.csv').values
points_df = pd.read_csv('EMO/ciudad_ixtepec_points.csv')
co2_matrix = pd.read_csv('EMO/ciudad_ixtepec_co2_matrix.csv').values
turns_matrix=pd.read_csv('EMO/ciudad_ixtepec_turns_matrix.csv').values
n_points = dist_matrix.shape[0]
start_node = n_points-1

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
        total_turns = 0
        for K in range(len(route) - 1):
            total_dist += dist_matrix[route[K], route[K + 1]]
            total_time += time_matrix[route[K], route[K + 1]]
            total_co2 += co2_matrix[route[K], route[K + 1]]
            total_turns += turns_matrix[route[K], route[K + 1]]

        out["F"] = [total_dist, total_turns]
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
    verbose=True,
    save_history=True
)

# Print Pareto front
for i, sol in enumerate(res.F):
    print(f"Solution {i}: Total Distance = {sol[0]}, Total Turns = {sol[1]}")

# Scatter plot of Pareto front (Total Distance vs Total Turns)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(res.F[:, 0], res.F[:, 1], c='blue', alpha=0.6)
plt.xlabel("Total Distance")
plt.ylabel("Total Turns")
plt.title("Pareto Front: Total Distance vs Total Turns")
plt.grid(True)
plt.tight_layout()
plt.savefig("pareto_front_scatter.png", dpi=300)
plt.show()

#%%
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio  # Use v2 to avoid deprecation warning
import os
from tqdm import tqdm  # For progress bar


def create_route_animation(solution_idx, X, F, output_gif="route_animation.gif", fps=1):
    """Create a GIF animation showing each segment of the route in sequence"""
    # Get the specific solution
    x = X[solution_idx]

    # Construct the route including start/end point
    route = [start_node] + list(x) + [start_node]

    print(f"Processing route with {len(route)} nodes: {route}")

    # Load road network
    place = "Ciudad Ixtepec, Oaxaca, Mexico"
    G = ox.graph_from_place(place, network_type='drive')

    # Create directory for frames
    frames_dir = "tmp_frames"
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    # Colors for segments
    colors = plt.cm.rainbow(np.linspace(0, 1, len(route) - 1))

    # Track completed segments
    completed_paths = []

    # Generate frames for each segment
    for i in tqdm(range(len(route) - 1)):
        # Create new figure for each frame
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot base road network
        ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0,
                      edge_color='lightgray', edge_linewidth=0.5)

        # Plot all points with correct parameter (color instead of c)
        for j, row in points_df.iterrows():
            if j in [route[i], route[i + 1]]:
                # Highlight current nodes
                size = 100
                color = 'green' if j == route[i] else 'red'
            else:
                size = 30
                color = 'blue'

            ax.scatter(row['lon'], row['lat'], color=color, s=size, zorder=3)
            ax.text(row['lon'], row['lat'], str(j), fontsize=12,
                    fontweight='bold', color='black', zorder=4)

        # Plot previously completed segments
        for idx, path in enumerate(completed_paths):
            try:
                ox.plot_graph_route(G, path, ax=ax, route_color=colors[idx].tolist(),
                                    route_linewidth=1.5, route_alpha=0.6,
                                    show=False, close=False)
            except Exception as e:
                print(f"Error plotting previous path: {str(e)}")

        # Get current segment path
        try:
            # Make sure points_df has the node column with OSM node IDs
            orig_node = points_df.loc[route[i], 'node']
            dest_node = points_df.loc[route[i + 1], 'node']

            path = ox.shortest_path(G, orig_node, dest_node, weight='length')

            # Plot current segment with bright color
            ox.plot_graph_route(G, path, ax=ax, route_color=colors[i].tolist(),
                                route_linewidth=3, route_alpha=1.0,
                                show=False, close=False)

            # Add current segment to completed paths
            completed_paths.append(path)

            # Add segment info as text on the plot
            mid_y = (points_df.loc[route[i], 'lat'] + points_df.loc[route[i + 1], 'lat']) / 2
            mid_x = (points_df.loc[route[i], 'lon'] + points_df.loc[route[i + 1], 'lon']) / 2
            ax.text(mid_x, mid_y, f"Segment {i + 1}: {route[i]}→{route[i + 1]}",
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7), zorder=5)

        except Exception as e:
            print(f"Error plotting path from {route[i]} to {route[i + 1]}: {str(e)}")

        # Add title
        plt.title(f"Route Animation - Segment {i + 1}/{len(route) - 1}\n" +
                  f"Total: Distance = {F[solution_idx][0]:.1f}m, Turns = {F[solution_idx][1]:.1f}")

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=100)
        frame_paths.append(frame_path)
        plt.close(fig)

    # Create GIF from frames
    print("Creating GIF animation...")
    with imageio.get_writer(output_gif, mode='I', duration=1 / fps) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    print(f"Animation saved to {output_gif}")

    # Also create a final image showing the complete route
    create_final_route_image(route, G, F[solution_idx], "complete_route.png")

    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(frames_dir)


def create_final_route_image(route, G, metrics, output_file="complete_route.png"):
    """Create a static image of the complete route"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot base road network
    ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0,
                  edge_color='lightgray', edge_linewidth=0.5)

    # Colors for segments
    colors = plt.cm.rainbow(np.linspace(0, 1, len(route) - 1))

    # Plot all segments
    for i in range(len(route) - 1):
        try:
            orig_node = points_df.loc[route[i], 'node']
            dest_node = points_df.loc[route[i + 1], 'node']

            path = ox.shortest_path(G, orig_node, dest_node, weight='length')
            ox.plot_graph_route(G, path, ax=ax, route_color=colors[i].tolist(),
                                route_linewidth=2, route_alpha=0.8,
                                show=False, close=False)
        except Exception as e:
            print(f"Error plotting path from {route[i]} to {route[i + 1]}: {str(e)}")

    # Plot all points
    for j, row in points_df.iterrows():
        size = 80 if j == route[0] else 50
        color = 'green' if j == route[0] else 'blue'

        ax.scatter(row['lon'], row['lat'], color=color, s=size, zorder=3)
        ax.text(row['lon'], row['lat'], str(j), fontsize=12,
                fontweight='bold', color='black', zorder=4)

    plt.title(f"Complete Route\nDistance = {metrics[0]:.1f}m, Turns = {metrics[1]:.1f}")
    plt.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Complete route image saved to {output_file}")
# Example usage:
# create_route_animation(0, res.X, res.F, output_gif="route_animation.gif", fps=0.5)

# Create animation for the first solution in the Pareto front
create_route_animation(0, res.X, res.F, output_gif="best_route_animation.gif", fps=10)




#%%

import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from moviepy import ImageSequenceClip


def create_route_video(solution_idx, X, F, output_video="route_video.mp4", fps=2):
    """Create a video showing each segment of the route in sequence"""
    # Get the specific solution
    x = X[solution_idx]

    # Construct the route including start/end point
    route = [start_node] + list(x) + [start_node]

    print(f"Processing route with {len(route)} nodes: {route}")

    # Load road network
    place = "Ciudad Ixtepec, Oaxaca, Mexico"
    G = ox.graph_from_place(place, network_type='drive')

    # Create directory for frames
    frames_dir = "tmp_frames"
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    # Colors for segments
    colors = plt.cm.rainbow(np.linspace(0, 1, len(route) - 1))

    # Track completed segments
    completed_paths = []

    # Generate frames for each segment
    for i in tqdm(range(len(route) - 1)):
        # Create new figure for each frame
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot base road network
        ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0,
                      edge_color='lightgray', edge_linewidth=0.5)

        # Plot all points
        for j, row in points_df.iterrows():
            if j in [route[i], route[i + 1]]:
                # Highlight current nodes
                size = 100
                color = 'green' if j == route[i] else 'red'
            else:
                size = 30
                color = 'blue'

            ax.scatter(row['lon'], row['lat'], color=color, s=size, zorder=3)
            ax.text(row['lon'], row['lat'], str(j), fontsize=12,
                    fontweight='bold', color='black', zorder=4)

        # Plot previously completed segments
        for idx, path in enumerate(completed_paths):
            try:
                ox.plot_graph_route(G, path, ax=ax, route_color=colors[idx].tolist(),
                                    route_linewidth=1.5, route_alpha=0.6,
                                    show=False, close=False)
            except Exception as e:
                print(f"Error plotting previous path: {str(e)}")

        # Get current segment path
        try:
            orig_node = points_df.loc[route[i], 'node']
            dest_node = points_df.loc[route[i + 1], 'node']

            path = ox.shortest_path(G, orig_node, dest_node, weight='length')

            # Plot current segment with bright color
            ox.plot_graph_route(G, path, ax=ax, route_color=colors[i].tolist(),
                                route_linewidth=3, route_alpha=1.0,
                                show=False, close=False)

            # Add current segment to completed paths
            completed_paths.append(path)

            # Add segment info as text on the plot
            mid_y = (points_df.loc[route[i], 'lat'] + points_df.loc[route[i + 1], 'lat']) / 2
            mid_x = (points_df.loc[route[i], 'lon'] + points_df.loc[route[i + 1], 'lon']) / 2
            ax.text(mid_x, mid_y, f"Segment {i + 1}: {route[i]}→{route[i + 1]}",
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7), zorder=5)

        except Exception as e:
            print(f"Error plotting path from {route[i]} to {route[i + 1]}: {str(e)}")

        # Add title with metrics
        plt.title(f"Route Animation - Segment {i + 1}/{len(route) - 1}\n" +
                  f"Total: Distance = {F[solution_idx][0]:.1f}m, Turns = {F[solution_idx][1]:.1f}")

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=150)
        frame_paths.append(frame_path)
        plt.close(fig)

    # Create video from frames
    print("Creating video...")
    clip = ImageSequenceClip(frame_paths, fps=fps)
    clip.write_videofile(output_video, codec='libx264')

    print(f"Video saved to {output_video}")

    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(frames_dir)
create_route_video(0, res.X, res.F, output_video="Best_route_animation.mp4", fps=0.5)

create_route_video(7, res.X, res.F, output_video="Worst_route_animation.mp4", fps=0.5)
