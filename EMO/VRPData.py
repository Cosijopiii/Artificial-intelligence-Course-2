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
    if random.random() < 0.5:
          bias=1.5  # 50% chance to increase travel time by 50%

    data['travel_time'] = data['length'] / speed_mps*bias  # seconds

# Build distance and time matrices
distance_matrix = []
time_matrix = []

for i in range(len(delivery_points)):
    row_dist = []
    row_time = []
    for j in range(len(delivery_points)):
        orig = delivery_points[i]['node']
        dest = delivery_points[j]['node']
        if orig == dest:
            row_dist.append(0)
            row_time.append(0)
        else:
            try:
                dist = nx.shortest_path_length(G, orig, dest, weight='length')
                time = nx.shortest_path_length(G, orig, dest, weight='travel_time')
            except:
                dist = float('inf')
                time = float('inf')
            row_dist.append(dist)
            row_time.append(time)
    distance_matrix.append(row_dist)
    time_matrix.append(row_time)

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

n_points = dist_matrix.shape[0]
start_node = n_points - 1  # Central node

# All permutations of delivery nodes (excluding start node)
nodes = list(range(n_points - 1))
all_perms = list(itertools.permutations(nodes))

results = np.zeros((len(all_perms), 3))  # columns: distance, time, co2

for idx, perm in enumerate(all_perms):
    route = [start_node] + list(perm) + [start_node]
    total_dist = 0
    total_time = 0
    total_co2 = 0
    for i in range(len(route) - 2):
        total_dist += dist_matrix[route[i], route[i + 1]]
        total_time += time_matrix[route[i], route[i + 1]]
        total_co2 += co2_matrix[route[i], route[i + 1]]
    results[idx] = [total_dist, total_time, total_co2]

# Save results to CSV
np.savetxt('EMO/all_permutations_objectives.csv', results, delimiter=',', header='distance,time,co2', comments='')


#%%
import matplotlib.pyplot as plt

# Scatter plot of the first two columns (distance vs time) of all permutations
plt.figure(figsize=(8, 6))
plt.scatter(results[:, 0], results[:, 2], alpha=0.5, s=10)
plt.xlabel('Total Distance')
plt.ylabel('Total Time')
plt.title('All Permutations: Distance vs Time')
plt.grid(True)
plt.tight_layout()
plt.show()
