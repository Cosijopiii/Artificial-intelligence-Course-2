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
    speed_kph = 40
    speed_mps = speed_kph * 1000 / 3600
    data['travel_time'] = data['length'] / speed_mps  # seconds

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

# Export results
pd.DataFrame(delivery_points).to_csv("ciudad_ixtepec_points.csv", index=False)
pd.DataFrame(distance_matrix).to_csv("ciudad_ixtepec_distance_matrix.csv", index=False)
pd.DataFrame(time_matrix).to_csv("ciudad_ixtepec_time_matrix.csv", index=False)

#%%
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

# Load road network
place = "Ciudad Ixtepec, Oaxaca, Mexico"
G = ox.graph_from_place(place, network_type='drive')

# Load delivery points
df_points = pd.read_csv("ciudad_ixtepec_points.csv")
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

