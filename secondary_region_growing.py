import geopandas as gpd
import shapely
import networkx as nx
import random
from shapely.ops import unary_union
import os
import pandas as pd

def assign_continents(input_gdf, continent_gdf, continent_name_col='name'):
    """
    Assigns each polygon in input_gdf to a continent based on the largest area of overlap.
    Returns the input_gdf with a new column 'continent'.
    """
    print("Assigning continents to polygons...")
    
    # Ensure both are in the same CRS for accurate area calculation
    if input_gdf.crs != continent_gdf.crs:
        continent_gdf = continent_gdf.to_crs(input_gdf.crs)
        print(f"Reprojected continents to match input CRS: {input_gdf.crs}")
    
    # Spatial join to find all intersections
    # 'intersects' finds any polygon that touches or overlaps a continent
    joined = gpd.sjoin(input_gdf, continent_gdf, how="left", predicate="intersects", lsuffix='poly', rsuffix='cont')
    
    # Handle polygons that might not intersect any continent (e.g., small islands far off, or data errors)
    # They will have NaN in the continent columns.
    
    results = []
    
    # Group by the original polygon index to handle multiple overlaps
    # joined.index corresponds to the original input_gdf index
    for idx in input_gdf.index:
        matches = joined[joined.index == idx]
        
        if matches.empty or pd.isna(matches[continent_name_col]).all():
            results.append({'index': idx, 'continent': 'Unknown'})
            continue
            
        # If only one match, take it
        if len(matches) == 1:
            cont_name = matches[continent_name_col].iloc[0]
            results.append({'index': idx, 'continent': cont_name})
        else:
            # Multiple matches: Calculate intersection area for each
            max_area = -1
            best_cont = None
            
            poly_geom = input_gdf.geometry.iloc[idx]
            
            for _, row in matches.iterrows():
                if pd.isna(row[continent_name_col]):
                    continue
                cont_geom = row['geometry'] # This is the continent geometry from the join
                
                # Calculate intersection
                intersection = poly_geom.intersection(cont_geom)
                if not intersection.is_empty:
                    area = intersection.area
                    if area > max_area:
                        max_area = area
                        best_cont = row[continent_name_col]
            
            if best_cont:
                results.append({'index': idx, 'continent': best_cont})
            else:
                # Fallback if intersections were somehow empty despite sjoin saying intersects (rare precision issues)
                # Take the first non-null name
                first_valid = matches[continent_name_col].dropna().iloc[0] if not matches[continent_name_col].dropna().empty else 'Unknown'
                results.append({'index': idx, 'continent': first_valid})

    # Create a mapping series
    continent_map = pd.DataFrame(results).set_index('index')['continent']
    
    # Assign to the original gdf
    input_gdf = input_gdf.copy()
    input_gdf['continent'] = input_gdf.index.map(continent_map)
    
    # Fill any remaining NaNs (shouldn't happen with the logic above, but safe)
    input_gdf['continent'] = input_gdf['continent'].fillna('Unknown')
    
    counts = input_gdf['continent'].value_counts()
    print("Continent assignment complete:")
    print(counts)
    
    return input_gdf

def get_neighbors_with_min_length(gdf, min_length_meters=50000):
    """
    Creates a graph where nodes are polygon indices.
    Edges are only added if the shared boundary length exceeds min_length_meters.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    edge_data = {}
    
    centroids = gdf.centroid
    neighbors = gpd.sjoin(gdf, gdf, how="inner", predicate="intersects")
    
    for i, row in neighbors.iterrows():
        u = i
        v = row['index_right']
        
        if u >= v:
            continue
            
        geom_u = gdf.geometry.iloc[u]
        geom_v = gdf.geometry.iloc[v]
        
        intersection = geom_u.intersection(geom_v)
        
        if intersection.is_empty:
            continue
            
        boundary_length = 0.0
        
        if intersection.geom_type in ['LineString', 'MultiLineString']:
            boundary_length = intersection.length
        elif intersection.geom_type in ['Polygon', 'MultiPolygon']:
            boundary_length = intersection.boundary.length
            
        if boundary_length >= min_length_meters:
            cent_u = centroids.iloc[u]
            cent_v = centroids.iloc[v]
            centroid_dist = cent_u.distance(cent_v)
            
            G.add_edge(u, v)
            edge_data[(u, v)] = {
                'length': boundary_length,
                'centroid_dist': centroid_dist
            }
            
    return G, edge_data

def find_surrounded_polygons(gdf, G, edge_data):
    """Identifies surrounded polygons and returns {node: best_neighbor}."""
    surrounded_forced_merges = {}
    # Only check a subset if dataset is huge to save time, otherwise all
    # Using a simple heuristic: nodes with many neighbors are candidates
    candidates = [n for n, deg in G.degree() if deg >= 3]
    
    for node in candidates:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
            
        geom_node = gdf.geometry.iloc[node]
        boundary_node = geom_node.boundary
        neighbor_geoms = [gdf.geometry.iloc[n] for n in neighbors]
        union_neighbors = unary_union(neighbor_geoms)
        
        diff = boundary_node.difference(union_neighbors)
        
        if diff.is_empty or diff.length < 1.0:
            best_neighbor = None
            max_len = -1
            for n in neighbors:
                u, v = sorted((node, n))
                length = edge_data.get((u, v), {}).get('length', 0)
                if length > max_len:
                    max_len = length
                    best_neighbor = n
            
            if best_neighbor is not None:
                surrounded_forced_merges[node] = best_neighbor
                
    return surrounded_forced_merges

def merge_polygons_continent_aware(gdf, continent_probs, default_prob=0.25, 
                                   min_neighbor_length_m=50000, 
                                   centroid_ratio_threshold=1.0, 
                                   seed=None):
    """
    Merges polygons using continent-specific probabilities.
    """
    if seed is not None:
        random.seed(seed)
        
    # 1. Build Graph
    G, edge_data = get_neighbors_with_min_length(gdf, min_length_meters=min_neighbor_length_m)
    
    # 2. Identify Enclaves
    surrounded_map = find_surrounded_polygons(gdf, G, edge_data)
    
    # 3. Build Active Graph
    G_active = nx.Graph()
    G_active.add_nodes_from(G.nodes())
    
    forced_edges = set()
    
    # Apply Enclave Logic
    for node, best_neighbor in surrounded_map.items():
        u, v = sorted((node, best_neighbor))
        G_active.add_edge(u, v)
        forced_edges.add((u, v))
    
    # Apply Random Merging with Continent Probabilities
    skipped_by_shape = 0
    skipped_by_prob = 0
    merged_count = 0
    
    for u, v in G.edges():
        if (u, v) in forced_edges:
            continue
            
        data = edge_data.get((u, v))
        if not data:
            continue
            
        # Get Continents for both nodes
        cont_u = gdf.iloc[u]['continent']
        cont_v = gdf.iloc[v]['continent']
        
        # Determine Probability
        # Strategy: Use the probability of the first polygon's continent, 
        # or if they differ, use the average or the minimum? 
        # ELM suggests using the probability of the 'u' polygon, or an average if they differ.
        # Let's use the average of the two continents' probabilities to be fair.
        
        prob_u = continent_probs.get(cont_u, default_prob)
        prob_v = continent_probs.get(cont_v, default_prob)
        
        # If one is Unknown, rely on the known one. If both unknown, use default.
        if cont_u == 'Unknown' and cont_v != 'Unknown':
            effective_base_prob = prob_v
        elif cont_v == 'Unknown' and cont_u != 'Unknown':
            effective_base_prob = prob_u
        else:
            effective_base_prob = (prob_u + prob_v) / 2.0
            
        # Shape Filter (Hard Filter)
        length = data['length']
        dist = data['centroid_dist']
        if dist == 0: dist = 0.1
        
        ratio = length / dist
        if ratio < centroid_ratio_threshold:
            skipped_by_shape += 1
            continue
            
        # Probability Check
        if random.random() < effective_base_prob:
            G_active.add_edge(u, v)
            merged_count += 1
        else:
            skipped_by_prob += 1
    
    print(f"Edge Processing: {merged_count} merged, {skipped_by_shape} skipped (shape), {skipped_by_prob} skipped (prob)")

    # 4. Clusters
    clusters = list(nx.connected_components(G_active))
    
    merged_features = []
    
    print(f"Total original polygons: {len(gdf)}")
    print(f"Number of clusters formed: {len(clusters)}")
    
    if not clusters:
        gdf_out = gdf.copy()
        gdf_out['cluster_id'] = [f"cluster_{i}" for i in range(len(gdf))]
        gdf_out['cluster_size'] = 1
        gdf_out['original_indices'] = [[i] for i in range(len(gdf))]
        return gdf_out

    cluster_sizes = [len(c) for c in clusters]
    print(f"Cluster size distribution: min={min(cluster_sizes)}, max={max(cluster_sizes)}")
    print(f"Isolated (size 1): {sum(1 for s in cluster_sizes if s == 1)}")
    print(f"Large (size >= 10): {sum(1 for s in cluster_sizes if s >= 10)}")

    for cluster in clusters:
        indices = sorted(list(cluster))
        subset = gdf.iloc[indices]
        
        if len(subset) == 1:
            new_geom = subset.geometry.iloc[0]
        else:
            new_geom = unary_union(subset.geometry)
            
        # Preserve continent info in the output (take the mode or first)
        cluster_continents = subset['continent'].tolist()
        main_continent = max(set(cluster_continents), key=cluster_continents.count)
            
        new_attrs = {
            'geometry': new_geom,
            'cluster_id': f"cluster_{min(indices)}", 
            'cluster_size': len(indices),
            'original_indices': indices,
            'continent': main_continent
        }
        merged_features.append(new_attrs)

    crs = gdf.crs
    geometries = [f['geometry'] for f in merged_features]
    data_keys = [k for k in merged_features[0].keys() if k != 'geometry']
    data_columns = {k: [f[k] for f in merged_features] for k in data_keys}
    
    return gpd.GeoDataFrame(data_columns, geometry=geometries, crs=crs)

# --- Main Execution ---
def main():
    input_path = './temp/merged_shapes.geojson'
    continent_path = './reference_data/continents.geojson' # Update this path to your continents file
    output_path = './temp/merged_shapes_secondary.geojson'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        print(f"Loading input data from {input_path}...")
        gdf = gpd.read_file(input_path)
        
        print(f"Loading continent data from {continent_path}...")
        if not os.path.exists(continent_path):
            raise FileNotFoundError(f"Continent file not found at {continent_path}. Please ensure the file exists.")
            
        continent_gdf = gpd.read_file(continent_path)
        
        # Ensure input is in a projected CRS (meters) for length/area checks
        # EPSG:3857 is good for this script as per previous context
        if gdf.crs and gdf.crs.to_epsg() == 3857:
            print("Input CRS confirmed as EPSG:3857.")
        else:
            print("Reprojecting input to EPSG:3857 for consistent meter calculations...")
            gdf = gdf.to_crs(epsg=3857)
            
        # Assign continents
        gdf = assign_continents(gdf, continent_gdf, continent_name_col='CONTINENT') # Adjust 'name' to your column name
        
        # Define Probabilities per Continent
        # Adjust these values to control clustering density per region
        continent_probabilities = {
            'North America': 0.65,
            'South America': 0.3,
            'Europe': 0,     
            'Asia': 0.55,        
            'Africa': 0.15,
            'Australia': 0.6,
            'Unknown': 0.10      # Fallback for islands/others
        }
        
        default_prob = 0
        min_len = 50000 
        centroid_threshold = 1.0 
        
        print(f"Running continent-aware merge...")
        print(f"Probabilities: {continent_probabilities}")
        
        merged_gdf = merge_polygons_continent_aware(
            gdf, 
            continent_probs=continent_probabilities,
            default_prob=default_prob,
            min_neighbor_length_m=min_len,
            centroid_ratio_threshold=centroid_threshold,
        )

        merged_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Success! Output saved to {output_path}")
        
        print("\nOutput Summary by Continent:")
        print(merged_gdf.groupby('continent')['cluster_size'].describe())

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()