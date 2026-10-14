import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
from scipy.spatial.distance import pdist, squareform
import numpy as np

def process_islands_aggressive_v3(islands_path, mainland_path, output_path):
    """
    Updated Logic:
    1. Filters out non-polygon geometries and islands < 100 km².
    2. Clusters non-touching islands based on distance.
       - Small (<3000 km²) involvement: 200km radius.
       - Large-Large interaction: 50km radius.
    3. CRITICAL: Never merges islands that directly touch/border each other.
    4. Merges resulting clusters to mainland if within 300km (with probability logic).
    """
    
    # Constants
    MIN_ISLAND_AREA_THRESHOLD_M2 = 100000000.0    # 100 km²
    SMALL_ISLAND_AREA_THRESHOLD_M2 = 3000000000.0 # 3,000 km²
    SMALL_ISLAND_CLUSTER_RADIUS = 200000.0        # 200 km
    LARGE_ISLAND_CLUSTER_RADIUS = 50000.0         # 50 km
    MAINLAND_MERGE_THRESHOLD_METERS = 400000.0    # 400 km
    LARGE_AREA_MERGE_PROB_THRESHOLD_M2 = 5000000000.0 # 5,000 km²
    LARGE_AREA_MERGE_PROBABILITY = 0.85

    print("Loading GeoJSON files...")
    islands_gdf = gpd.read_file(islands_path)
    mainland_gdf = gpd.read_file(mainland_path)

    # 1. Filter Geometries: Keep only Polygons and MultiPolygons
    print("Filtering non-polygon geometries...")
    def is_polygon(geom):
        if geom is None:
            return False
        return geom.geom_type in ['Polygon', 'MultiPolygon']

    islands_gdf = islands_gdf[islands_gdf.geometry.apply(is_polygon)].copy()
    mainland_gdf = mainland_gdf[mainland_gdf.geometry.apply(is_polygon)].copy()

    if len(islands_gdf) == 0:
        print("No valid polygon islands found. Saving mainland only.")
        mainland_gdf.to_file(output_path, driver="GeoJSON")
        return

    # Validate CRS
    target_crs = "EPSG:3857"
    if islands_gdf.crs is None or str(islands_gdf.crs) != target_crs:
        print(f"Reprojecting Islands to {target_crs}...")
        islands_gdf = islands_gdf.to_crs(target_crs)
    
    if mainland_gdf.crs is None or str(mainland_gdf.crs) != target_crs:
        print(f"Reprojecting Mainland to {target_crs}...")
        mainland_gdf = mainland_gdf.to_crs(target_crs)

    # 2. Preprocessing: Discard islands < 100 km²
    print(f"Filtering islands smaller than 100 km²...")
    initial_island_count = len(islands_gdf)
    islands_gdf['area_m2'] = islands_gdf.geometry.area
    islands_gdf = islands_gdf[islands_gdf['area_m2'] >= MIN_ISLAND_AREA_THRESHOLD_M2].copy()
    
    removed_count = initial_island_count - len(islands_gdf)
    print(f"Removed {removed_count} islands < 100 km². Remaining: {len(islands_gdf)}")
    
    if 'area_m2' in islands_gdf.columns:
        islands_gdf.drop(columns=['area_m2'], inplace=True)

    if len(islands_gdf) == 0:
        print("No islands remain. Saving mainland only.")
        mainland_gdf.to_file(output_path, driver="GeoJSON")
        return

    # Reset index for clean integer handling
    islands_gdf = islands_gdf.reset_index(drop=True)
    mainland_gdf = mainland_gdf.reset_index(drop=True)

    # 3. Identify Isolated Islands (No Borders)
    # We only cluster islands that do NOT touch anything else.
    # Islands that touch neighbors are kept as-is (Anchors) and excluded from distance clustering.
    all_geoms = gpd.GeoDataFrame(geometry=pd.concat([islands_gdf.geometry, mainland_gdf.geometry], ignore_index=True))
    sindex = all_geoms.sindex
    
    isolated_islands_indices = []
    
    print("Identifying isolated islands (no direct borders)...")
    for idx in range(len(islands_gdf)):
        island_geom = islands_gdf.iloc[idx].geometry
        if island_geom.is_empty:
            continue
            
        possible_matches = list(sindex.intersection(island_geom.bounds))
        has_border = False
        
        for match_idx in possible_matches:
            other_geom = all_geoms.iloc[match_idx].geometry
            # Check for touch or intersect
            if island_geom.touches(other_geom) or island_geom.intersects(other_geom):
                if not island_geom.equals_exact(other_geom, tolerance=0.001):
                    has_border = True
                    break
        
        if not has_border:
            isolated_islands_indices.append(idx)

    isolated_islands_gdf = islands_gdf.iloc[isolated_islands_indices].copy()
    
    # Separate non-isolated (keep as is)
    mask = pd.Series(True, index=islands_gdf.index)
    mask.iloc[isolated_islands_indices] = False
    remaining_islands_gdf = islands_gdf[mask].copy()

    print(f"Found {len(isolated_islands_gdf)} isolated islands to consider for clustering.")
    print(f"Found {len(remaining_islands_gdf)} islands with existing neighbors (kept separate).")

    output_rows = []
    
    # Add non-isolated islands directly to output (unchanged)
    if len(remaining_islands_gdf) > 0:
        for _, row in remaining_islands_gdf.iterrows():
            output_rows.append(row.to_dict())

    if len(isolated_islands_gdf) == 0:
        for _, row in mainland_gdf.iterrows():
            output_rows.append(row.to_dict())
    else:
        # 4. Distance-Based Clustering (Only for Non-Touching Islands)
        print("Clustering isolated islands based on distance (ignoring direct borders)...")
        
        n_islands = len(isolated_islands_gdf)
        centroids = np.array([list(geom.centroid.coords)[0] for geom in isolated_islands_gdf.geometry])
        areas = np.array([geom.area for geom in isolated_islands_gdf.geometry])
        
        clusters = []
        
        if n_islands > 1:
            dist_matrix = squareform(pdist(centroids))
            
            adjacency = np.zeros((n_islands, n_islands), dtype=bool)
            
            for i in range(n_islands):
                for j in range(i + 1, n_islands):
                    dist = dist_matrix[i, j]
                    
                    # Since these are all "isolated" (no borders), we don't need to re-check touch here.
                    # They are guaranteed not to touch by the previous step.
                    
                    min_area = min(areas[i], areas[j])
                    
                    if min_area < SMALL_ISLAND_AREA_THRESHOLD_M2:
                        threshold = SMALL_ISLAND_CLUSTER_RADIUS
                    else:
                        threshold = LARGE_ISLAND_CLUSTER_RADIUS
                    
                    if dist <= threshold:
                        adjacency[i, j] = True
                        adjacency[j, i] = True
            
            # Find connected components
            visited = np.zeros(n_islands, dtype=bool)
            for i in range(n_islands):
                if not visited[i]:
                    cluster_indices = []
                    stack = [i]
                    while stack:
                        node = stack.pop()
                        if not visited[node]:
                            visited[node] = True
                            cluster_indices.append(node)
                            neighbors = np.where(adjacency[node])[0]
                            for n in neighbors:
                                if not visited[n]:
                                    stack.append(n)
                    clusters.append(cluster_indices)
        else:
            clusters = [[0]]

        print(f"Formed {len(clusters)} island clusters from isolated islands.")

        mainland_modifications = {} 
        merged_mainland_indices = set()

        for cluster_indices in clusters:
            cluster_islands = isolated_islands_gdf.iloc[cluster_indices]
            
            cluster_geom = unary_union(cluster_islands.geometry)
            if cluster_geom.geom_type == 'MultiPolygon':
                cluster_multipolygon = cluster_geom
            else:
                cluster_multipolygon = MultiPolygon([cluster_geom])
            
            cluster_area = cluster_multipolygon.area
            cluster_centroid = cluster_multipolygon.centroid
            
            # Find nearest mainland
            min_dist = float('inf')
            nearest_idx = -1
            
            for idx, m_row in mainland_gdf.iterrows():
                d = cluster_centroid.distance(m_row.geometry)
                if d < min_dist:
                    min_dist = d
                    nearest_idx = idx
            
            should_merge = False
            
            if min_dist <= MAINLAND_MERGE_THRESHOLD_METERS:
                should_merge = True
                if cluster_area > LARGE_AREA_MERGE_PROB_THRESHOLD_M2:
                    if np.random.random() > LARGE_AREA_MERGE_PROBABILITY:
                        should_merge = False
                        print(f"Large cluster ({cluster_area/1e9:.2f} km²) rolled against merge chance.")
            
            if should_merge:
                if nearest_idx in mainland_modifications:
                    current_geom = mainland_modifications[nearest_idx]
                    mainland_modifications[nearest_idx] = unary_union([current_geom, cluster_multipolygon])
                else:
                    original_geom = mainland_gdf.iloc[nearest_idx].geometry
                    mainland_modifications[nearest_idx] = unary_union([original_geom, cluster_multipolygon])
                
                merged_mainland_indices.add(nearest_idx)
                print(f"Merged cluster (Area: {cluster_area/1e9:.2f} km², Dist: {min_dist/1000:.1f}km) with mainland.")
            else:
                new_record = {
                    'geometry': cluster_multipolygon,
                    'type': 'isolated_cluster',
                    'area_m2': cluster_area,
                    'dist_to_mainland_m': min_dist,
                    'merge_attempted': False
                }
                output_rows.append(new_record)

        # Add modified mainland
        for idx, new_geom in mainland_modifications.items():
            original_row = mainland_gdf.iloc[idx].copy()
            original_row.geometry = new_geom
            original_row['source_status'] = 'merged_with_islands'
            output_rows.append(original_row.to_dict())
        
        # Add unmodified mainland
        for idx, row in mainland_gdf.iterrows():
            if idx not in merged_mainland_indices:
                output_rows.append(row.to_dict())

    final_gdf = gpd.GeoDataFrame(output_rows, crs=target_crs, geometry='geometry')
    
    print(f"Saving output to {output_path}...")
    final_gdf.to_file(output_path, driver="GeoJSON")
    print("Process complete.")

def main():
    islands_file = "./temp/islands_consolidated.geojson"
    mainland_file = "./temp/merged_shapes_secondary.geojson"
    output_file = "./temp/processed_map.geojson"
    process_islands_aggressive_v3(islands_file, mainland_file, output_file)

if __name__ == "__main__":
    main()
