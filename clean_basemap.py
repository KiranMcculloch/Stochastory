import sys
import os

# # Get the directory where the current script is located
# script_dir = os.path.dirname(os.path.realpath(__file__))

# # Define the path to your bundled libraries
# libs_path = os.path.join(script_dir, 'libs')

# # Add to sys.path if it's not already there
# if libs_path not in sys.path:
#     sys.path.append(libs_path)

import geopandas as gpd
import numpy as np
from shapely.ops import snap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
import logging
import time
from collections import defaultdict

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Compatibility Imports for Shapely ---
try:
    from shapely import make_valid as shp_make_valid
except ImportError:
    try:
        from shapely.validation import make_valid as shp_make_valid
    except ImportError:
        def shp_make_valid(geom):
            return geom.buffer(0)

class UnionFind:
    """Optimized Union-Find with path compression and rank."""
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)
        self.area = np.zeros(n, dtype=np.float64) # Track area per root

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self.area[root_j] += self.area[root_i]
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self.area[root_i] += self.area[root_j]
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                self.area[root_i] += self.area[root_j]
            return True
        return False

def main():
    start_time = time.time()
    
    # Configuration
    INPUT_FILE = "./temp/basemap.geojson"
    OUTPUT_FILE = "./temp/clean_basemap.geojson"
    
    SNAP_TOLERANCE = 0.5  # Meters
    MIN_BORDER_LENGTH_M = 0.1 
    SMALL_AREA_THRESHOLD_KM2 = 3000.0
    CLUSTER_AREA_THRESHOLD_KM2 = 7_000_000.0 # Islands < 10M km²
    
    logger.info(f"Loading data from {INPUT_FILE}...")
    try:
        try:
            gdf = gpd.read_file(INPUT_FILE, engine='pyarrow')
        except:
            gdf = gpd.read_file(INPUT_FILE)
    except FileNotFoundError:
        logger.error(f"File not found at {INPUT_FILE}")
        return

    if gdf.empty:
        logger.error("Input file contains no data.")
        return

    logger.info(f"Loaded {len(gdf)} polygons.")
    
    if gdf.crs is None:
        logger.warning("No CRS found. Assuming EPSG:3857.")
        gdf.set_crs(epsg=3857, inplace=True)
    
    # --- Step 1: Topological Cleaning ---
    logger.info("Cleaning topology (make_valid + snap)...")
    gdf['geometry'] = gdf.geometry.apply(shp_make_valid)
    gdf['geometry'] = gdf.geometry.apply(lambda g: snap(g, g, tolerance=SNAP_TOLERANCE))
    
    # --- Step 2: Calculate Initial Areas ---
    logger.info("Calculating areas...")
    gdf['area_km2'] = gdf.geometry.area / 1e6
    initial_areas = gdf['area_km2'].values.copy()
    n = len(gdf)
    
    # --- Step 3: Build Adjacency Graph ---
    # We need both connectivity (for clusters) and weights (for merging small polys)
    logger.info("Building adjacency graph...")
    rows = []
    cols = []
    weights = [] # Store border length
    
    sindex = gdf.sindex
    boundaries = [geom.boundary for geom in gdf.geometry]
    all_bounds = gdf.geometry.bounds.values
    
    for i in tqdm(range(n), desc="Detecting Neighbors"):
        bounds = all_bounds[i]
        neighbors = list(sindex.intersection(bounds))
        my_bound = boundaries[i]
        
        for j in neighbors:
            if j <= i: 
                continue
            
            inter = my_bound.intersection(boundaries[j])
            if not inter.is_empty and inter.length > MIN_BORDER_LENGTH_M:
                length = inter.length
                # Add to lists (symmetric)
                rows.append(i); cols.append(j); weights.append(length)
                rows.append(j); cols.append(i); weights.append(length)
    
    # Create Sparse Matrix for Weights (for merging logic)
    weight_matrix = csr_matrix((weights, (rows, cols)), shape=(n, n))
    
    # Create Binary Sparse Matrix for Connectivity (for cluster logic)
    # Any non-zero weight implies connection
    connectivity_matrix = (weight_matrix > 0).astype(int)
    
    # --- Step 4: Merge Small Polygons (Graph Contraction) ---
    logger.info(f"Merging polygons < {SMALL_AREA_THRESHOLD_KM2} km²...")
    uf = UnionFind(n)
    uf.area[:] = initial_areas # Initialize areas
    
    # Identify initially small polygons
    small_mask = initial_areas < SMALL_AREA_THRESHOLD_KM2
    small_indices = np.where(small_mask)[0]
    
    logger.info(f"Found {len(small_indices)} small polygons to process.")
    
    max_passes = 20
    for passage in range(max_passes):
        # Refresh current areas from UF
        # We need to check the area of the *group* each small poly belongs to
        roots = np.array([uf.find(i) for i in range(n)])
        current_group_areas = uf.area
        
        # Find polygons that are STILL small (their group area is < threshold)
        # Note: We check the root's area
        group_is_small = current_group_areas < SMALL_AREA_THRESHOLD_KM2
        active_mask = group_is_small[roots]
        
        # We only care about original small polygons or groups that are still small
        # But strictly, we want to merge any GROUP that is small.
        # Let's find unique roots that are small and have neighbors
        small_roots = np.unique(roots[active_mask])
        
        if len(small_roots) == 0:
            logger.info("No more small groups found. Convergence reached.")
            break
            
        merges_this_pass = 0
        
        # For each small root, find the best neighbor root to merge into
        for r_idx in small_roots:
            # Find all original polygons belonging to this root
            members = np.where(roots == r_idx)[0]
            if len(members) == 0: continue
            
            # Find neighbors of this group by checking neighbors of any member
            # Optimization: Check neighbors of the first member, if none, try others?
            # Better: Aggregate neighbors from all members
            neighbor_candidates = set()
            for m in members:
                start = weight_matrix.indptr[m]
                end = weight_matrix.indptr[m+1]
                neighs = weight_matrix.indices[start:end]
                neighbor_candidates.update(neighs)
            
            if not neighbor_candidates:
                continue # Isolated small group (will be tagged as island later)
            
            best_target_root = -1
            max_shared_len = -1.0
            
            # Check candidates
            for cand_idx in neighbor_candidates:
                cand_root = uf.find(cand_idx)
                if cand_root == r_idx:
                    continue # Same group
                
                # Calculate total shared border between group r_idx and group cand_root
                # This is expensive if done naively. 
                # Approximation: Use the max border found between any member of r and any member of cand
                # For performance, we just pick the single strongest link found in the matrix
                # Let's iterate members again to find max link to this specific cand_root
                # Actually, let's just find the single strongest edge from ANY member of r_idx to ANY neighbor
                
                # Optimized: Just find the max weight edge from any member of r_idx to any node NOT in r_idx
                pass 
            
            # Simpler Approach for Performance:
            # Iterate all members of r_idx, find their strongest neighbor that is NOT in r_idx
            local_best_target = -1
            local_max_len = -1.0
            
            for m in members:
                start = weight_matrix.indptr[m]
                end = weight_matrix.indptr[m+1]
                neighs = weight_matrix.indices[start:end]
                wts = weight_matrix.data[start:end]
                
                if len(neighs) == 0: continue
                
                # Find neighbor with max weight that is not in our group
                for k, n_idx in enumerate(neighs):
                    n_root = uf.find(n_idx)
                    if n_root == r_idx:
                        continue
                    if wts[k] > local_max_len:
                        local_max_len = wts[k]
                        local_best_target = n_root
            
            if local_best_target != -1:
                uf.union(r_idx, local_best_target)
                merges_this_pass += 1
        
        logger.info(f"Pass {passage+1}: Merged {merges_this_pass} groups.")
        if merges_this_pass == 0:
            break

    # Update roots after merging
    final_roots = np.array([uf.find(i) for i in range(n)])
    gdf['merge_group_id'] = final_roots
    
    # Update areas in GDF for the next step (cluster analysis)
    # Map root -> area
    root_areas = uf.area
    gdf['group_area_km2'] = root_areas[final_roots]
    
    logger.info(f"Merging complete. Reduced to {len(np.unique(final_roots))} groups.")
    
    # --- Step 5: Cluster Analysis on MERGED Groups ---
    # We need to find connected components of the *merged* groups.
    # Since we used Union-Find based on adjacency, the 'merge_group_id' already represents 
    # connected components IF we merged everything. 
    # BUT we only merged small things. Large things might still be separate clusters.
    # We need to run connected_components on the graph of the FINAL groups.
    
    logger.info("Running final cluster analysis on merged groups...")
    
    # Create a mapping from old index to new group ID
    # We need to build a new adjacency matrix between GROUP IDs
    unique_groups = np.unique(final_roots)
    group_map = {gid: i for i, gid in enumerate(unique_groups)}
    n_groups = len(unique_groups)
    
    g_rows = []
    g_cols = []
    
    # Iterate original edges and map to groups
    coo = weight_matrix.tocoo()
    for i, j in zip(coo.row, coo.col):
        g_i = group_map[final_roots[i]]
        g_j = group_map[final_roots[j]]
        if g_i != g_j:
            g_rows.append(g_i)
            g_cols.append(g_j)
            
    group_adj = csr_matrix((np.ones(len(g_rows)), (g_rows, g_cols)), shape=(n_groups, n_groups))
    
    n_clusters, cluster_labels = connected_components(csgraph=group_adj, directed=False, return_labels=True)
    logger.info(f"Found {n_clusters} final clusters.")
    
    # Map cluster labels back to original rows
    # group_id -> cluster_label
    group_to_cluster = {gid: cluster_labels[group_map[gid]] for gid in unique_groups}
    gdf['cluster_id'] = [group_to_cluster[gid] for gid in gdf['merge_group_id']]
    
    # Calculate Total Cluster Area
    cluster_areas = gdf.groupby('cluster_id')['group_area_km2'].first() # Area is same for all members of group
    # Wait, group_area_km2 is the area of the merged group. 
    # We need sum of group areas per cluster.
    # Since each row is a polygon, and we have group_area_km2 repeated, we can't just sum.
    # Better: Sum the ORIGINAL areas per cluster.
    gdf['original_area_km2'] = initial_areas
    cluster_total_areas = gdf.groupby('cluster_id')['original_area_km2'].transform('sum')
    gdf['cluster_total_area_km2'] = cluster_total_areas
    
    # --- Step 6: Tag Islands ---
    logger.info(f"Tagging islands (Cluster Area < {CLUSTER_AREA_THRESHOLD_KM2:,} km²)...")
    gdf['is_island'] = (gdf['cluster_total_area_km2'] < CLUSTER_AREA_THRESHOLD_KM2) | ((gdf['cluster_total_area_km2'] > 10000000.0) & (gdf['cluster_total_area_km2'] < 35000000.0))
    gdf['is_island'] = gdf['is_island'].astype(bool)
    gdf['is_island'] = gdf['is_island'].fillna(False)
    
    island_count = gdf['is_island'].sum()
    logger.info(f"Tagged {island_count} polygons as islands.")
    
    # Debug Stats
    cluster_stats = gdf.groupby('cluster_id').agg(
        total_area=('original_area_km2', 'sum'),
        count=('cluster_id', 'count'),
        is_island=('is_island', 'first')
    ).sort_values(by='total_area', ascending=False)
    
    logger.info("Top 5 Clusters:")
    for i, row in cluster_stats.head(5).iterrows():
        status = "ISLAND" if row['is_island'] else "MAINLAND"
        logger.info(f"Cluster {i}: {row['total_area']:,.2f} km² -> {status}")

    # --- Step 7: Final Dissolve ---
    # We dissolve by 'merge_group_id' to get the merged small polygons.
    # The 'is_island' tag is consistent within a group.
    logger.info("Dissolving geometries...")
    dissolved = gdf.dissolve(by='merge_group_id', as_index=False)
    
    # Ensure is_island is preserved (take first value, they are all same in a group)
    dissolved['is_island'] = dissolved['is_island'].astype(bool)
    dissolved['cluster_total_area_km2'] = dissolved['cluster_total_area_km2'] # Should be consistent
    
    # Final Snap
    logger.info("Final topology cleanup...")
    dissolved['geometry'] = dissolved.geometry.apply(lambda g: snap(g, g, tolerance=SNAP_TOLERANCE))
    
    # Save
    cols_to_save = ['geometry', 'is_island', 'cluster_total_area_km2']
    dissolved = dissolved[cols_to_save]
    
    logger.info(f"Saving to {OUTPUT_FILE}...")
    dissolved.to_file(OUTPUT_FILE, driver="GeoJSON")
    
    end_time = time.time()
    logger.info(f"Process complete in {end_time - start_time:.2f} seconds.")
    logger.info(f"Output features: {len(dissolved)}")
    logger.info(f"Islands: {dissolved['is_island'].sum()}")

if __name__ == "__main__":
    main()
