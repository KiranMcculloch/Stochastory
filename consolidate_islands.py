import geopandas as gpd
import numpy as np
import heapq
from shapely.ops import snap, unary_union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.strtree import STRtree
from tqdm import tqdm
import logging
import time
import random
from collections import Counter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Compatibility Imports ---
try:
    from shapely import make_valid as shp_make_valid
    from shapely.ops import snap, unary_union
except ImportError:
    try:
        from shapely.validation import make_valid as shp_make_valid
        from shapely.ops import snap, unary_union
    except ImportError:
        def shp_make_valid(geom):
            return geom.buffer(0)
        from shapely.ops import snap, unary_union

class UnionFind:
    """Disjoint Set Union for tracking polygon groups."""
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)
        self.count = np.ones(n, dtype=np.int32)
        self.is_island = np.zeros(n, dtype=bool)

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self.count[root_j] += self.count[root_i]
                self.is_island[root_j] = self.is_island[root_j] or self.is_island[root_i]
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self.count[root_i] += self.count[root_j]
                self.is_island[root_i] = self.is_island[root_i] or self.is_island[root_j]
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                self.count[root_i] += self.count[root_j]
                self.is_island[root_i] = self.is_island[root_i] or self.is_island[root_j]
            return True
        return False

def get_island_target_count():
    """
    Returns a random target count for an island cluster.
    Weighted to favor fewer polygons.
    """
    rand = random.random()
    if rand < 0.50:      # 50% chance
        return 1
    elif rand < 0.80:    # 30% chance
        return 2
    elif rand < 0.95:    # 15% chance
        return 3
    else:                # 5% chance
        return 4

def find_neighbors_within_distance(geom, all_geoms, tree, max_distance_m=100000.0):
    """
    Find all geometries within max_distance_m of the input geometry.
    Returns list of indices of neighboring geometries.
    """
    bounds = geom.bounds
    expanded_bounds = (
        bounds[0] - max_distance_m,
        bounds[1] - max_distance_m,
        bounds[2] + max_distance_m,
        bounds[3] + max_distance_m
    )
    
    candidate_indices = list(tree.query(expanded_bounds))
    
    neighbors = []
    for idx in candidate_indices:
        other_geom = all_geoms[idx]
        dist = geom.distance(other_geom)
        if dist <= max_distance_m:
            neighbors.append(idx)
    
    return neighbors

def build_merge_queue(cluster_global_indices, geometries, boundaries, perimeters, sindex, island_set):
    """Build priority queue for island merges (compactness-weighted)."""
    pq = []
    cluster_set = set(cluster_global_indices)
    
    for idx in cluster_global_indices:
        geom = geometries[idx]
        bounds = geom.bounds
        my_bound = boundaries[idx]
        my_perim = perimeters[idx]
        
        neighbors = list(sindex.intersection(bounds))
        
        for n_idx in neighbors:
            if n_idx <= idx:
                continue
            if n_idx not in cluster_set:
                continue
            if n_idx not in island_set:
                continue
            
            neighbor_bound = boundaries[n_idx]
            inter = my_bound.intersection(neighbor_bound)
            
            if inter.is_empty:
                continue
            
            length = inter.length
            if length < 10.0:
                continue
            
            neighbor_perim = perimeters[n_idx]
            smaller_perim = min(my_perim, neighbor_perim)
            
            if smaller_perim <= 0:
                continue
            
            score = length / smaller_perim
            score += random.uniform(0, 0.0001)
            
            heapq.heappush(pq, (-score, idx, n_idx))
    
    return pq

def merge_cluster_probabilistic(uf, cluster_global_indices, target_count, 
                                 geometries, boundaries, perimeters, sindex, island_set):
    """
    Merge island cluster down to target_count polygons using probabilistic approach.
    Returns (merges_performed, final_count)
    """
    original_count = len(cluster_global_indices)
    
    if original_count <= target_count:
        return 0, original_count
    
    required_merges = original_count - target_count
    pq = build_merge_queue(cluster_global_indices, geometries, boundaries, perimeters, sindex, island_set)
    
    if not pq:
        logger.debug(f"    Cluster with {original_count} polygons has no adjacencies.")
        return 0, original_count
    
    merges = 0
    max_iter = len(pq) * 10
    iter_count = 0
    
    while pq and merges < required_merges and iter_count < max_iter:
        iter_count += 1
        score, i, j = heapq.heappop(pq)
        
        root_i = uf.find(i)
        root_j = uf.find(j)
        
        if root_i == root_j:
            continue
        
        uf.union(i, j)
        merges += 1
    
    final_count = original_count - merges
    return merges, final_count

def main():
    start_time = time.time()
    
    # ================= CONFIGURATION =================
    INPUT_FILE = "./temp/clean_basemap.geojson"
    OUTPUT_MAINLAND = "./temp/mainland_only.geojson"
    OUTPUT_ISLANDS = "./temp/islands_consolidated.geojson"
    
    SNAP_TOLERANCE = 0.5
    MIN_BORDER_LENGTH_M = 10.0
    SMALL_ISLAND_THRESHOLD_KM2 = 3000.0
    NEARBY_DISTANCE_M = 100000.0  # 100 km
    # ================================================
    
    logger.info("=" * 70)
    logger.info("ISLAND CONSOLIDATION SCRIPT V3")
    logger.info(f"Small islands (<{SMALL_ISLAND_THRESHOLD_KM2:,.0f} km²): Merge all within {NEARBY_DISTANCE_M/1000:.0f}km → 1 polygon")
    logger.info(f"Normal islands (≥{SMALL_ISLAND_THRESHOLD_KM2:,.0f} km²): Probabilistic 1-4 polygons")
    logger.info("Output: Separate mainland and island GeoJSON files")
    logger.info("=" * 70)
    
    # --- Load Data ---
    logger.info(f"Loading data from {INPUT_FILE}...")
    try:
        gdf = gpd.read_file(INPUT_FILE)
    except FileNotFoundError:
        logger.error(f"File not found: {INPUT_FILE}")
        return
    
    if gdf.empty:
        logger.error("Input file contains no data.")
        return
    
    logger.info(f"Loaded {len(gdf)} polygons.")
    
    if gdf.crs is None:
        logger.warning("No CRS found. Assuming EPSG:3857.")
        gdf.set_crs(epsg=3857, inplace=True)
    
    if 'is_island' not in gdf.columns:
        logger.error("Required column 'is_island' not found.")
        return
    
    # --- Preserve Original Island Status ---
    gdf['original_is_island'] = gdf['is_island'].values.astype(bool)
    
    # --- Pre-processing ---
    logger.info("Cleaning geometries...")
    gdf['geometry'] = gdf.geometry.apply(shp_make_valid)
    gdf['geometry'] = gdf.geometry.apply(lambda g: snap(g, g, tolerance=SNAP_TOLERANCE))
    
    # Pre-extract arrays
    geometries = gdf.geometry.values
    boundaries = [g.boundary for g in geometries]
    perimeters = gdf.geometry.length.values
    areas_km2 = (gdf.geometry.area / 1e6).values
    is_island = gdf['original_is_island'].values.astype(bool)
    
    n_total = len(gdf)
    
    # --- Initialize Union-Find ---
    uf = UnionFind(n_total)
    uf.is_island[:] = is_island
    
    # --- Identify Island Polygons ---
    island_indices = np.where(is_island)[0]
    mainland_indices = np.where(~is_island)[0]
    n_islands = len(island_indices)
    n_mainland = len(mainland_indices)
    island_set = set(island_indices)
    
    logger.info(f"Mainland polygons: {n_mainland}")
    logger.info(f"Island polygons: {n_islands}")
    
    # ================= ISLAND PROCESSING =================
    if n_islands > 0:
        logger.info("\n" + "=" * 70)
        logger.info("ISLAND PROCESSING")
        logger.info("=" * 70)
        
        # --- Step 1: Identify Island Clusters (by adjacency) ---
        logger.info("Building island adjacency graph...")
        
        island_geoms = [geometries[i] for i in island_indices]
        
        # Map: local_index (0..n_islands-1) -> global_index
        local_to_global = {i: gid for i, gid in enumerate(island_indices)}
        global_to_local = {gid: i for i, gid in enumerate(island_indices)}
        
        # Build adjacency matrix for islands
        rows = []
        cols = []
        
        island_sindex = gdf.sindex
        
        for i_local, i_global in enumerate(island_indices):
            bounds = geometries[i_global].bounds
            neighbors_global = list(island_sindex.intersection(bounds))
            
            my_bound = boundaries[i_global]
            
            for j_global in neighbors_global:
                if j_global <= i_global:
                    continue
                if j_global not in global_to_local:
                    continue
                
                j_local = global_to_local[j_global]
                neighbor_bound = boundaries[j_global]
                
                inter = my_bound.intersection(neighbor_bound)
                
                if not inter.is_empty and inter.length > MIN_BORDER_LENGTH_M:
                    rows.append(i_local)
                    cols.append(j_local)
                    rows.append(j_local)
                    cols.append(i_local)
        
        if rows:
            adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_islands, n_islands))
        else:
            adj_matrix = csr_matrix((n_islands, n_islands), dtype=int)
        
        # Find connected components
        logger.info("Identifying island clusters...")
        n_components, labels = connected_components(
            csgraph=adj_matrix, 
            directed=False, 
            return_labels=True
        )
        
        logger.info(f"Found {n_components} distinct island clusters.")
        
        # --- Step 2: Build STRtree for Distance Queries ---
        logger.info("Building spatial index for distance queries...")
        island_tree = STRtree(island_geoms)
        
        # --- Step 3: Process Each Cluster ---
        logger.info("\nProcessing clusters...")
        
        cluster_stats = []
        small_island_merges = 0
        normal_island_merges = 0
        target_distribution = Counter()
        
        for c_id in tqdm(range(n_components), desc="Processing Clusters"):
            # Get polygons in this cluster
            cluster_local = np.where(labels == c_id)[0]
            cluster_global = [local_to_global[i] for i in cluster_local]
            original_count = len(cluster_global)
            
            # Calculate total cluster area
            cluster_area = sum(areas_km2[i] for i in cluster_global)
            
            if original_count <= 1:
                cluster_stats.append((c_id, original_count, 1, 0, cluster_area, "single", 1))
                target_distribution[1] += 1
                continue
            
            # --- Check if this is a SMALL island cluster ---
            if cluster_area < SMALL_ISLAND_THRESHOLD_KM2:
                # SMALL ISLAND: Merge ALL polygons within 100km into single multipolygon
                logger.debug(f"  Cluster {c_id}: SMALL ({cluster_area:,.2f} km², {original_count} polys) → auto-merge to 1")
                
                # Find all island polygons within 100km of ANY polygon in this cluster
                merge_candidates = set(cluster_global)
                
                for global_idx in cluster_global:
                    geom = geometries[global_idx]
                    nearby_local = find_neighbors_within_distance(
                        geom, island_geoms, island_tree, NEARBY_DISTANCE_M
                    )
                    for local_idx in nearby_local:
                        merge_candidates.add(local_to_global[local_idx])
                
                # Merge all candidates into single region
                merge_list = list(merge_candidates)
                candidate_geoms = [geometries[i] for i in merge_list if geometries[i] is not None and not geometries[i].is_empty]
                
                if len(candidate_geoms) > 1:
                    try:
                        # Union all geometries
                        merged_geom = unary_union(candidate_geoms)
                        
                        # Update Union-Find: merge all into first candidate
                        first_idx = merge_list[0]
                        for other_idx in merge_list[1:]:
                            uf.union(first_idx, other_idx)
                            small_island_merges += 1
                        
                        final_count = 1
                        method = "small_auto_merge"
                        target_distribution[1] += 1
                    except Exception as e:
                        logger.warning(f"  Cluster {c_id}: Merge failed - {e}")
                        final_count = original_count
                        method = "merge_failed"
                        target_distribution[original_count] += 1
                else:
                    final_count = 1
                    method = "single_geom"
                    target_distribution[1] += 1
            else:
                # NORMAL ISLAND: Use probabilistic 1-4 polygon targeting
                target_count = get_island_target_count()
                target_count = min(target_count, original_count)
                
                logger.debug(f"  Cluster {c_id}: NORMAL ({cluster_area:,.2f} km², {original_count} polys) → target {target_count}")
                
                # Merge using priority queue
                merges, final_count = merge_cluster_probabilistic(
                    uf, cluster_global, target_count,
                    geometries, boundaries, perimeters, island_sindex, island_set
                )
                
                normal_island_merges += merges
                method = "probabilistic_1-4"
                target_distribution[final_count] += 1
            
            cluster_stats.append((c_id, original_count, final_count, original_count - final_count, cluster_area, method, final_count))
        
        # --- Log Cluster Statistics ---
        logger.info("\n" + "=" * 70)
        logger.info("ISLAND CONSOLIDATION RESULTS")
        logger.info("=" * 70)
        
        # Final polygon count distribution
        final_counts = [s[2] for s in cluster_stats]
        count_distribution = Counter(final_counts)
        
        logger.info("\nFinal Island Polygon Distribution:")
        for count in sorted(count_distribution.keys()):
            num_clusters = count_distribution[count]
            pct = num_clusters / len(cluster_stats) * 100 if len(cluster_stats) > 0 else 0
            logger.info(f"  {count} polygon(s): {num_clusters} clusters ({pct:.1f}%)")
        
        # By method
        method_counts = Counter([s[5] for s in cluster_stats])
        logger.info("\nMerge Methods:")
        for method, count in method_counts.items():
            logger.info(f"  {method}: {count} clusters")
        
        # Target distribution (for probabilistic merges)
        logger.info("\nTarget Distribution (probabilistic clusters only):")
        for target in sorted(target_distribution.keys()):
            num_clusters = target_distribution[target]
            pct = num_clusters / len(cluster_stats) * 100 if len(cluster_stats) > 0 else 0
            logger.info(f"  Target {target} polygon(s): {num_clusters} clusters ({pct:.1f}%)")
        
        total_original = sum(s[1] for s in cluster_stats)
        total_final = sum(s[2] for s in cluster_stats)
        total_merges = sum(s[3] for s in cluster_stats)
        
        logger.info("\n" + "-" * 70)
        logger.info(f"Total island clusters:        {len(cluster_stats)}")
        logger.info(f"Total island polygons before: {total_original}")
        logger.info(f"Total island polygons after:  {total_final}")
        logger.info(f"Small island merges:          {small_island_merges}")
        logger.info(f"Normal island merges:         {normal_island_merges}")
        logger.info(f"Total merges performed:       {total_merges}")
        logger.info(f"Reduction:                    {(1 - total_final/total_original)*100:.1f}%" if total_original > 0 else "N/A")
        logger.info("=" * 70)
    else:
        logger.info("No islands found. Skipping island processing.")
    
    # ================= SPLIT OUTPUT =================
    logger.info("\n" + "=" * 70)
    logger.info("CREATING SPLIT OUTPUT FILES")
    logger.info("=" * 70)
    
    # Get final region assignments
    final_roots = np.array([uf.find(i) for i in range(n_total)])
    gdf['region_id'] = final_roots
    
    # Restore island status from UnionFind
    gdf['is_island'] = np.array([uf.is_island[uf.find(i)] for i in range(n_total)])
    
    # --- Output 1: Mainland Only ---
    logger.info(f"\nSaving mainland to {OUTPUT_MAINLAND}...")
    mainland_gdf = gdf[gdf['is_island'] == False].copy()
    
    if len(mainland_gdf) > 0:
        mainland_dissolved = mainland_gdf.dissolve(by='region_id', as_index=False)
        mainland_dissolved['area_km2'] = mainland_dissolved.geometry.area / 1e6
        mainland_dissolved['polygon_count'] = mainland_gdf.groupby('region_id').size().reindex(
            mainland_dissolved.index, fill_value=1
        ).astype(int)
        mainland_dissolved['geometry'] = mainland_dissolved.geometry.apply(
            lambda g: snap(g, g, tolerance=SNAP_TOLERANCE)
        )
        mainland_cols = ['geometry', 'region_id', 'area_km2', 'polygon_count']
        mainland_dissolved = mainland_dissolved[[c for c in mainland_cols if c in mainland_dissolved.columns]]
        mainland_dissolved.to_file(OUTPUT_MAINLAND, driver="GeoJSON")
        logger.info(f"  Saved {len(mainland_dissolved)} mainland regions")
    else:
        logger.warning("  No mainland polygons to save!")
        empty_gdf = gpd.GeoDataFrame({'geometry': []}, crs=gdf.crs)
        empty_gdf.to_file(OUTPUT_MAINLAND, driver="GeoJSON")
    
    # --- Output 2: Islands Only ---
    logger.info(f"\nSaving islands to {OUTPUT_ISLANDS}...")
    island_gdf = gdf[gdf['is_island'] == True].copy()
    
    if len(island_gdf) > 0:
        island_dissolved = island_gdf.dissolve(by='region_id', as_index=False)
        island_dissolved['area_km2'] = island_dissolved.geometry.area / 1e6
        island_dissolved['polygon_count'] = island_gdf.groupby('region_id').size().reindex(
            island_dissolved.index, fill_value=1
        ).astype(int)
        island_dissolved['is_island'] = True
        island_dissolved['geometry'] = island_dissolved.geometry.apply(
            lambda g: snap(g, g, tolerance=SNAP_TOLERANCE)
        )
        island_cols = ['geometry', 'region_id', 'area_km2', 'polygon_count', 'is_island']
        island_dissolved = island_dissolved[[c for c in island_cols if c in island_dissolved.columns]]
        island_dissolved.to_file(OUTPUT_ISLANDS, driver="GeoJSON")
        logger.info(f"  Saved {len(island_dissolved)} island regions")
    else:
        logger.warning("  No island polygons to save!")
        empty_gdf = gpd.GeoDataFrame({'geometry': [], 'is_island': []}, crs=gdf.crs)
        empty_gdf.to_file(OUTPUT_ISLANDS, driver="GeoJSON")
    
    # ================= FINAL SUMMARY =================
    end_time = time.time()
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Process complete in {end_time - start_time:.2f} seconds.")
    logger.info(f"Output files:")
    logger.info(f"  Mainland: {OUTPUT_MAINLAND}")
    logger.info(f"  Islands:  {OUTPUT_ISLANDS}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
