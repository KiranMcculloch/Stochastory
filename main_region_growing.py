import geopandas as gpd
import shapely
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.validation import make_valid
import numpy as np
from scipy.spatial import KDTree
import os
import time
import gc
import random

class OptimizedRegionMerger:
    def __init__(self, input_path, output_path, target_count, batch_size=100, min_border_length_km=50):
        self.input_path = input_path
        self.output_path = output_path
        self.target_count = target_count
        self.batch_size = batch_size
        self.min_border_length_km = min_border_length_km
        
        # Cleanup Threshold (sq meters) - Applied at start
        self.initial_cleanup_threshold_sq_km = 3000 
        
        self.geometries = []
        self.areas = []
        self.active_mask = None
        self.tree = None
        self.adjacency = None # Now stores tuples: (neighbor_idx, border_length_m)
        self.rebuild_interval = 500
        self.merges_since_rebuild = 0
        self.crs = None
        self.active_indices = []

    def load_and_prepare(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File {self.input_path} not found.")
        
        print(f"Loading {self.input_path}...")
        gdf = gpd.read_file(self.input_path)
        
        # Basic Cleaning
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf[gdf.geometry.notna()]
        
        if gdf.crs.is_geographic:
            print("Reprojecting to EPSG:3857...")
            gdf = gdf.to_crs(epsg=3857)
            
        self.crs = gdf.crs
        self.geometries = list(gdf.geometry)
        self.areas = [g.area for g in self.geometries]
        
        self.active_mask = np.ones(len(self.geometries), dtype=bool)
        self.active_indices = list(range(len(self.geometries)))
        
        print(f"Loaded {len(self.geometries)} valid polygons.")

    def initial_cleanup_and_islands(self):
        # 4. Rebuild index after changes
        self.build_spatial_index()
        print(f"Phase 1 Complete. Active polygons: {len(self.active_indices)}")

    def build_spatial_index(self):
        valid_indices = []
        valid_geoms = []
        
        for idx in self.active_indices:
            if self.active_mask[idx]:
                geom = self.geometries[idx]
                if geom is not None and not geom.is_empty:
                    valid_indices.append(idx)
                    valid_geoms.append(geom)
                else:
                    self.active_mask[idx] = False
        
        self.active_indices = valid_indices
        
        if len(valid_geoms) < 2:
            self.adjacency = {}
            self.tree = None
            return

        self.tree = STRtree(valid_geoms)
        # Adjacency now stores {idx: [(neighbor_idx, border_length_m), ...]}
        self.adjacency = {idx: [] for idx in valid_indices}
        
        min_border_m = self.min_border_length_km * 1000.0

        for i, idx in enumerate(valid_indices):
            geom = valid_geoms[i]
            try:
                neighbor_local_idxs = self.tree.query(geom, predicate='touches')
                for n_local_idx in neighbor_local_idxs:
                    n_global_idx = valid_indices[n_local_idx]
                    if n_global_idx != idx:
                        neighbor_geom = valid_geoms[n_local_idx]
                        
                        # Calculate intersection boundary
                        intersection = geom.boundary.intersection(neighbor_geom.boundary)
                        
                        border_length = 0.0
                        if intersection.is_empty:
                            continue
                        elif intersection.geom_type == 'LineString':
                            border_length = intersection.length
                        elif intersection.geom_type == 'MultiLineString':
                            border_length = sum(line.length for line in intersection.geoms)
                        # Points (corner touches) have length 0.0, so they are automatically filtered below
                        
                        if border_length >= min_border_m:
                            self.adjacency[idx].append((n_global_idx, border_length))
                            
            except Exception:
                continue
        
        self.merges_since_rebuild = 0

    
    def find_best_merges(self):
        candidates = []
        processed = set()
        
        shuffled_indices = self.active_indices.copy()
        random.shuffle(shuffled_indices)
        
        weights = []
        valid_selection_indices = []
        
        for idx in shuffled_indices:
            if not self.active_mask[idx]:
                continue
            area = self.areas[idx]
            if area <= 0: area = 0.1
            # Less aggressive weighting to allow more variety
            weights.append(1.0 / np.sqrt(area))
            valid_selection_indices.append(idx)
        
        if not valid_selection_indices:
            return []

        weights = np.array(weights)
        weights = weights / weights.sum()
        
        pool_size = min(len(valid_selection_indices), self.batch_size * 4) # Larger pool
        selected_candidates = np.random.choice(
            valid_selection_indices, 
            size=pool_size, 
            replace=False, 
            p=weights
        )
        
        for small_idx in selected_candidates:
            if small_idx in processed or not self.active_mask[small_idx]:
                continue
                
            # Adjacency now contains tuples (idx, length), filter for active neighbors
            neighbors_data = self.adjacency.get(small_idx, [])
            if not neighbors_data:
                continue
            
            active_neighbors = [
                n_idx for n_idx, n_len in neighbors_data 
                if self.active_mask[n_idx] and n_idx not in processed
            ]
            
            if not active_neighbors:
                continue
            
            geom_small = self.geometries[small_idx]
            area_small = self.areas[small_idx]
            
            best_score = -np.inf
            best_target = None
            
            for target_idx in active_neighbors:
                geom_target = self.geometries[target_idx]
                area_target = self.areas[target_idx]
                
                try:
                    merged = unary_union([geom_small, geom_target])
                    
                    # RELAXED VALIDATION:
                    # If invalid, try to fix it instead of rejecting
                    if merged and not merged.is_empty:
                        if not merged.is_valid:
                            merged = make_valid(merged)
                        
                        # Check again after repair
                        if merged.is_empty:
                            continue

                        area = merged.area
                        perim = merged.length
                        if perim == 0: continue
                        compactness = (4 * np.pi * area) / (perim ** 2)
                        
                        # Diversity Bonus
                        size_ratio = min(area_small, area_target) / max(area_small, area_target)
                        diversity_bonus = 0
                        if size_ratio > 0.5:
                            diversity_bonus = 0.1
                        elif area_small > 1e8 and area_target > 1e8:
                            diversity_bonus = -0.1 # Reduced penalty
                        
                        score = compactness + diversity_bonus
                        
                        if score > best_score:
                            best_score = score
                            best_target = target_idx
                except Exception:
                    continue
            
            if best_target is not None:
                candidates.append((small_idx, best_target, best_score))
                processed.add(small_idx)
                processed.add(best_target)
                
                if len(candidates) >= self.batch_size:
                    break
        
        return candidates

    def execute_merges(self, merges):
        if not merges:
            return 0
            
        count = 0
        for idx_small, idx_target, _ in merges:
            if not self.active_mask[idx_small] or not self.active_mask[idx_target]:
                continue
                
            g1 = self.geometries[idx_small]
            g2 = self.geometries[idx_target]
            
            try:
                merged = unary_union([g1, g2])
                if merged and not merged.is_empty:
                    if not merged.is_valid:
                        merged = make_valid(merged)
                    
                    if merged.is_empty: # Double check after repair
                        self.active_mask[idx_small] = False # Drop small at least
                        count += 1
                        continue

                    self.active_mask[idx_small] = False
                    self.active_mask[idx_target] = False
                    
                    new_idx = len(self.geometries)
                    self.geometries.append(merged)
                    self.areas.append(merged.area)
                    self.active_mask = np.append(self.active_mask, True)
                    self.active_indices.append(new_idx)
                    
                    # Re-calculate neighbors for the new polygon based on raw geometry touch
                    # Since we don't have pre-calculated borders for the new geom, we do a quick local check
                    # or simply union the neighbor lists and clean up. 
                    # For strict correctness with the new border rule, we should ideally re-scan neighbors 
                    # but for performance in the loop, we union sets and let the next rebuild_spatial_index handle validation.
                    
                    n1 = set([n[0] for n in self.adjacency.get(idx_small, [])])
                    n2 = set([n[0] for n in self.adjacency.get(idx_target, [])])
                    
                    # Remove the merged polygons from the neighbor list
                    combined_neighbors = (n1 | n2) - {idx_small, idx_target}
                    
                    # Store temporarily as (idx, 0.0) until next rebuild calculates exact borders
                    self.adjacency[new_idx] = [(n, 0.0) for n in combined_neighbors]
                    
                    count += 1
                else:
                    # Fallback: drop the smaller one to ensure progress
                    if self.areas[idx_target] > self.areas[idx_small]:
                        self.active_mask[idx_small] = False
                    else:
                        self.active_mask[idx_target] = False
            except Exception:
                self.active_mask[idx_small] = False
        
        self.merges_since_rebuild += count
        return count

    def run(self):
        self.load_and_prepare()
        
        # PHASE 1: Cleanup & Islands
        self.initial_cleanup_and_islands()
        
        start = time.time()
        
        # PHASE 2: Main Merging
        print("--- Phase 2: Main Merging Loop ---")
        while len(self.active_indices) > self.target_count:
            if self.merges_since_rebuild >= self.rebuild_interval:
                self.build_spatial_index()
                gc.collect()
            
            if len(self.active_indices) <= 1:
                break
                
            merges = self.find_best_merges()
            if not merges:
                print("No valid merges found. Stopping.")
                break
                
            performed = self.execute_merges(merges)
            print(f"Batch: Merged {performed} pairs. Remaining: {len(self.active_indices)}")
            
            if performed == 0:
                self.build_spatial_index()
        
        print(f"Done in {time.time()-start:.2f}s")
        self.save()

    def save(self):
        final_geoms = [self.geometries[i] for i in self.active_indices if self.active_mask[i]]
        if not final_geoms:
            print("No geometries to save!")
            return
            
        gdf = gpd.GeoDataFrame(geometry=final_geoms, crs=self.crs)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        gdf.to_file(self.output_path, driver='GeoJSON')
        print(f"Saved {len(final_geoms)} polygons to {self.output_path}")

def main():
    INPUT = './temp/mainland_only.geojson'
    OUTPUT = './temp/merged_shapes.geojson'
    TARGET = 600
    BATCH = 200
    MIN_BORDER_KM = 50  # Configurable minimum border length
    
    try:
        # Pass the new parameter to the class
        merger = OptimizedRegionMerger(INPUT, OUTPUT, TARGET, BATCH, min_border_length_km=MIN_BORDER_KM)
        merger.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
