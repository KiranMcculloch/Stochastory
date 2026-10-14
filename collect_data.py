from pathlib import Path
import json
from shapely.strtree import STRtree
from shapely.geometry import shape, mapping
from shapely import area, intersection
import networkx as nx
import random

# Configuration
INPUT_FILE = "./temp/processed_map.geojson"
OUTPUT_FILE = "./temp/data.geojson"
REFERENCE_DIR = "./reference_data"  # Change this to your folder name

def load_geojson_features(filepath):
    """Load GeoJSON and return features list with parsed geometries."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    features = []
    for feat in data.get('features', []):
        geom = shape(feat['geometry']) if feat.get('geometry') else None
        props = feat.get('properties', {})
        # Initialize the unified list here
        props['matched_references'] = []
        features.append({
            'geometry': geom,
            'properties': props,
            'name_en': props.get('NAME_EN', None)
        })
    return features, data.get('crs', {'type': 'name', 'properties': {'name': 'EPSG:4326'}})

def build_spatial_index(features):
    """Build STRtree spatial index from features with valid geometries."""
    geometries = [f['geometry'] for f in features if f['geometry'] is not None]
    valid_indices = [i for i, f in enumerate(features) if f['geometry'] is not None]
    
    if not geometries:
        return None, []
    
    tree = STRtree(geometries)
    return tree, valid_indices

def process_intersections(input_features, input_areas, input_tree, input_valid_idx,
                          ref_features, ref_areas, ref_tree, ref_valid_idx):
    """Calculate intersections and append weighted results to input features."""
    matches_count = 0
    
    for ref_pos, ref_feat in enumerate(ref_features):
        if ref_feat['geometry'] is None:
            continue
        
        ref_idx = ref_valid_idx[ref_pos]
        ref_geom = ref_feat['geometry']
        ref_name_en = ref_feat['name_en']
        ref_area_val = ref_areas[ref_pos]
        
        if ref_name_en is None or ref_area_val == 0:
            continue
        
        # Query the spatial index for candidate input geometries
        candidate_positions = input_tree.query(ref_geom, predicate='intersects')
        
        for candidate_pos in candidate_positions:
            input_idx = input_valid_idx[candidate_pos]
            input_geom = input_features[input_idx]['geometry']
            input_area_val = input_areas[input_idx]
            
            if input_area_val == 0:
                continue
            
            # Calculate precise intersection
            inter_geom = intersection(input_geom, ref_geom)
            
            if inter_geom.is_empty:
                continue
            
            inter_area = area(inter_geom)
            
            # Calculate percentages
            pct_of_input = int(round(100 * inter_area / input_area_val))
            pct_of_ref = int(round(100 * inter_area / ref_area_val))
            
            # Clamp to 0-100
            pct_of_input = max(0, min(100, pct_of_input))
            pct_of_ref = max(0, min(100, pct_of_ref))
            
            # Discard if percent_of_input is 0
            if pct_of_input == 0:
                continue
            
            # Calculate composite weight
            reference_weight = pct_of_input * pct_of_ref
            
            if pct_of_ref >= 98 and random.random() > 0.8:
                reference_weight = 10000
            
            # Append to the unified list (duplicates handled later)
            input_features[input_idx]['properties']['matched_references'].append({
                'name': ref_name_en,
                'reference_weight': reference_weight
            })
            matches_count += 1
    
    return matches_count

def build_border_graph(features, valid_indices):
    """
    Build a NetworkX graph where nodes are feature indices and edges represent 
    shared boundaries OR proximity within 100km.
    
    ASSUMPTION: Input data is in EPSG:3857 (Web Mercator).
    - Units are meters.
    - Distance calculations are direct (no degree conversion needed).
    - Buffers are specified in meters.
    """
    G = nx.Graph()
    
    # Add nodes for all valid features
    for idx in valid_indices:
        G.add_node(idx)
    
    if len(valid_indices) < 2:
        return G
        
    # Extract geometries for the valid indices
    geometries = [features[i]['geometry'] for i in valid_indices]
    
    # Build a spatial index on the POLYGONS themselves.
    # In EPSG:3857, this index operates in meters.
    tree = STRtree(geometries)
    
    # Pre-calculate boundaries to avoid repeated computation during the loop
    boundaries = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            boundaries.append(None)
        else:
            boundaries.append(geom.boundary)
    
    # Define the fallback distance threshold in meters (100km)
    DISTANCE_THRESHOLD = 100000.0
    
    # Iterate through each feature to find neighbors
    for i, local_idx_a in enumerate(valid_indices):
        global_idx_a = valid_indices[local_idx_a]
        geom_a = geometries[local_idx_a]
        bound_a = boundaries[local_idx_a]
        
        if geom_a is None or bound_a is None:
            continue
            
        # OPTIMIZATION: Query candidates using a buffer of exactly 100km.
        # Since CRS is EPSG:3857, buffer(100000) creates a 100km zone around the geometry.
        # Any polygon intersecting this zone is a candidate for the distance check.
        # This also naturally includes all touching/overlapping polygons.
        try:
            query_zone = geom_a.buffer(DISTANCE_THRESHOLD)
        except Exception:
            # Fallback if buffer fails (e.g., invalid geometry), use the geom itself
            query_zone = geom_a
            
        candidate_local_positions = tree.query(query_zone, predicate='intersects')
        
        seen_neighbors = set()
        
        for candidate_local_pos in candidate_local_positions:
            local_idx_b = valid_indices[candidate_local_pos]
            global_idx_b = valid_indices[local_idx_b]
            
            # Avoid self-loops and duplicate edges (ensure a < b)
            if global_idx_b <= global_idx_a:
                continue
            
            if global_idx_b in seen_neighbors:
                continue
                
            geom_b = geometries[candidate_local_pos]
            bound_b = boundaries[candidate_local_pos]
            
            if geom_b is None:
                continue
            
            is_neighbor = False
            
            # CHECK 1: Shared Boundary (Topological)
            # We look for a linear intersection (LineString) between boundaries.
            if bound_a is not None and bound_b is not None:
                try:
                    inter_bound = bound_a.intersection(bound_b)
                    
                    if not inter_bound.is_empty:
                        # Check if the intersection is a line (shared edge)
                        if inter_bound.geom_type in ['LineString', 'MultiLineString']:
                            if inter_bound.length > 1e-6: # Small tolerance for floating point errors in meters
                                is_neighbor = True
                        # Handle GeometryCollection (e.g., line + point)
                        elif inter_bound.geom_type == 'GeometryCollection':
                            has_line = any(
                                g.geom_type in ['LineString', 'MultiLineString'] and g.length > 1e-6
                                for g in inter_bound.geoms
                            )
                            if has_line:
                                is_neighbor = True
                except Exception:
                    pass # Skip geometric errors for this pair
            
            # CHECK 2: Fallback Distance (within 100km)
            # Only run this expensive calculation if topological check failed.
            # Since we queried with a 100km buffer, we know they are close, 
            # but we need the exact distance to confirm <= 100km.
            if not is_neighbor:
                try:
                    # In EPSG:3857, .distance() returns meters directly.
                    dist_meters = geom_a.distance(geom_b)
                    if dist_meters <= DISTANCE_THRESHOLD:
                        is_neighbor = True
                except Exception:
                    pass
            
            if is_neighbor:
                G.add_edge(global_idx_a, global_idx_b)
                seen_neighbors.add(global_idx_b)
    
    return G


def apply_topological_coloring(features, valid_indices):
    """Apply equitable coloring to features based on topological borders."""
    if not valid_indices:
        print("No valid features to color.")
        return
        
    # Build border graph
    print("Building border graph...")
    border_graph = build_border_graph(features, valid_indices)
    
    if len(border_graph.nodes) == 0:
        print("No nodes in border graph.")
        return
            
    # # Check maximum degree to ensure valid coloring
    # if len(border_graph.edges) > 0:
    #     max_degree = max([d for n, d in border_graph.degree()])
    #     if num_colors <= max_degree:
    #         print(f"Warning: Number of colors ({num_colors}) should be greater than maximum degree ({max_degree}) for guaranteed equitable coloring. Increasing colors to {max_degree + 1}.")
    #         num_colors = max_degree + 1
    # else:
    #     # No edges, 1 color is sufficient, but we respect user input if > 0
    #     if num_colors < 1:
    #         num_colors = 1
    
    try:
        # Apply equitable coloring
        # Note: equitable_color requires num_colors > max_degree
        coloring = nx.coloring.greedy_color(border_graph)
        
        
        
        # Assign colors to features
        for node_idx, color in coloring.items():
            color += 1
            if random.random() < 0.5:
                features[node_idx]['properties']['color'] = color*2
            else:
                features[node_idx]['properties']['color'] = (color*2)-1

            
        # Handle any features that might not be in the graph (e.g., invalid geoms) though valid_indices should cover them
        # If a feature was isolated and not added due to logic gaps, ensure it gets a color (default 0)
        # for idx in valid_indices:
        #     if idx not in coloring:
        #         features[idx]['properties']['color'] = 0
                
        # print(f"Applied topological border coloring with {num_colors} colors.")
        
    except nx.NetworkXAlgorithmError as e:
        print(f"Error applying equitable coloring: {e}")
        print("Skipping coloring step.")
    except Exception as e:
        print(f"Unexpected error during coloring: {e}")
        print("Skipping coloring step.")

def main():
    input_path = Path(INPUT_FILE)
    ref_dir = Path(REFERENCE_DIR)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    if not ref_dir.exists() or not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {REFERENCE_DIR}")

    # Discover all .geojson files in the directory
    all_geojsons = list(ref_dir.glob("*.geojson"))
    all_geojsons += list(ref_dir.glob("*.json"))
    
    # Filter out the input file if it happens to be in the same directory
    reference_files = [
        f for f in all_geojsons 
        if f.resolve() != input_path.resolve()
    ]
    
    if not reference_files:
        raise ValueError(f"No reference GeoJSON files found in '{REFERENCE_DIR}' (excluding input file).")

    print(f"Found {len(reference_files)} reference files in '{REFERENCE_DIR}'.")

    # Load input features once
    input_features, input_crs = load_geojson_features(INPUT_FILE)
    print(f"Loaded {len(input_features)} input features")
    
    # Pre-calculate input areas
    input_areas = []
    valid_input_indices = []
    valid_input_geoms = []
    
    for i, feat in enumerate(input_features):
        if feat['geometry'] is not None and not feat['geometry'].is_empty:
            input_areas.append(area(feat['geometry']))
            valid_input_indices.append(i)
            valid_input_geoms.append(feat['geometry'])
        else:
            input_areas.append(0.0)
    
    if not valid_input_geoms:
        raise ValueError("No valid geometries in input file")
        
    input_tree = STRtree(valid_input_geoms)
    print(f"Built input spatial index with {len(valid_input_indices)} geometries")
    
    total_matches = 0
    
    # Process each reference file found in the directory
    for ref_path in reference_files:
        print(f"\nProcessing {ref_path.name}...")
        
        ref_features, ref_crs = load_geojson_features(ref_path)
        print(f"  Loaded {len(ref_features)} reference features")
        
        # Validate 'NAME_EN' field exists
        if not any(f['name_en'] is not None for f in ref_features):
            print(f"  Warning: {ref_path.name} has no valid 'NAME_EN' fields, skipping")
            continue
        
        # Pre-calculate reference areas
        ref_areas = []
        valid_ref_indices = []
        valid_ref_geoms = []
        
        for i, feat in enumerate(ref_features):
            if feat['geometry'] is not None and not feat['geometry'].is_empty:
                ref_areas.append(area(feat['geometry']))
                valid_ref_indices.append(i)
                valid_ref_geoms.append(feat['geometry'])
            else:
                ref_areas.append(0.0)
        
        if not valid_ref_geoms:
            print(f"  Warning: No valid geometries in {ref_path.name}, skipping")
            continue
            
        ref_tree = STRtree(valid_ref_geoms)
        print(f"  Built reference spatial index with {len(valid_ref_indices)} geometries")
        
        # Process intersections
        matches = process_intersections(
            input_features, input_areas, input_tree, valid_input_indices,
            ref_features, ref_areas, ref_tree, valid_ref_indices
        )
        total_matches += matches
        print(f"  Found {matches} matches")
        
        # Explicitly free memory
        del ref_features, ref_tree, ref_areas, valid_ref_indices, valid_ref_geoms
    
    print(f"\nTotal raw matches found: {total_matches}")
    print("Deduplicating, sorting, and cleaning output...")
    
    # Post-processing: Deduplicate, Sort, and Clean
    features_with_matches = 0
    total_removed_duplicates = 0
    
    for feat in input_features:
        matches = feat['properties'].get('matched_references', [])
        
        if not matches:
            if 'matched_references' in feat['properties']:
                del feat['properties']['matched_references']
            continue
        
        # --- DEDUPLICATION LOGIC ---
        # Dictionary to store the best match for each name
        best_matches = {}
        
        for item in matches:
            name = item['name']
            weight = item['reference_weight']
            
            if name not in best_matches:
                best_matches[name] = weight
            else:
                # Keep the higher weight
                if weight > best_matches[name]:
                    best_matches[name] = weight
        
        # Reconstruct the list with only unique names and their highest weights
        unique_matches = [
            {'name': name, 'reference_weight': weight}
            for name, weight in best_matches.items()
        ]
        
        duplicates_removed = len(matches) - len(unique_matches)
        total_removed_duplicates += duplicates_removed
        
        # Update the feature property
        if not unique_matches:
            del feat['properties']['matched_references']
        else:
            # Sort by reference_weight descending
            unique_matches.sort(key=lambda x: x['reference_weight'], reverse=True)
            feat['properties']['matched_references'] = unique_matches
            features_with_matches += 1
            
    print(f"Removed {total_removed_duplicates} duplicate entries.")
    print(f"Features with matches: {features_with_matches}")
    
    # Apply topological coloring (NEW: Based on Borders)
    print("\nApplying topological border coloring...")
    apply_topological_coloring(input_features, valid_input_indices)
    
    # Build output GeoJSON structure
    output_features = []
    for feat in input_features:
        out_feat = {
            'type': 'Feature',
            'geometry': mapping(feat['geometry']) if feat['geometry'] else None,
            'properties': feat['properties']
        }
        output_features.append(out_feat)
    
    output_data = {
        'type': 'FeatureCollection',
        'features': output_features,
        'crs': input_crs
    }
    
    # Write output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
