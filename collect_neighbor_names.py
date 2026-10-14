import geopandas as gpd
from shapely.geometry import mapping

def find_polygon_neighbors(input_file, output_file, id_field='generated_name'):
    """
    Loads a GeoJSON of polygons, identifies neighbors based on shared boundaries,
    and saves the neighbor IDs (based on a specific field) to the properties.
    
    Parameters:
    - input_file: Path to the input GeoJSON file.
    - output_file: Path for the output GeoJSON file.
    - id_field: The field name to use as the unique identifier for neighbors.
    """
    
    # Load the GeoJSON file
    print(f"Loading data from {input_file}...")
    try:
        gdf = gpd.read_file(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Validate that the ID field exists
    if id_field not in gdf.columns:
        print(f"Error: The field '{id_field}' does not exist in the GeoJSON properties.")
        print(f"Available fields: {list(gdf.columns)}")
        return

    # Ensure the data is projected correctly for spatial operations.
    # While touches/intersects works in lat/lon, projecting to a local metric 
    # system is often safer for complex geometries to avoid precision issues.
    # However, for simple topological checks, we can proceed if the CRS is consistent.
    if gdf.crs is None:
        print("Warning: No CRS found. Assuming coordinates are valid for topological checks.")
    
    print("Calculating neighbors...")
    
    # Initialize a list to store neighbor lists
    neighbors_list = []

    # Create a spatial index for faster lookups (optional but recommended for large datasets)
    sindex = gdf.sindex

    # Iterate through each feature
    for index, row in gdf.iterrows():
        current_geom = row.geometry
        current_id = row[id_field]
        neighbor_ids = []

        # Find potential neighbors using the spatial index (bounding box match)
        # This speeds up the process significantly compared to checking every pair
        possible_matches_index = list(sindex.intersection(current_geom.bounds))
        
        for match_index in possible_matches_index:
            if match_index == index:
                continue
            
            candidate_geom = gdf.iloc[match_index].geometry
            candidate_id = gdf.iloc[match_index][id_field]

            # Check for topological touch (shared boundary)
            # 'touches' returns True if geometries share at least one point but do not share interior points.
            # This is the standard definition of polygon adjacency.
            if current_geom.touches(candidate_geom):
                neighbor_ids.append(candidate_id)
            
            # Alternative: Use 'intersects' if you consider overlapping polygons as neighbors
            # elif current_geom.intersects(candidate_geom) and not current_geom.equals(candidate_geom):
            #     neighbor_ids.append(candidate_id)

        neighbors_list.append(neighbor_ids)

    # Add the new list of neighbors to the GeoDataFrame
    # We create a new column called 'neighbors' containing the list of names
    gdf['neighbors'] = neighbors_list

    # Save to output GeoJSON
    print(f"Saving results to {output_file}...")
    try:
        gdf.to_file(output_file, driver='GeoJSON')
        print("Success! Neighbor lists have been added to the 'neighbors' property.")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    # Configuration
    input_path = './temp/names.geojson'   # Replace with your input file path
    output_path = './temp/neighbors.geojson' # The requested output filename
    id_column = 'generated_name'   # The field to use as the ID

    find_polygon_neighbors(input_path, output_path, id_column)

if __name__ == "__main__":
    main()
