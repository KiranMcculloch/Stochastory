import json
import webbrowser
import sys
import folium
import numpy as np
import geopandas as gpd
import uuid
import os

def convert_arrays_to_lists(val):
    """Recursively converts numpy arrays to lists for JSON serialization."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {k: convert_arrays_to_lists(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [convert_arrays_to_lists(i) for i in val]
    else:
        return val

def main():
    new_id = uuid.uuid1()
    input_path = "./temp/output.geojson"
    output_path = f"./outputs/map_{new_id}.html"

    # 1. Load and Reproject
    try:
        gdf = gpd.read_file(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if gdf.crs is None:
        print("Warning: No CRS found. Assuming EPSG:3857.")
        gdf = gdf.set_crs(epsg=3857)
    
    # We need the area in meters for logic, so we calculate it NOW while in EPSG:3857
    # .area returns square meters for EPSG:3857
    gdf['area_sq_meters'] = gdf.geometry.area
    
    # Then reproject to Lat/Lon for Folium visualization
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    # 2. Clean Data (NumPy arrays -> Lists)
    for col in gdf_wgs84.columns:
        if col != gdf_wgs84.geometry.name:
            gdf_wgs84[col] = gdf_wgs84[col].apply(convert_arrays_to_lists)
    
    # Ensure the area column is also a standard float (not numpy float)
    gdf_wgs84['area_sq_meters'] = gdf_wgs84['area_sq_meters'].apply(lambda x: float(x) if not np.isnan(x) else 0.0)

    # 3. Define Color Mapping
    colors = {
        1: "#dd1013", 2: "#1C6BFF", 3: "#2da635", 4: "#a753b4",
        5: "#ff7f00", 6: "#ffff33", 7: "#a65628", 8: "#f781bf",
        9: "#379eb8", 10: "#DD1763", 11: "#999898", 12: "#8AEE1F"
    }
    
    dark_colors = {
        1: "#8c1012", 2: "#0D3379", 3: "#1A3619", 4: "#4b2550",
        5: "#bb5e00", 6: "#b9ad26", 7: "#643317", 8: "#9b5078",
        9: "#1f486a", 10: "#8A103F", 11: "#686868", 12: "#4A810F"
    }

    # 5. Initialize Map
    m = folium.Map(location=[30,0], zoom_start=3, tiles="http://services.arcgisonline.com/arcgis/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}",attr="ARCGIS Online")
    m._name = "map"
    m._id = "1"
    # 6. Add GeoJson Layer (Polygons only)
    gj = folium.GeoJson(
        gdf_wgs84,
        style_function=lambda feature: {
            "fillColor": colors.get(feature["properties"].get("color"), "#000000"),
            "color": dark_colors.get(feature["properties"].get("color"), "#000000"),
            "weight": 2,
            "fillOpacity": 0.5,
        },
        highlight_function=lambda feature: {
            "color":"#000000",
            "fillOpacity": 0.7,
            "weight": 4,
        },
        # style_function=lambda feature: {
        #     "fillColor": "green"
        #     if "e" in feature["properties"]["name"].lower()
        #     else "#ffff00",
        #     "color": "black",
        #     "weight": 2,
        #     "dashArray": "5, 5",
        # }
    
        popup=folium.GeoJsonPopup(
            fields=["history"],
            aliases=["History:"],
            # labels=True,
        ),
        popup_keep_highlighted=True,
    )
    gj._name = "geojson"
    gj._id = "1"
    gj.add_to(m)

    # 7. Prepare Data for Dynamic Markers
    # We will create a list of dictionaries containing: id, lat, lon, name, area
    markers_data = []
    
    for idx, row in gdf_wgs84.iterrows():
        name = row["generated_name"]
        color = colors[row["color"]]
        dark_color = dark_colors[row["color"]]
        if not name:
            continue
            
        centroid = row.geometry.centroid
        lon, lat = centroid.coords[0]
        area = row['area_sq_meters']
        
        markers_data.append({
            "id": idx,
            "name": name,
            "lat": lat,
            "lon": lon,
            "area": area,
            "color": color,
            "dark_color": dark_color
        })

       # ... (Previous code remains the same until Step 8) ...

        # 8. Inject Custom JavaScript for Dynamic Visibility
    js_code = f"""
        document.addEventListener('DOMContentLoaded', function() {{
        // Wait slightly to ensure Folium's own initialization scripts have run
        setTimeout(function() {{
            
            if (typeof map_1 === 'undefined') {{
                console.error("ELM Debug: Map object not found. Ensure no other maps are on the page conflicting with the name 'map'.");
                return;
            }}

            const markersData = {json.dumps(markers_data)};
            const markers = [];

            // Create markers
            markersData.forEach((data, index) => {{
                const htmlLabel = `
                    <div style="
                        font-size: 10pt; 
                        font-family: serif;
                        font-weight: bold; 
                        color: black; 
                        text-shadow: 2px 2px 2px ${{data.dark_color}}, -1px -1px 2px ${{data.color}};
                        width: 7.5vw;
                        text-align:center;
                        pointer-events: none;
                        line-height: 10pt;
                        ">
                        ${{data.name}}
                    </div>
                `;

                const marker = L.marker([data.lat, data.lon], {{
                    icon: L.divIcon({{
                        html: htmlLabel,
                        className: 'dynamic-label',
                        iconSize: [150, 36],
                        iconAnchor: [75, 18]
                    }})
                }});
                
                marker.featureArea = data.area;
                marker.featureName = data.name;
                markers.push(marker);
            }});

            function updateMarkersVisibility() {{
                const zoom = map_1.getZoom();
                const currentThreshold =  100_000_000 / Math.pow(zoom,5);
                
                markers.forEach(marker => {{
                    const isVisible = map_1.hasLayer(marker);
                    const shouldBeVisible = (marker.featureArea/1_000_000) > currentThreshold;

                    if (shouldBeVisible && !isVisible) {{
                        map_1.addLayer(marker);
                    }} else if (!shouldBeVisible && isVisible) {{
                        map_1.removeLayer(marker);
                    }}
                }});
            }}

            // Initial run
            updateMarkersVisibility();
            
            // Attach listeners
            map_1.on('zoomend', updateMarkersVisibility);
            map_1.on('moveend', updateMarkersVisibility);
            
            console.log("ELM Debug: Dynamic labels initialized with " + markers.length + " markers.");

        }}, 100); // 100ms delay to let Folium finish setup
       
    }});
    """

    # Inject into the script section
    m.get_root().script.add_child(folium.Element(js_code))
    
    if not os.path.exists("./outputs"):
    
    # if the demo_folder directory is not present 
    # then create it.
        os.makedirs("./outputs")

    m.save(output_path)
    print(f"Map with dynamic labels successfully saved to {output_path}")
    
        # Read in the file
    with open(output_path, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('e.target.setStyle(highlightStyle);', 'e.target.setStyle(highlightStyle);\ne.target.bringToFront();')
    filedata = filedata.replace(".leaflet-container { font-size: 1rem; }",".leaflet-container { font-size: 1rem; }\n\t\t\t.leaflet-marker-icon.leaflet-interactive { pointer-events: none}")
    # Write the file out again
    with open(output_path, 'w') as file:
        file.write(filedata)
    
    
    os.rename(input_path, f"./outputs/output_{new_id}.geojson")
    
    webbrowser.open(output_path)

if __name__ == "__main__":
    main()
