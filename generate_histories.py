import requests
import json
import time
import os
import uuid
import multiprocessing as mp
from functools import partial

# Security Best Practice: Load API key from environment variable
MODEL_NAME = 'Qwen/Qwen3.5-397B-A17B-FP8' 
FILE_ENCODING = 'UTF-8'
new_field_name = 'history' 


def send_prompt_to_elm(prompt_text):  
    headers = {
        'accept': 'application/json',
        'Authorization': 'Bearer '+API_KEY,
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': MODEL_NAME,
        'reasoning_effort': 'none',
        'messages': [
            {
                'role': 'user',
                'content': prompt_text,
            },
        ],
    }

    try:
        response = requests.post('https://elm.edina.ac.uk/api/v1/chat/completions', headers=headers, json=json_data, timeout=30)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        result_content = response.json()["choices"][0]["message"]["content"]
        print(f"{result_content[:100]}...")
        return result_content
        
    except:
        return "Nothing is known about this place."


def create_context_prompt(feature_data):
    """
    Constructs a prompt to generate a descriptive paragraph based on 
    a generated name and its neighbors.
    """
    props = feature_data.get('properties', {})
    
    # Extract inputs
    generated_name = props.get('generated_name', 'Unknown Location')
    neighbors_raw = props.get('neighbors', [])
    
    # Normalize neighbors to a simple list of names
    neighbor_names = []
    for n in neighbors_raw:
        if isinstance(n, str):
            neighbor_names.append(n)
        elif isinstance(n, dict) and 'name' in n:
            neighbor_names.append(n['name'])
        elif isinstance(n, dict) and 'generated_name' in n:
            neighbor_names.append(n['generated_name'])
    
    neighbors_str = ", ".join(neighbor_names) if neighbor_names else "no immediate named neighbors"

    return f'''
You are a creative historian, living in the year 2026. You are an expert on technical historic writing. You are contributing to a comprehensive World Atlas. Your task is to analyze a GeoJSON feature, representing a country, containing a `generated_name` and a list of `neighbors`, and use this data to write a short paragraph on the history of this country.

**Input Data:**
- **Feature Name**: {generated_name}
- **Nearby Neighbors**: {neighbors_str}

**Logic for Name Selection:**
1. Analyze all items in the `neighbors` list.
2. Come up with a plausible-sounding fictional history of the country, sometimes including details about its neighbors.
3. Write exclusively in the third person, as if writing an entry for an encyclopedia. Make no allusions to yourself as the author; be completely impartial.
4. Be as creative as you can. Feel free to make up details as you see fit.
5. Do not necessarily include every single neighbor in the list, but throw in a reference here or there.
6. Do not mention any countries outside of the neighbors list, but names of physical features or continents are acceptable.

**Output Requirements:**
- You must output a short paragraph, 5 or 6 sentences maximum.
- Do not include markdown formatting (like ```json), explanations, or any text outside of the aformentioned paragraph.
- 

**Task:**
Generate the historical summary uning the provided data under **Input Data**.
'''

def process_single_feature(args):
    """
    Worker function executed by each process.
    Returns a dictionary with index, id, result text, and error status.
    """
    index, feature = args
    feature_id = feature.get('properties', {}).get('id', f"Index_{index}")
    
    result = {
        "feature_index": index,
        "feature_id": feature_id,
        "content": None,
        "error": None
    }

    try:
        prompt_text = create_context_prompt(feature)
        response = send_prompt_to_elm(prompt_text)
        
        if not response:
            raise ValueError("Empty response from API")
            
        clean_response = response.strip()
        
        # Remove markdown code blocks if present
        if clean_response.startswith("```"):
            lines = clean_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_response = "\n".join(lines).strip()
        
        if len(clean_response) < 5:
            raise ValueError("Response too short")
            
        result["content"] = clean_response
        
    except Exception as e:
        result["error"] = str(e)
        result["content"] = f"ERROR: {str(e)}"
        
    return result

def process_and_save_geojson(input_path, output_path, num_workers=2):
    """
    Main orchestration function: Loads, Processes, Injects Results, and Saves.
    """
    # 1. Load Data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Critical Error loading file: {e}")
        return

    if 'features' not in data:
        print("Error: Invalid GeoJSON structure.")
        return

    features = data['features']
    
    # Prepare task list for multiprocessing
    task_list = list(enumerate(features))
    
    print(f"Starting processing of all features with {num_workers} workers...")
    
    # 2. Multiprocessing Execution
    results = []
    
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        print("Runtime Error!")
        pass 

    with mp.Pool(processes=num_workers) as pool:
        async_result = pool.map_async(process_single_feature, task_list)
        try:
            results = async_result.get()
        except Exception as e:
            print(f"Pool execution failed: {e}")
            return

    # 3. Inject Results into Original Features
    success_count = 0
    fail_count = 0
    
    for res in results:
        idx = res['feature_index']
        if idx >= len(features):
            continue
            
        feature = features[idx]
        
        # Ensure properties exists
        if 'properties' not in feature:
            feature['properties'] = {}
            
        # Ensure history list exists
        if 'history' not in feature['properties']:
            feature['properties']['history'] = ""
            
        # Append the result content to the history list
        # We append the text regardless of success/error to maintain a log
        feature['properties']['history'] = res['content']


        if res['error'] is None:
            success_count += 1
        else:
            fail_count += 1

    print(f"Processing complete. Success: {success_count}, Failures: {fail_count}")
    print(f"Saving updated GeoJSON to '{output_path}'...")

    # 4. Save to File
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(features)} features to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main(user_api_key):
    global API_KEY
    API_KEY = user_api_key
    # Configuration
    INPUT_FILE = "./temp/neighbors.geojson" 
    OUTPUT_FILE = "./temp/output.geojson"
    WORKERS = 4

    # Run the pipeline
    process_and_save_geojson(INPUT_FILE, OUTPUT_FILE, num_workers=WORKERS)
    
    print("Done.")
    
# # --- Main Execution Block ---
# if __name__ == "__main__":
#     main(user_api_key)

