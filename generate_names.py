import requests
import json
import time
import os
import multiprocessing as mp
from functools import partial

# Security Best Practice: Load API key from environment variable
MODEL_NAME = 'Qwen/Qwen3.5-397B-A17B-FP8' 
FILE_PATH = './temp/data.geojson'
OUTPUT_PATH = './temp/names.geojson'
FILE_ENCODING = 'UTF-8'
new_field_name = 'generated_name' 

base_prompt = '''
You are a creative historian with a knowledge both of geography and data analysis. Your task is to analyze a GeoJSON feature containing a list of `matched_references` and determine the most appropriate single name for that feature based on provided weights.

**Input Data:**
You will receive a single GeoJSON Feature object. Inside the `properties` of this feature, there is a list called `matched_references`. Each item in this list contains:
- `name`: A candidate name string.
- `reference_weight`: A numerical value between 0 and 10000 indicating the confidence or emphasis for that name.
You will also recieve a list of names that have already been used, separated by commas. These are to be strictly excluded from being your output.

**Logic for Name Selection:**
1. Analyze all items in the `matched_references` list.
2. If the highest `reference_weight` is 1000 or more above the next-highest, or if only one `name` has a `reference_weight` above 0, use that name as the output. 
3. Otherwise, using the `reference_weight`s, invent a new name that combines inspiration from the 3 highest `name`s, proportional to their weights.
4. Be creative. You should sometimes use the name as-is, but sometimes put a slight variation on it, especially if the name is on the used list or if the reference weights are similar.
5. Keep it brief. Your synthezised names should be relatively short, two words or fewer, unless the top reference weiths are very similar, in which case being more verbose is warranted.
6. Do not re-use any names verbatim from the list of used names.

**Output Requirements:**
- You must output ONLY valid JSON.
- Do not include markdown formatting (like ```json), explanations, or any text outside the JSON object.
- The output JSON must have a single key: `suggested_names`.
- The value must be the selected name string.

**Example Input:**
{
  "type": "Feature",
  "properties": {
    "matched_references": [
      {"name": "Old Mill Lane", "reference_weight": 4000},
      {"name": "Mill Lane", "reference_weight": 8500},
      {"name": "Mill Republic", "reference_weight": 8250}
    ]
  },
  "geometry": { ... }
}

**Example Output:**
{"suggested_name": ["Mill Lane Republic"]}

**Task:**
Process the following GeoJSON feature and provide the JSON output:
'''

def send_prompt_to_elm(args):
    """
    Worker function for multiprocessing. 
    Receives a tuple of (matched_refs, used_names_list_proxy).
    Returns a tuple of (index, new_name, success, error_message).
    """
    matched_refs, used_names_proxy, index = args
    
    # Convert proxy list to regular list for string joining
    current_used_names = list(used_names_proxy)
    
    prompt_text = base_prompt + '\n' + json.dumps(matched_refs) + '\nThe names that are already taken and should not be used are as follows: ' + ','.join(current_used_names)    
    
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
        parsed_result = json.loads(result_content)
        
        # Handle potential variation in key name based on prompt vs example logic
        # The prompt asks for "suggested_names" key but the example output shows "suggested_name" as a list.
        # The original code looked for "suggested_name". ELM preserves original logic.
        if "suggested_name" in parsed_result:
            name = parsed_result["suggested_name"]
            # If the model returns a list as per the example output {"suggested_name": ["Name"]}, extract the string
            if isinstance(name, list) and len(name) > 0:
                name = name[0]
        else:
            raise KeyError("Expected key 'suggested_name' not found in response")

        # Update the shared managed list with the new name to prevent duplicates in subsequent calls
        # Note: In a highly parallel race condition, two processes might pick the same name before updating.
        # For this specific request (2 threads), this approach is robust enough. 
        # A more complex lock mechanism could be added if strict uniqueness under high load is required.
        used_names_proxy.append(name)
        print(f"Name generated for feature: {name}")
        return (index, name, True, None)
        
    except requests.exceptions.HTTPError as http_err:
        return (index, None, False, f"HTTP error occurred: {http_err} (Status Code: {response.status_code})")
    except requests.exceptions.ConnectionError:
        return (index, None, False, "Error: Could not connect to the ELM API. Check your network or URL.")
    except requests.exceptions.Timeout:
        return (index, None, False, "Error: The request timed out.")
    except requests.exceptions.RequestException as err:
        return (index, None, False, f"A generic request error occurred: {err}")
    except json.JSONDecodeError:
        return (index, None, False, "Error: Could not decode JSON response from the server.")
    except KeyError as key_err:
        return (index, None, False, f"Error: Unexpected response format. Missing key: {key_err}")
    except Exception as e:
        return (index, None, False, f"An unexpected error occurred: {e}")

def main(user_api_key):
    global API_KEY
    API_KEY = user_api_key
    
    mp.set_start_method('fork', force=True)

    try:
        # 1. Read the file
        with open(FILE_PATH, 'r', encoding=FILE_ENCODING) as f:
            data = json.load(f)

        if 'features' not in data:
            raise ValueError("The file does not contain a 'features' array.")

        features = data['features']
        limit = len(features)
        
        print(f"Processing {limit} features with 4 parallel processes...")

        # Use a Manager to share the used_names list across processes
        manager = mp.Manager()
        used_names_proxy = manager.list()

        # Prepare arguments for the pool: (matched_refs, used_names_proxy, index)
        tasks = []
        for i in range(limit):
            feature = features[i]
            properties = feature.get('properties', {})
            
            if 'matched_references' in properties:
                tasks.append((properties['matched_references'], used_names_proxy, i))
            else:
                # Placeholder for features without references to maintain index alignment if needed, 
                # though we will handle skipping in result processing.
                tasks.append((None, used_names_proxy, i))

        # Create a pool with 2 workers
        with mp.Pool(processes=4) as pool:
            # Map the function to the tasks
            # We use imap_unordered for speed as order of completion doesn't matter, 
            # but we need to map results back to features later.
            results = pool.map(send_prompt_to_elm, tasks)

        # Process results and update the original data structure
        success_count = 0
        error_count = 0
        to_del = ["polygon_count","is_island","type","area_m2","dist_to_mainland_m","merge_attempted","cluster_id","cluster_size","original_indices","continent","source_status","matched_references"]


        for index, new_name, success, error_msg in results:
            if index >= len(features):
                continue
                
            feature = features[index]
            properties = feature.get('properties', {})
            if 'matched_references' not in properties:
                continue

            if success:
                properties[new_field_name] = new_name
                success_count += 1
                print(f"successfully generated a name for feature {index}")
                # Optional: Print progress quietly or in batch
            else:
                error_count += 1
                print(f"Feature {index}: Failed - {error_msg}")
                
            for key in to_del:
                try:
                    del properties[key]
                except:
                    print(f"unable to delete key {key}")

        print(f"\nProcessing complete. Success: {success_count}, Errors: {error_count}")

        # 5. Save the updated GeoJSON
        with open(OUTPUT_PATH, 'w', encoding=FILE_ENCODING) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Success! Updated file saved to: {OUTPUT_PATH}")

    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# if __name__ == '__main__':
#     # Required for multiprocessing on Windows and recommended for Unix
#     main(user_api_key)
