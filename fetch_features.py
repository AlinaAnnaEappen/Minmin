import requests

def get_safety_features(lat, lon):
    # We changed the range from 1000 to 5000 (which is 5km)
    query = f"""
    [out:json];
    (
      node["amenity"="police"](around:5000, {lat}, {lon});
      way["amenity"="police"](around:5000, {lat}, {lon});
      relation["amenity"="police"](around:5000, {lat}, {lon});
    );
    out count;
    """
    
    url = "https://overpass-api.de/api/interpreter"
    
    try:
        response = requests.get(url, params={'data': query})
        data = response.json()
        
        # This extracts the number from the map response
        police_count = data.get('elements', [{}])[0].get('tags', {}).get('total', 0)
        return int(police_count)
    except Exception as e:
        print(f"Error connecting to Map: {e}")
        return 0

# --- TESTING IT ---
# Using your specific coordinates:
my_lat = 9.3182808  
my_lon = 76.6085318

print(f"Checking for police stations within 5km of your location...")
count = get_safety_features(my_lat, my_lon)
print(f"Found {count} police stations within 5km!")