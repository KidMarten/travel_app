import googlemaps
import pandas as pd
from multiprocessing import Pool

API_KEY = 'AIzaSyA-yxpC9WMCvJsBgB4rWvIr-6nTJvhTCjE'
gmaps = googlemaps.Client(API_KEY)


def collect_hotel_coords(name):
    try:

        # Use hotel name and Zanzibar keyword to increase precision
        geocode_result = gmaps.geocode(name + ' Zanzibar')
        
        # Fetch first result
        result = geocode_result[0]

        # Get hotel coordinates
        lat = result['geometry']['location']['lat']
        lon = result['geometry']['location']['lng']
        return (name, result['formatted_address'], lat, lon)

    except:

        return (None, None, None, None)


def get_coords(names):
    # Speed up search by sending many requests at the same time
    with Pool(len(names)) as p:
        hotels_coords = p.map(collect_hotel_coords, names)
    
    # Convert to pandas for st.map
    map_df = pd.DataFrame(hotels_coords)
    map_df.columns = ['name', 'address', 'lat', 'lon']
    map_df.dropna(inplace=True)
    return map_df