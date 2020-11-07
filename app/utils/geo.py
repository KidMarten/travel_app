import googlemaps
import pandas as pd
API_KEY = 'AIzaSyA-yxpC9WMCvJsBgB4rWvIr-6nTJvhTCjE'
gmaps = googlemaps.Client(API_KEY)

def get_coords(names):
    hotels_coords = []

    for hotel in names:
        geocode_result = gmaps.geocode(hotel + ' Zanzibar')
        result = geocode_result[0]
        lat = result['geometry']['location']['lat']
        lon = result['geometry']['location']['lng']
        hotels_coords.append((hotel, result['formatted_address'], lat, lon))
    
    map_df = pd.DataFrame(hotels_coords)
    map_df.columns = ['name', 'address', 'lat', 'lon']
    return map_df