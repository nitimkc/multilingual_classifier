"""
###
# Places are specific, named locations with corresponding geo coordinates that users decide to assign
# The place object is always present when a Tweet is geo-tagged, 
#   The place object can also contain a bounding box of coordinates which encloses this place
# The tweet with exact location will have geo object to it (16293 example in geo.jsonl)
#    with corresnponding type and coordinates (in [long, lat] format)
#    Geo object may have a place id attached to it without coordinates
# Having geo can trigger coordinates object (with type and coordinates) in old tweets
#   The coordinates object (Not Applicable) is only present (non-null) when the Tweet is assigned an exact location 

"""
import time
from random import randint

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

class TweetGeoProcessor:
    """
    A geo processor class to enable processing of geo information from tweets
    """

    def __init__(self, locator=Nominatim(user_agent="geoapiExercises"), wait=1):
        self.locator = locator
        self.wait = wait

    def geocode_location(self, geotweet):
        """
        If the tweet contains location information, convert to geo cordinates
        """
        location = geotweet.get("location", "")
        if location is not None:
            try:
                geocoded = self.locator.geocode(location, exactly_one=True)
                if geocoded is None:
                    # clean up location
                    geocoded = self.locator.geocode(location, exactly_one=True) #fail
                else:
                    return geocoded # sucess
            except GeocoderTimedOut as error:
                print(f"TIMED OUT: GeocoderTimedOut: Retrying...{error}")
                time.sleep(randint(1*100, self.wait*100)/100)
                return self.geocode_location(geotweet)
            except GeocoderServiceError as serv_err:
                print(f"CONNECTION REFUSED: GeocoderServiceError encountered. {serv_err}")
                return None


# def geo_converter(lonlat, geolocator, wait):
#     try:
#         if isinstance(lonlat, list):
#             geo = geolocator.geocode(f"{lonlat[-1]}, {lonlat[0]}",  language="en")
#             # geo = RateLimiter(geo, min_delay_seconds=1)
#         else:
#             geo = lonlat
#         return geo            
#     except GeocoderTimedOut as error:
#         print(f"TIMED OUT: GeocoderTimedOut: Retrying...{error}")
#         time.sleep(randint(1*100,wait*100)/100)
#         return geo_converter(lonlat, geolocator, wait)
#     except GeocoderServiceError as e:
#         print(f"CONNECTION REFUSED: GeocoderServiceError encountered. {e}")
#         return None
#     except Exception as e:
#         print(f"ERROR: Terminating due to exception {e}")
#         return None