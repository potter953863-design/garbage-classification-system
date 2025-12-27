"""
Map Service Module
Integrates Gaode (Amap) API and Folium for map display and navigation
"""

import requests
import folium
from folium import plugins
from typing import Dict, List, Optional, Tuple
import json
from config import GAODE_API_KEY, GAODE_API_BASE_URL, CATEGORY_COLORS, DEFAULT_LANGUAGE
from i18n import get_category_label, get_recycling_station_term


class MapService:
    """
    Map service for recycling point location and navigation.
    Integrates Gaode (Amap) API and Folium for map visualization.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize map service.
        
        Args:
            api_key: Gaode API key (default: from config)
        """
        self.api_key = api_key or GAODE_API_KEY
        self.api_available = self.api_key and self.api_key != 'your_gaode_api_key_here'
    
    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Convert address to coordinates using Gaode Geocoding API.
        
        Args:
            address: Address string
            
        Returns:
            Tuple of (latitude, longitude) or None if failed
        """
        if not self.api_available:
            return None
        
        try:
            url = f"{GAODE_API_BASE_URL}/geocode/geo"
            params = {
                'key': self.api_key,
                'address': address,
                'output': 'json'
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get('status') == '1' and data.get('geocodes'):
                location = data['geocodes'][0]['location']
                lon, lat = map(float, location.split(','))
                return lat, lon
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return None
    
    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[str]:
        """
        Convert coordinates to address using Gaode Reverse Geocoding API.
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Address string or None if failed
        """
        if not self.api_available:
            return None
        
        try:
            url = f"{GAODE_API_BASE_URL}/geocode/regeo"
            params = {
                'key': self.api_key,
                'location': f"{longitude},{latitude}",
                'output': 'json',
                'radius': 1000,
                'extensions': 'all'
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get('status') == '1' and data.get('regeocode'):
                formatted_address = data['regeocode'].get('formatted_address', '')
                return formatted_address
        except Exception as e:
            print(f"Reverse geocoding error: {e}")
        
        return None
    
    def get_route(self, 
                  start_lat: float, 
                  start_lon: float,
                  end_lat: float, 
                  end_lon: float,
                  strategy: str = '0') -> Optional[Dict]:
        """
        Get route planning from Gaode Direction API.
        
        Args:
            start_lat, start_lon: Start point coordinates
            end_lat, end_lon: End point coordinates
            strategy: Route strategy (0=fastest, 1=shortest, 2=avoid toll, etc.)
            
        Returns:
            Dictionary containing route information or None if failed
        """
        if not self.api_available:
            return None
        
        try:
            url = f"{GAODE_API_BASE_URL}/direction/driving"
            params = {
                'key': self.api_key,
                'origin': f"{start_lon},{start_lat}",
                'destination': f"{end_lon},{end_lat}",
                'strategy': strategy,
                'output': 'json'
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get('status') == '1' and data.get('route'):
                route_info = data['route']
                paths = route_info.get('paths', [])
                if paths:
                    path = paths[0]
                    distance = path.get('distance', 0)
                    duration = path.get('duration', 0)
                    try:
                        distance = float(distance)
                    except (TypeError, ValueError):
                        distance = 0.0
                    try:
                        duration = float(duration)
                    except (TypeError, ValueError):
                        duration = 0.0
                    return {
                        'distance': distance,  # meters
                        'duration': duration,  # seconds
                        'steps': path.get('steps', [])
                    }
        except Exception as e:
            print(f"Route planning error: {e}")
        
        return None
    
    def create_map(self, 
                   center_lat: float, 
                   center_lon: float,
                   zoom_start: int = 13,
                   use_gaode: bool = False) -> folium.Map:
        """
        Create a Folium map centered at specified coordinates.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            zoom_start: Initial zoom level
            use_gaode: Whether to use Gaode tiles (requires API key and special setup)
            
        Returns:
            Folium Map object
        """
        # For now, use OpenStreetMap as base (free, no API key needed)
        # Gaode tiles integration requires additional setup
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add tile layer options
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def add_markers(self, 
                   map_obj: folium.Map,
                   points: List[Dict],
                   user_location: Tuple[float, float] = None,
                   language: str = DEFAULT_LANGUAGE):
        """
        Add markers to the map for recycling points.
        
        Args:
            map_obj: Folium Map object
            points: List of point dictionaries with 'latitude', 'longitude', 'name', 'category', etc.
            user_location: User's location tuple (lat, lon) for adding user marker
        """
        # Add user location marker if provided
        if user_location:
            folium.Marker(
                location=user_location,
                popup='我的位置',
                tooltip='我的位置',
                icon=folium.Icon(color='blue', icon='user', prefix='fa')
            ).add_to(map_obj)
        
        # Add recycling point markers
        for point in points:
            category = point.get('category', 'Other')
            color = CATEGORY_COLORS.get(category, '#616161')
            label_localized = get_category_label(category, language)
            
            # Create popup content
            popup_html = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4 style="margin: 5px 0; color: {color};">{point.get('name', '回收点')}</h4>
                <p style="margin: 3px 0;"><strong>类别:</strong> {label_localized}</p>
                <p style="margin: 3px 0;"><strong>地址:</strong> {point.get('address', '未知')}</p>
            """
            
            if point.get('distance_km'):
                popup_html += f"<p style='margin: 3px 0;'><strong>距离:</strong> {point['distance_km']} 公里</p>"
            
            if point.get('phone'):
                popup_html += f"<p style='margin: 3px 0;'><strong>电话:</strong> {point['phone']}</p>"
            
            if point.get('opening_hours'):
                popup_html += f"<p style='margin: 3px 0;'><strong>营业时间:</strong> {point['opening_hours']}</p>"
            
            if point.get('description'):
                popup_html += f"<p style='margin: 3px 0;'><strong>说明:</strong> {point['description']}</p>"
            
            popup_html += "</div>"
            
            # Create icon based on category
            icon_color = color.replace('#', '')
            icon_name = {
                'Recyclable': 'recycle',
                'Hazardous': 'exclamation-triangle',
                'Kitchen': 'leaf',
                'Other': 'trash'
            }.get(category, 'map-marker')
            
            folium.Marker(
                location=[point['latitude'], point['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{point.get('name', '回收点')} ({label_localized})",
                icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
            ).add_to(map_obj)
    
    def add_route(self, 
                 map_obj: folium.Map,
                 start_lat: float,
                 start_lon: float,
                 end_lat: float,
                 end_lon: float,
                 route_info: Dict = None):
        """
        Add route line to the map.
        
        Args:
            map_obj: Folium Map object
            start_lat, start_lon: Start point coordinates
            end_lat, end_lon: End point coordinates
            route_info: Route information from API (optional, for detailed route)
        """
        if route_info and route_info.get('steps'):
            # Use detailed route from API
            points = []
            for step in route_info['steps']:
                polyline = step.get('polyline', '')
                # Decode polyline (simplified - Gaode uses custom encoding)
                # For now, use straight line
                pass
            
            # Fallback to straight line
            folium.PolyLine(
                locations=[[start_lat, start_lon], [end_lat, end_lon]],
                color='blue',
                weight=4,
                opacity=0.7,
                popup=f"距离: {route_info.get('distance', 0)/1000:.2f} 公里"
            ).add_to(map_obj)
        else:
            # Simple straight line
            folium.PolyLine(
                locations=[[start_lat, start_lon], [end_lat, end_lon]],
                color='blue',
                weight=4,
                opacity=0.7
            ).add_to(map_obj)
    
    def generate_navigation_url(self, 
                               start_lat: float,
                               start_lon: float,
                               end_lat: float,
                               end_lon: float,
                               start_name: str = '我的位置',
                               end_name: str = '目的地',
                               mode: str = 'car',
                               policy: int = 1) -> str:
        """
        Generate Gaode map navigation URL.
        
        Args:
            start_lat, start_lon: Start point coordinates
            end_lat, end_lon: End point coordinates
            end_name: Destination name
            
        Returns:
            Navigation URL string
        """
        from urllib.parse import quote
        
        safe_start_name = quote(start_name or '我的位置')
        safe_end_name = quote(end_name or '目的地')
        
        from_param = f"{start_lon},{start_lat},{safe_start_name}"
        to_param = f"{end_lon},{end_lat},{safe_end_name}"
        
        url = (
            "https://www.amap.com/direction"
            f"?from={from_param}"
            f"&to={to_param}"
            f"&mode={mode}"
            f"&policy={policy}"
        )
        return url
    
    def generate_mobile_navigation_url(self,
                                     end_lat: float,
                                     end_lon: float,
                                     end_name: str = '目的地') -> str:
        """
        Generate Gaode mobile app navigation URL.
        
        Args:
            end_lat, end_lon: Destination coordinates
            end_name: Destination name
            
        Returns:
            Mobile navigation URL string
        """
        # Gaode mobile app URL format
        url = f"androidamap://navi?sourceApplication=垃圾分类助手&lat={end_lat}&lon={end_lon}&dev=0&style=2"
        return url
    
    def generate_poi_search_url(self,
                                category: str,
                                language: str = DEFAULT_LANGUAGE,
                                latitude: float = None,
                                longitude: float = None,
                                address_text: str = None) -> str:
        """
        Generate Gaode web search URL for nearby recycling POIs.
        
        Args:
            category: Waste category (Hazardous, Recyclable, Kitchen, Other)
            language: UI language code for query text
            latitude, longitude: Optional center coordinates
            address_text: Optional textual address label for query fallback
            
        Returns:
            Gaode web search URL string
        """
        from urllib.parse import quote
        
        category_label = get_category_label(category, language)
        station_term = get_recycling_station_term(language)
        
        cleaned_address = (address_text or "").strip()
        query_parts = []
        if cleaned_address:
            query_parts.append(cleaned_address)
        query_parts.append(f"{category_label} {station_term}")
        query_text = " ".join(query_parts).strip()
        
        q = quote(query_text)
        
        if latitude is not None and longitude is not None:
            center = f"{longitude},{latitude}"
            return f"https://www.amap.com/search?query={q}&center={center}&zoom=14"
        
        # Fallback: pure text search (include address if available)
        return f"https://www.amap.com/search?query={q}"
    
    def search_recycling_pois(self,
                              latitude: float,
                              longitude: float,
                              category: str,
                              radius: int = 5000,
                              keywords: str = None) -> List[Dict]:
        """
        Search for recycling POIs using Gaode Place API.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            category: Waste category (Hazardous, Recyclable, Kitchen, Other)
            radius: Search radius in meters (default: 5000 = 5km)
            keywords: Additional keywords for search (optional)
            
        Returns:
            List of POI dictionaries with location and details
        """
        if not self.api_available:
            return []
        
        # Map category to search keywords
        category_keywords = {
            'Hazardous': '有害垃圾回收站|电池回收|荧光灯管回收|有害废物处理',
            'Recyclable': '可回收物回收点|废品回收站|资源回收|再生资源',
            'Kitchen': '厨余垃圾处理站|湿垃圾处理|有机垃圾处理',
            'Other': '垃圾处理站|垃圾转运站|垃圾收集点'
        }
        
        # Use provided keywords or default category keywords
        search_keywords = keywords or category_keywords.get(category, '垃圾回收')
        
        try:
            # Use around search API for better results
            url = f"{GAODE_API_BASE_URL}/place/around"
            params = {
                'key': self.api_key,
                'location': f"{longitude},{latitude}",
                'keywords': search_keywords,
                'types': '190000',  # Environmental facilities
                'radius': radius,
                'offset': 20,
                'page': 1,
                'extensions': 'all',
                'output': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            pois = []
            if data.get('status') == '1' and data.get('pois'):
                for poi in data['pois']:
                    location_str = poi.get('location', '')
                    if location_str:
                        lon, lat = map(float, location_str.split(','))
                        poi_dict = {
                            'name': poi.get('name', '未知'),
                            'address': poi.get('address', poi.get('pname', '') + poi.get('cityname', '') + poi.get('adname', '')),
                            'latitude': lat,
                            'longitude': lon,
                            'category': category,
                            'distance': int(poi.get('distance', 0)),  # meters
                            'distance_km': round(int(poi.get('distance', 0)) / 1000, 2),  # kilometers
                            'phone': poi.get('tel', ''),
                            'type': poi.get('type', ''),
                            'typecode': poi.get('typecode', ''),
                            'source': 'gaode_api'  # Mark as from API
                        }
                        pois.append(poi_dict)
            
            return pois
            
        except Exception as e:
            print(f"POI search error: {e}")
            return []
    
    def search_nearby_pois(self,
                          latitude: float,
                          longitude: float,
                          category: str,
                          radius_km: float = 5.0) -> List[Dict]:
        """
        Search for nearby recycling POIs with multiple radius attempts.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            category: Waste category
            radius_km: Initial search radius in kilometers
            
        Returns:
            List of POI dictionaries, sorted by distance
        """
        # Try multiple search radii: 5km -> 10km -> 20km
        search_radii = [
            int(radius_km * 1000),  # Convert to meters
            int(10 * 1000),
            int(20 * 1000)
        ]
        
        all_pois = []
        for radius_m in search_radii:
            pois = self.search_recycling_pois(
                latitude=latitude,
                longitude=longitude,
                category=category,
                radius=radius_m
            )
            
            if pois:
                all_pois.extend(pois)
                # If we found results in smaller radius, we can stop
                if radius_m <= int(radius_km * 1000):
                    break
        
        # Remove duplicates based on name and location
        seen = set()
        unique_pois = []
        for poi in all_pois:
            key = (poi['name'], round(poi['latitude'], 4), round(poi['longitude'], 4))
            if key not in seen:
                seen.add(key)
                unique_pois.append(poi)
        
        # Sort by distance
        unique_pois.sort(key=lambda x: x.get('distance', float('inf')))
        
        return unique_pois[:20]  # Limit to 20 results

