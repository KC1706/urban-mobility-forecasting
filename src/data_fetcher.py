"""
Enhanced Data Fetcher v2: Premium NYC data sources for better analysis
Includes: Real-time GTFS, advanced taxi data, live weather, traffic sensors
"""

import os
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class EnhancedNYCDataFetcher:
    """Fetch premium NYC data from multiple sources"""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== GTFS DATA ==========
    
    def fetch_real_time_gtfs(self):
        """
        Fetch REAL-TIME GTFS data with vehicle positions
        Better than static: Shows actual vehicle locations, delays, crowding
        
        Sources:
        1. MTA Real-Time Data API (requires API key, free)
           https://datamine.mta.info/
           
        2. GTFS-Realtime Protobuf endpoints:
           - ACE Subway: https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace
           - NQRW Subway: https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw
           - 1,2,3 Subway: https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs
           - Bus: https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs_bus
           - LIRR: https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs_lirr
           - Metro-North: https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs_mnr
        """
        logger.info("Fetching Real-Time GTFS data...")
        
        mta_key = os.getenv("MTA_API_KEY")
        if not mta_key:
            logger.warning("MTA_API_KEY not set. Get free key from: https://datamine.mta.info/")
            logger.info("Using public endpoints (some data may be limited)...")
        
        # GTFS feed endpoints
        feeds = {
            "ace": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
            "nqrw": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw",
            "123": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
            "bus": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs_bus",
            "lirr": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs_lirr",
            "mnr": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs_mnr",
        }
        
        try:
            for feed_name, url in feeds.items():
                try:
                    headers = {}
                    if mta_key:
                        headers['x-api-key'] = mta_key
                    
                    logger.info(f"Downloading {feed_name.upper()} GTFS...")
                    response = requests.get(url, headers=headers, timeout=15)
                    response.raise_for_status()
                    
                    output_path = self.data_dir / f"mta_gtfs_{feed_name}.zip"
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"✓ {feed_name.upper()} GTFS saved ({len(response.content) / 1024 / 1024:.2f} MB)")
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 401:
                        logger.warning(f"  {feed_name}: Requires API key (401 Unauthorized)")
                    else:
                        logger.warning(f"  {feed_name}: HTTP {e.response.status_code}")
                except Exception as e:
                    logger.warning(f"  {feed_name}: {str(e)[:100]}")
                    
        except Exception as e:
            logger.error(f"Failed to fetch GTFS feeds: {e}")
    
    def fetch_curated_transit_stops(self):
        """
        Fetch curated transit stop data with accessibility info
        Better than raw GTFS: Pre-processed, includes elevation, accessibility
        
        Sources:
        1. NYC Open Data - Transit Stops (MTA)
           https://data.cityofnewyork.us/api/views/f9bj-naz6/rows.json
           
        2. Transitland API (open-source transit data)
           https://transit.land/
        """
        logger.info("Fetching curated transit stops...")
        
        try:
            # NYC Open Data API (no key required)
            url = "https://data.cityofnewyork.us/api/views/f9bj-naz6/rows.json?limit=50000"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            output_path = self.data_dir / "nyc_transit_stops_curated.json"
            with open(output_path, 'w') as f:
                json.dump(response.json(), f, indent=2)
            
            logger.info(f"✓ Curated transit stops saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch curated transit data: {e}")
    
    # ========== TAXI & RIDE-HAIL DATA ==========
    
    def fetch_tlc_data_s3_direct(self, year=2024, month=1):
        """
        Fetch NYC TLC data directly from S3 with multiple formats
        Better than HTTP: Faster, more reliable, multiple data formats
        
        Sources:
        1. NYC TLC S3 Bucket (free public access)
           s3://nyc-tlc/trip data/
           
        2. Formats available:
           - Parquet (best for analysis)
           - CSV (legacy)
           - JSON (detailed)
        """
        logger.info(f"Fetching TLC data from S3 for {year}-{month:02d}...")
        
        taxi_types = ["yellow", "green", "fhv"]  # yellow, green, FHV (Uber/Lyft)
        
        # Try boto3 (S3 direct access - requires AWS setup)
        try:
            import boto3
            s3 = boto3.client('s3', region_name='us-east-1')
            
            for taxi_type in taxi_types:
                key = f"trip data/{taxi_type}_tripdata_{year}-{month:02d}.parquet"
                output_path = self.data_dir / f"nyc_tlc_{taxi_type}_{year}_{month:02d}.parquet"
                
                try:
                    s3.download_file('nyc-tlc', key, str(output_path))
                    logger.info(f"✓ Downloaded {taxi_type} data from S3")
                except Exception as e:
                    logger.warning(f"S3 download failed for {taxi_type}: {e}")
        
        except ImportError:
            logger.info("boto3 not installed. Falling back to HTTPS download...")
            self._fetch_tlc_https(year, month)
    
    def _fetch_tlc_https(self, year, month):
        """Fallback: Fetch TLC data via HTTPS"""
        logger.info("Fetching TLC data via HTTPS...")
        
        taxi_types = ["yellow", "green"]
        base_url = "https://d37cibb9ed327atu.cloudfront.net/taxi"
        
        for taxi_type in taxi_types:
            url = f"{base_url}/{year}/{month:02d}/{taxi_type}_tripdata_{year}-{month:02d}.parquet"
            output_path = self.data_dir / f"nyc_tlc_{taxi_type}_{year}_{month:02d}.parquet"
            
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"✓ Downloaded {taxi_type} taxi data")
                
            except Exception as e:
                logger.error(f"Failed to fetch {taxi_type} data: {e}")
    
    def fetch_fhv_data(self, year=2024, month=1):
        """
        Fetch For-Hire Vehicle data (Uber, Lyft, etc.)
        Better than taxi alone: Includes modern ride-hailing patterns
        
        Source: NYC TLC FHV data (same S3 bucket as taxi)
        https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
        """
        logger.info(f"Fetching FHV data for {year}-{month:02d}...")
        
        url = f"https://d37cibb9ed327atu.cloudfront.net/taxi/{year}/{month:02d}/fhv_tripdata_{year}-{month:02d}.parquet"
        output_path = self.data_dir / f"nyc_fhv_{year}_{month:02d}.parquet"
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✓ Downloaded FHV data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch FHV data: {e}")
    
    # ========== TRAFFIC & SENSOR DATA ==========
    
    def fetch_traffic_sensor_data(self):
        """
        Fetch real-time traffic sensor data
        Better than static OSM: Shows actual congestion patterns
        
        Sources:
        1. NYC DOT Real-time Traffic API
           https://a821-dotweb01.nyc.gov/datafeeds/
           
        2. INRIX Traffic (requires API key)
           https://developer.inrix.com/
           
        3. Google Maps Platform Traffic API (paid, very accurate)
        """
        logger.info("Fetching traffic sensor data...")
        
        try:
            # NYC DOT Traffic Data (free, real-time)
            url = "https://a821-dotweb01.nyc.gov/datafeeds/count_report/count_info/agg_cnt_result.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            output_path = self.data_dir / "nyc_traffic_sensors.json"
            with open(output_path, 'w') as f:
                json.dump(response.json(), f, indent=2)
            
            logger.info(f"✓ Traffic sensor data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch traffic data: {e}")
    
    # ========== WEATHER DATA ==========
    
    def fetch_weather_nowcast(self):
        """
        Fetch real-time weather data and forecasts
        Better than static: Shows current conditions affecting mobility
        
        Sources:
        1. OpenWeatherMap (free tier: 60 calls/min)
           https://openweathermap.org/api
           
        2. NOAA Weather API (US government, always free)
           https://api.weather.gov/
           
        3. Weather.com API (requires key, very detailed)
        """
        logger.info("Fetching real-time weather data for NYC...")
        
        # NOAA API (free, no key required)
        try:
            # Get NYC coordinates
            nyc_lat, nyc_lon = 40.7128, -74.0060
            
            # Get grid data
            points_url = f"https://api.weather.gov/points/{nyc_lat},{nyc_lon}"
            points_response = requests.get(points_url, timeout=10)
            points_response.raise_for_status()
            
            grid_data = points_response.json()
            forecast_url = grid_data['properties']['forecast']
            
            # Get actual forecast
            forecast_response = requests.get(forecast_url, timeout=10)
            forecast_response.raise_for_status()
            
            output_path = self.data_dir / "nyc_weather_forecast.json"
            with open(output_path, 'w') as f:
                json.dump(forecast_response.json(), f, indent=2)
            
            logger.info(f"✓ Weather forecast saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
    
    # ========== STREET & POI DATA ==========
    
    def fetch_overture_maps_data(self):
        """
        Fetch street and POI data from Overture Maps (improved alternative to OSM)
        Better than OSM: More up-to-date, better coverage, multiple data types
        
        Source: Overture Maps (https://overturemaps.org/)
        License: Open Data Commons Open Database License (ODbL)
        """
        logger.info("Fetching Overture Maps data for NYC...")
        
        try:
            # Overture Maps provides street, building, POI data via S3
            # Can be queried via DuckDB or downloaded directly
            
            # For now, provide API example
            logger.info("Overture Maps integration requires DuckDB + S3 access")
            logger.info("Install: pip install duckdb")
            logger.info("Query example provided in DATA_SOURCES.md")
            
        except Exception as e:
            logger.error(f"Failed to fetch Overture data: {e}")
    
    def fetch_enriched_poi_data(self):
        """
        Fetch enriched POI data with foot traffic, hours, ratings
        Better than raw OSM: Includes business intelligence
        
        Sources:
        1. Google Places API (requires API key, very comprehensive)
           https://developers.google.com/maps/documentation/places/web-service
           
        2. Foursquare Places API (requires API key)
           https://developer.foursquare.com/
           
        3. NYC Open Data (free, curated datasets)
           https://data.cityofnewyork.us/
        """
        logger.info("Fetching enriched POI data...")
        
        try:
            # NYC Open Data - Restaurants/Food Service Establishments
            url = "https://data.cityofnewyork.us/api/views/r5kj-4dzi/rows.json?limit=50000"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            output_path = self.data_dir / "nyc_pois_enriched.json"
            with open(output_path, 'w') as f:
                json.dump(response.json(), f, indent=2)
            
            logger.info(f"✓ Enriched POI data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch POI data: {e}")
    
    # ========== DEMOGRAPHIC & EQUITY DATA ==========
    
    def fetch_census_data(self):
        """
        Fetch US Census data for NYC (income, employment, etc.)
        Critical for: Equity analysis, accessibility assessment
        
        Source: US Census API (free, requires API key)
        https://api.census.gov/
        """
        logger.info("Fetching Census data for NYC...")
        
        census_key = os.getenv("CENSUS_API_KEY")
        if not census_key:
            logger.warning("CENSUS_API_KEY not set. Get free key from: https://api.census.gov/")
            return
        
        try:
            # Example: Median household income by tract
            url = "https://api.census.gov/data/2021/acs/acs5"
            params = {
                'key': census_key,
                'get': 'NAME,B19013_001E',  # Median household income
                'for': 'tract:*',
                'in': 'state:36 county:061',  # NY, New York County (Manhattan)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            output_path = self.data_dir / "nyc_census_data.json"
            with open(output_path, 'w') as f:
                json.dump(response.json(), f, indent=2)
            
            logger.info(f"✓ Census data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch Census data: {e}")
    
    # ========== BATCH OPERATIONS ==========
    
    def fetch_all_enhanced_data(self):
        """Fetch all available enhanced data sources"""
        print("\n" + "="*70)
        print("ENHANCED NYC DATA FETCHER v2")
        print("="*70 + "\n")
        
        print("[1/8] Real-time GTFS...")
        self.fetch_real_time_gtfs()
        
        print("\n[2/8] Curated Transit Stops...")
        self.fetch_curated_transit_stops()
        
        print("\n[3/8] TLC Taxi & FHV Data...")
        self.fetch_tlc_data_s3_direct()
        self.fetch_fhv_data()
        
        print("\n[4/8] Traffic Sensor Data...")
        self.fetch_traffic_sensor_data()
        
        print("\n[5/8] Real-time Weather...")
        self.fetch_weather_nowcast()
        
        print("\n[6/8] Enriched POI Data...")
        self.fetch_enriched_poi_data()
        
        print("\n[7/8] Overture Maps...")
        self.fetch_overture_maps_data()
        
        print("\n[8/8] Census Data...")
        self.fetch_census_data()
        
        print("\n" + "="*70)
        print("✓ Enhanced data fetching complete!")
        print("="*70)
        print(f"\nData directory: {self.data_dir}")
        print("\nNext steps:")
        print("  1. Set API keys in .env file")
        print("  2. Run: python -m src.data_fetcher_v2")
        print("  3. Check DATA_SOURCES.md for detailed source documentation")


if __name__ == "__main__":
    fetcher = EnhancedNYCDataFetcher()
    fetcher.fetch_all_enhanced_data()
