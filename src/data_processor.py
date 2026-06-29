"""
Data Processor: Clean, aggregate, and prepare data for LLM analysis
"""

import os
import json
from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
from datetime import datetime
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class MobilityDataProcessor:
    """Process GTFS, taxi, and OSM data for analysis"""
    
    def __init__(self, raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
    
    def process_gtfs(self):
        """Extract and summarize GTFS data"""
        logger.info("Processing GTFS data...")
        
        gtfs_summaries = {}
        
        for feed_type in ["subway", "bus", "railroad"]:
            gtfs_dir = self.raw_dir / f"nyc_mta_{feed_type}_gtfs"
            
            if not gtfs_dir.exists():
                logger.warning(f"GTFS {feed_type} directory not found")
                continue
            
            try:
                # Read GTFS files
                stops = pd.read_csv(gtfs_dir / "stops.txt")
                routes = pd.read_csv(gtfs_dir / "routes.txt")
                stop_times = pd.read_csv(gtfs_dir / "stop_times.txt")
                trips = pd.read_csv(gtfs_dir / "trips.txt")
                
                # Basic statistics
                summary = {
                    "feed_type": feed_type,
                    "num_stops": len(stops),
                    "num_routes": len(routes),
                    "num_trips": len(trips),
                    "num_stop_times": len(stop_times),
                    "avg_stops_per_route": len(stop_times) / len(routes) if len(routes) > 0 else 0,
                    "routes": routes[['route_id', 'route_short_name', 'route_long_name', 'route_type']].to_dict('records')[:10]
                }
                
                gtfs_summaries[feed_type] = summary
                logger.info(f"✓ Processed {feed_type}: {summary['num_stops']} stops, {summary['num_routes']} routes")
                
                # Save processed GTFS
                stops.to_csv(self.processed_dir / f"stops_{feed_type}.csv", index=False)
                routes.to_csv(self.processed_dir / f"routes_{feed_type}.csv", index=False)
                
            except Exception as e:
                logger.error(f"Failed to process {feed_type} GTFS: {e}")
        
        # Save summary
        with open(self.processed_dir / "gtfs_summary.json", "w") as f:
            json.dump(gtfs_summaries, f, indent=2)
        
        return gtfs_summaries
    
    def process_taxi_data(self):
        """Process NYC TLC taxi trip records"""
        logger.info("Processing taxi trip data...")
        
        taxi_summary = {}
        
        for taxi_type in ["yellow", "green"]:
            parquet_file = self.raw_dir / f"nyc_tlc_{taxi_type}_tripdata_2024_01.parquet"
            
            if not parquet_file.exists():
                logger.warning(f"Taxi data {parquet_file} not found")
                continue
            
            try:
                df = pd.read_parquet(parquet_file)
                
                # Data cleaning
                df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
                
                # Trip duration
                df['trip_duration_min'] = (
                    pd.to_datetime(df['tpep_dropoff_datetime']) - 
                    pd.to_datetime(df['tpep_pickup_datetime'])
                ).dt.total_seconds() / 60
                
                # Distance-based metrics
                if 'trip_distance' in df.columns:
                    df['speed_mph'] = df['trip_distance'] / (df['trip_duration_min'] / 60 + 0.01)
                
                summary = {
                    "taxi_type": taxi_type,
                    "total_trips": len(df),
                    "avg_trip_duration_min": df['trip_duration_min'].mean(),
                    "avg_trip_distance": df['trip_distance'].mean() if 'trip_distance' in df.columns else 0,
                    "avg_fare": df['fare_amount'].mean() if 'fare_amount' in df.columns else 0,
                    "peak_hour": df['tpep_pickup_datetime'].dt.hour.mode()[0] if len(df) > 0 else 0
                }
                
                taxi_summary[taxi_type] = summary
                logger.info(f"✓ Processed {taxi_type} taxi: {summary['total_trips']} trips")
                
                # Save sample and summary
                df.sample(min(10000, len(df))).to_csv(
                    self.processed_dir / f"taxi_sample_{taxi_type}.csv", index=False
                )
                
            except Exception as e:
                logger.error(f"Failed to process {taxi_type} taxi data: {e}")
        
        with open(self.processed_dir / "taxi_summary.json", "w") as f:
            json.dump(taxi_summary, f, indent=2)
        
        return taxi_summary
    
    def process_osm_data(self):
        """Process OSM road network"""
        logger.info("Processing OSM road network...")
        
        try:
            import osmnx as ox
            
            osm_file = self.raw_dir / "nyc_osm_road_network.graphml"
            
            if not osm_file.exists():
                logger.warning("OSM file not found")
                return {}
            
            G = ox.load_graphml(osm_file)
            
            summary = {
                "num_nodes": len(G.nodes()),
                "num_edges": len(G.edges()),
                "avg_node_degree": sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0,
            }
            
            logger.info(f"✓ Processed OSM: {summary['num_nodes']} nodes, {summary['num_edges']} edges")
            
            with open(self.processed_dir / "osm_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to process OSM data: {e}")
            return {}
    
    def create_analysis_dataset(self):
        """Combine processed data into unified analysis dataset"""
        logger.info("Creating unified analysis dataset...")
        
        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "city": "New York City",
            "data_sources": {}
        }
        
        # Load all summaries
        for summary_file in self.processed_dir.glob("*_summary.json"):
            with open(summary_file) as f:
                data_type = summary_file.stem.replace("_summary", "")
                analysis_data["data_sources"][data_type] = json.load(f)
        
        # Save combined dataset
        output_file = self.processed_dir / "analysis_dataset.json"
        with open(output_file, "w") as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"✓ Analysis dataset created: {output_file}")
        
        return analysis_data
    
    def generate_data_report(self):
        """Generate summary report of processed data"""
        logger.info("Generating data report...")
        
        report_path = self.processed_dir / "data_report.md"
        
        report_content = f"""
# NYC Urban Mobility Data Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Summary

### GTFS Transit Data
- **Subway**: NYC MTA subway system (A-Z lines)
- **Bus**: NYC MTA bus routes
- **Railroad**: Long Island Rail Road (LIRR), Metro-North

### Taxi Trip Records
- **Yellow Taxis**: Yellow cab medallion data
- **Green Taxis**: Green boro taxi data
- **Period**: January 2024
- **Format**: Trip origins/destinations, times, fares

### Road Network (OpenStreetMap)
- **Coverage**: NYC 5 boroughs
- **Data**: Road segments, intersections, attributes

## Data Quality

### Processed Files
"""
        
        for f in sorted(self.processed_dir.glob("*.csv")):
            size_kb = f.stat().st_size / 1024
            report_content += f"\n- {f.name} ({size_kb:.2f} KB)"
        
        report_content += f"""

## Next Steps

1. **Exploratory Analysis**: Run notebooks/01_data_exploration.ipynb
2. **LLM Analysis**: Use src/llm_analyzer.py for mobility insights
3. **Evaluation**: src/evaluator.py for quantitative assessment

## Data Access Notes

- All data is publicly available
- GTFS: Updated regularly by MTA
- Taxi data: Monthly updates from NYC TLC
- OSM: Continuously updated by community

"""
        
        with open(report_path, "w") as f:
            f.write(report_content)
        
        logger.info(f"✓ Data report saved: {report_path}")


def main():
    """Process all downloaded data"""
    processor = MobilityDataProcessor()
    
    print("\n" + "="*60)
    print("NYC Mobility Data Processor")
    print("="*60 + "\n")
    
    print("[1/4] Processing GTFS...")
    processor.process_gtfs()
    
    print("\n[2/4] Processing Taxi Data...")
    processor.process_taxi_data()
    
    print("\n[3/4] Processing OSM...")
    processor.process_osm_data()
    
    print("\n[4/4] Creating Analysis Dataset...")
    processor.create_analysis_dataset()
    
    print("\nGenerating Report...")
    processor.generate_data_report()
    
    print("\n" + "="*60)
    print(f"✓ Data processing complete!")
    print(f"  Processed data: {PROCESSED_DATA_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
