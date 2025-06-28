#!/usr/bin/env python3
"""
Meet Data Integrator
Download and integrate OpenPowerlifting competition data with training data
"""

import pandas as pd
import requests
import zipfile
import os
from pathlib import Path
from datetime import datetime
import io
from typing import Optional, List, Dict, Any


class OpenPowerliftingIntegrator:
    """Integrates OpenPowerlifting meet data with training data."""
    
    def __init__(self, cache_dir: str = "opl_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data_file = self.cache_dir / "openpowerlifting.csv"
        self.last_update_file = self.cache_dir / "last_update.txt"
        
        # OpenPowerlifting URLs (updated)
        self.zip_url = "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
    
    def download_data(self, force_update: bool = False) -> bool:
        """Download OpenPowerlifting data if not cached or if force update."""
        
        # Check if we need to download
        if not force_update and self.data_file.exists():
            if self.last_update_file.exists():
                with open(self.last_update_file, 'r') as f:
                    last_update = f.read().strip()
                    print(f"📁 Using cached data from {last_update}")
                    return True
        
        print("📥 Downloading OpenPowerlifting database (145MB)...")
        print("⏱️ This may take a few minutes...")
        
        try:
            # Download ZIP file
            response = requests.get(self.zip_url, timeout=120, stream=True)
            response.raise_for_status()
            
            print("✅ Download complete, extracting CSV...")
            
            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # The CSV file should be named 'openpowerlifting.csv' in the ZIP
                csv_filename = 'openpowerlifting.csv'
                
                if csv_filename in zip_file.namelist():
                    with zip_file.open(csv_filename) as csv_file:
                        with open(self.data_file, 'wb') as f:
                            f.write(csv_file.read())
                    print(f"✅ Extracted CSV to {self.data_file}")
                else:
                    # Try to find any CSV file
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                    if csv_files:
                        csv_filename = csv_files[0]
                        with zip_file.open(csv_filename) as csv_file:
                            with open(self.data_file, 'wb') as f:
                                f.write(csv_file.read())
                        print(f"✅ Extracted CSV ({csv_filename}) to {self.data_file}")
                    else:
                        raise FileNotFoundError("No CSV file found in ZIP")
            
            # Update timestamp
            with open(self.last_update_file, 'w') as f:
                f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load OpenPowerlifting data from cache."""
        if not self.data_file.exists():
            print("❌ No cached data found. Run download_data() first.")
            return None
        
        try:
            print("📊 Loading OpenPowerlifting database...")
            df = pd.read_csv(self.data_file, low_memory=False)
            
            # Convert dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Convert weights to float and handle missing values
            weight_cols = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'BodyweightKg']
            for col in weight_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"✅ Loaded {len(df):,} competition entries")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def search_lifter(self, df: pd.DataFrame, name: str, fuzzy: bool = True) -> pd.DataFrame:
        """Search for a lifter by name."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Direct match first
        direct_match = df[df['Name'].str.contains(name, case=False, na=False)]
        
        if not direct_match.empty:
            return direct_match
        
        if fuzzy:
            # Fuzzy matching - split name and search for parts
            name_parts = name.lower().split()
            mask = pd.Series([True] * len(df))
            
            for part in name_parts:
                if len(part) > 2:  # Only search for parts longer than 2 chars
                    mask &= df['Name'].str.lower().str.contains(part, na=False)
            
            return df[mask]
        
        return pd.DataFrame()
    
    def get_lifter_meets(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Get all meet results for a specific lifter."""
        lifter_data = self.search_lifter(df, name)
        
        if lifter_data.empty:
            print(f"❌ No meet results found for '{name}'")
            return pd.DataFrame()
        
        # Sort by date
        lifter_data = lifter_data.sort_values('Date')
        
        # Select relevant columns
        columns = [
            'Date', 'MeetName', 'Federation', 'Equipment', 'Division',
            'WeightClassKg', 'BodyweightKg', 'Place',
            'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg',
            'Dots', 'Wilks'
        ]
        
        available_columns = [col for col in columns if col in lifter_data.columns]
        result = lifter_data[available_columns].copy()
        
        print(f"✅ Found {len(result)} meet results for '{name}'")
        return result
    
    def convert_to_lbs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert weights from kg to lbs for consistency with training data."""
        if df.empty:
            return df
        
        df = df.copy()
        weight_cols = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'BodyweightKg']
        
        for col in weight_cols:
            if col in df.columns:
                new_col = col.replace('Kg', 'Lbs')
                df[new_col] = df[col] * 2.20462
                df[new_col] = df[new_col].round(1)
        
        return df
    
    def format_for_dashboard(self, meet_data: pd.DataFrame) -> pd.DataFrame:
        """Format meet data for integration with training dashboard."""
        if meet_data.empty:
            return pd.DataFrame()
        
        df = meet_data.copy()
        
        # Convert to lbs
        df = self.convert_to_lbs(df)
        
        # Create standardized format matching training data
        formatted_data = []
        
        for _, row in df.iterrows():
            date = row['Date']
            meet_name = row.get('MeetName', 'Unknown Meet')
            
            # Add each lift as a separate row
            lifts = [
                ('Squat', row.get('Best3SquatLbs', 0)),
                ('Bench', row.get('Best3BenchLbs', 0)),
                ('Deadlift', row.get('Best3DeadliftLbs', 0))
            ]
            
            for lift_type, weight in lifts:
                if pd.notna(weight) and weight > 0:
                    formatted_data.append({
                        'Date': date,
                        'Block': f"MEET: {meet_name}",
                        'Day': 'Competition',
                        'Exercise': f'Competition {lift_type}',
                        'Sets': 1,
                        'Reps': 1,
                        'Load': weight,
                        'RPE': None,
                        'Notes': f"Competition - {row.get('Federation', '')} {row.get('Division', '')}",
                        'Lift_Type': lift_type,
                        'Meet_Result': True,
                        'Place': row.get('Place', ''),
                        'Total': row.get('TotalLbs', 0),
                        'Dots': row.get('Dots', 0)
                    })
        
        return pd.DataFrame(formatted_data)
    
    def integrate_with_training(self, training_df: pd.DataFrame, meet_df: pd.DataFrame) -> pd.DataFrame:
        """Combine training data with meet results."""
        if meet_df.empty:
            # Add meet_result flag to training data
            training_df = training_df.copy()
            training_df['Meet_Result'] = False
            return training_df
        
        # Add meet_result flag to training data
        training_df = training_df.copy()
        training_df['Meet_Result'] = False
        
        # Add missing columns to meet data with default values
        for col in training_df.columns:
            if col not in meet_df.columns:
                if col in ['Place', 'Total', 'Dots']:
                    continue  # These are meet-specific
                elif col == 'Meet_Result':
                    continue  # Already set
                else:
                    meet_df[col] = None
        
        # Combine datasets
        combined_df = pd.concat([training_df, meet_df], ignore_index=True)
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Integrated {len(meet_df)} meet results with {len(training_df)} training entries")
        return combined_df


def main():
    """Demo usage of the OpenPowerlifting integrator."""
    
    # Initialize integrator
    integrator = OpenPowerliftingIntegrator()
    
    # Download data (comment out if you want to use cache)
    if not integrator.download_data():
        print("❌ Failed to download data")
        return
    
    # Load data
    df = integrator.load_data()
    if df is None:
        return
    
    # Example: Search for a lifter
    print("\n🔍 Searching for lifters...")
    
    # You can search for your own name here
    lifter_name = input("Enter lifter name to search for: ").strip()
    
    if lifter_name:
        meet_results = integrator.get_lifter_meets(df, lifter_name)
        
        if not meet_results.empty:
            print(f"\n📊 Meet results for '{lifter_name}':")
            print(meet_results[['Date', 'MeetName', 'Federation', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg']].to_string())
            
            # Format for dashboard
            formatted = integrator.format_for_dashboard(meet_results)
            print(f"\n🎯 Formatted {len(formatted)} entries for dashboard integration")
        
        else:
            print(f"❌ No results found for '{lifter_name}'")
            print("💡 Try variations of your name or check if you've competed in tracked federations")


if __name__ == "__main__":
    main() 