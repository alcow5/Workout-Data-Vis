#!/usr/bin/env python3
"""
Spot Checker for Workout Log Normalizer
Validates normalized data against original Excel file
"""

import pandas as pd
import random
from datetime import datetime
import argparse


class SpotChecker:
    """Tool to validate normalized workout data against original Excel file."""
    
    def __init__(self, excel_file, csv_file):
        """Initialize with paths to Excel and CSV files."""
        self.excel_file = excel_file
        self.csv_file = csv_file
        self.workbook = None
        self.normalized_df = None
        
    def load_data(self):
        """Load both the original Excel and normalized CSV data."""
        print("📂 Loading data for validation...")
        
        # Load Excel workbook
        self.workbook = pd.ExcelFile(self.excel_file, engine='openpyxl')
        print(f"✅ Loaded Excel: {len(self.workbook.sheet_names)} sheets")
        
        # Load normalized CSV
        self.normalized_df = pd.read_csv(self.csv_file)
        self.normalized_df['Date'] = pd.to_datetime(self.normalized_df['Date']).dt.date
        print(f"✅ Loaded CSV: {len(self.normalized_df)} entries")
        
    def get_random_samples(self, n=5):
        """Get n random samples from the normalized data."""
        if self.normalized_df is None:
            raise Exception("Data not loaded. Call load_data() first.")
        
        samples = self.normalized_df.sample(n=min(n, len(self.normalized_df)))
        return samples.reset_index(drop=True)
    
    def show_sample_details(self, sample_row):
        """Show detailed information about a sample row."""
        print(f"\n{'='*60}")
        print(f"🔍 SAMPLE VALIDATION")
        print(f"{'='*60}")
        
        # Extract sample info
        date = sample_row['Date']
        block = sample_row['Block']
        day = sample_row['Day']
        exercise = sample_row['Exercise']
        sets = sample_row['Sets']
        reps = sample_row['Reps']
        load = sample_row['Load']
        rpe = sample_row['RPE']
        notes = sample_row['Notes']
        lift_type = sample_row['Lift_Type']
        
        print(f"📅 Date: {date}")
        print(f"📋 Block: {block}")
        print(f"🏷️  Day: {day}")
        print(f"🏋️  Exercise: {exercise}")
        print(f"📊 Sets: {sets} | Reps: {reps} | Load: {load} | RPE: {rpe}")
        print(f"📝 Notes: {notes}")
        print(f"🏷️  Classified as: {lift_type}")
        
        return self.find_in_excel(date, block, exercise)
    
    def find_in_excel(self, target_date, block_name, exercise_name):
        """Find the corresponding entry in the original Excel file."""
        try:
            # Read the specific sheet
            df = pd.read_excel(self.workbook, sheet_name=block_name, header=None)
            
            print(f"\n🔍 Looking in Excel sheet '{block_name}'...")
            
            # Find the week column that contains this date
            week_col = None
            for col_idx, cell_value in enumerate(df.iloc[1] if len(df) > 1 else []):
                if pd.notna(cell_value):
                    if hasattr(cell_value, 'date'):
                        cell_date = cell_value.date()
                    elif isinstance(cell_value, datetime):
                        cell_date = cell_value.date()
                    else:
                        continue
                    
                    if cell_date == target_date:
                        week_col = col_idx
                        break
            
            if week_col is None:
                print(f"⚠️  Could not find date {target_date} in sheet {block_name}")
                return False
            
            print(f"📅 Found date {target_date} at column {week_col}")
            
            # Look for the exercise in that week's data
            found_exercise = False
            for row_idx in range(2, len(df)):
                row = df.iloc[row_idx]
                
                # Check exercise name in the base column
                if week_col < len(row):
                    cell_value = row.iloc[week_col]
                    if pd.notna(cell_value) and isinstance(cell_value, str):
                        if exercise_name.lower().strip() in str(cell_value).lower().strip():
                            found_exercise = True
                            print(f"✅ Found exercise '{exercise_name}' at row {row_idx}")
                            
                            # Show the raw Excel data for this row
                            print(f"\n📋 Raw Excel data around column {week_col}:")
                            for offset in range(0, 11):  # Show exercise + 10 columns
                                col = week_col + offset
                                if col < len(row):
                                    value = row.iloc[col]
                                    label = ["Exercise", "?", "?", "?", "?", "?", "Load", "Sets", "Reps", "RPE", "Notes"][offset]
                                    print(f"  {label:10}: {value}")
                            break
            
            if not found_exercise:
                print(f"⚠️  Could not find exercise '{exercise_name}' in expected location")
                
                # Show some context around the expected area
                print(f"\n🔍 Context around column {week_col}:")
                for row_idx in range(max(0, 4), min(len(df), 15)):
                    row = df.iloc[row_idx]
                    if week_col < len(row):
                        cell_value = row.iloc[week_col]
                        if pd.notna(cell_value) and str(cell_value).strip():
                            print(f"  Row {row_idx}: {cell_value}")
                            
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error checking Excel: {e}")
            return False
    
    def validate_data_quality(self):
        """Run data quality checks on the normalized data."""
        print(f"\n🔍 DATA QUALITY ANALYSIS")
        print(f"{'='*60}")
        
        df = self.normalized_df
        
        # Check for missing data
        print(f"📊 Missing Data Analysis:")
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"  {col}: {count} missing ({pct:.1f}%)")
        
        # Check for suspicious values
        print(f"\n⚠️  Suspicious Values:")
        
        # Extremely high loads
        if 'Load' in df.columns:
            high_loads = df[df['Load'] > 1000]['Load'].count()
            if high_loads > 0:
                print(f"  • {high_loads} entries with load > 1000 lbs")
                print(f"    Max load: {df['Load'].max()}")
        
        # Very high reps
        if 'Reps' in df.columns:
            high_reps = df[df['Reps'] > 50]['Reps'].count()
            if high_reps > 0:
                print(f"  • {high_reps} entries with reps > 50")
        
        # RPE outside normal range
        if 'RPE' in df.columns:
            weird_rpe = df[(df['RPE'] < 1) | (df['RPE'] > 10)]['RPE'].count()
            if weird_rpe > 0:
                print(f"  • {weird_rpe} entries with RPE outside 1-10 range")
        
        # Check lift type distribution
        print(f"\n🏷️  Lift Type Distribution:")
        lift_counts = df['Lift_Type'].value_counts()
        for lift_type, count in lift_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {lift_type}: {count} ({pct:.1f}%)")
        
        # Check date coverage
        print(f"\n📅 Date Coverage:")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Total days: {(df['Date'].max() - df['Date'].min()).days}")
        
        # Block coverage
        print(f"\n📋 Block Coverage:")
        block_counts = df['Block'].value_counts().head(10)
        for block, count in block_counts.items():
            print(f"  {block}: {count} entries")
        
    def show_exercise_examples(self, exercise_name, n=3):
        """Show examples of a specific exercise."""
        matches = self.normalized_df[
            self.normalized_df['Exercise'].str.contains(exercise_name, case=False, na=False)
        ]
        
        if len(matches) == 0:
            print(f"❌ No exercises found containing '{exercise_name}'")
            return
        
        print(f"\n🏋️  Found {len(matches)} entries for exercises containing '{exercise_name}'")
        print(f"📋 Showing {min(n, len(matches))} examples:")
        
        samples = matches.sample(min(n, len(matches)))
        for _, row in samples.iterrows():
            print(f"\n  📅 {row['Date']} | {row['Block']} | {row['Day']}")
            print(f"  🏋️  {row['Exercise']}")
            print(f"  📊 {row['Sets']} sets × {row['Reps']} reps @ {row['Load']} lbs (RPE {row['RPE']})")
    
    def interactive_spot_check(self):
        """Run an interactive spot checking session."""
        print(f"\n🔍 INTERACTIVE SPOT CHECKER")
        print(f"{'='*60}")
        
        while True:
            print(f"\nOptions:")
            print(f"1. Random sample validation")
            print(f"2. Data quality analysis")
            print(f"3. Search specific exercise")
            print(f"4. Quit")
            
            try:
                choice = input("\nChoose option (1-4): ").strip()
                
                if choice == '1':
                    n_samples = int(input("How many samples to check (default 3): ") or "3")
                    samples = self.get_random_samples(n_samples)
                    
                    for i, (_, sample) in enumerate(samples.iterrows()):
                        print(f"\n{'-'*40}")
                        print(f"SAMPLE {i+1}/{len(samples)}")
                        print(f"{'-'*40}")
                        self.show_sample_details(sample)
                        
                        if i < len(samples) - 1:
                            input("\nPress Enter for next sample...")
                
                elif choice == '2':
                    self.validate_data_quality()
                
                elif choice == '3':
                    exercise = input("Enter exercise name to search: ").strip()
                    if exercise:
                        self.show_exercise_examples(exercise)
                
                elif choice == '4':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid option. Please choose 1-4.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Command line interface for spot checking."""
    parser = argparse.ArgumentParser(description='Spot check normalized workout data')
    parser.add_argument('--excel', default='AlexWOLog.xlsx', help='Excel file path')
    parser.add_argument('--csv', default='AlexWOLog_normalized.csv', help='Normalized CSV file path')
    parser.add_argument('--samples', '-n', type=int, default=5, help='Number of random samples')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--exercise', '-e', help='Show examples of specific exercise')
    
    args = parser.parse_args()
    
    try:
        checker = SpotChecker(args.excel, args.csv)
        checker.load_data()
        
        if args.interactive:
            checker.interactive_spot_check()
        elif args.exercise:
            checker.show_exercise_examples(args.exercise)
        else:
            # Default: show random samples
            print(f"\n🔍 SPOT CHECKING {args.samples} RANDOM SAMPLES")
            print(f"{'='*60}")
            
            samples = checker.get_random_samples(args.samples)
            for i, (_, sample) in enumerate(samples.iterrows()):
                print(f"\n{'-'*40}")
                print(f"SAMPLE {i+1}/{len(samples)}")
                print(f"{'-'*40}")
                checker.show_sample_details(sample)
                
                if i < len(samples) - 1:
                    input("\nPress Enter for next sample...")
            
            # Show data quality summary
            checker.validate_data_quality()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 