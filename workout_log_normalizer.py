import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from pathlib import Path
import argparse


class WorkoutLogNormalizer:
    """
    Workout Log Normalizer - Parses structured Excel workout logs and normalizes them into long-form data.
    
    Each sheet represents a training block with weeks laid out horizontally every 12 columns.
    Week start dates are in row 0, and exercise data follows specific column mappings.
    """
    
    def __init__(self, excel_file_path):
        """Initialize with path to Excel file."""
        self.excel_file_path = Path(excel_file_path)
        self.workbook = None
        self.normalized_data = []
        
    def load_workbook(self):
        """Load the Excel workbook."""
        try:
            self.workbook = pd.ExcelFile(self.excel_file_path, engine='openpyxl')
            print(f"✅ Loaded workbook: {self.excel_file_path}")
            print(f"📊 Found {len(self.workbook.sheet_names)} sheets: {self.workbook.sheet_names}")
        except Exception as e:
            raise Exception(f"❌ Failed to load workbook: {e}")
    
    def detect_week_columns(self, df):
        """
        Detect week start columns by finding dates in row 1.
        Returns list of (column_index, date) tuples.
        """
        week_columns = []
        
        # Check row 1 for dates (row index 1)
        if len(df) > 1:
            date_row = df.iloc[1]
            
            for col_idx, cell_value in enumerate(date_row):
                if pd.notna(cell_value):
                    # Try to parse as date
                    date_value = self._parse_date(cell_value)
                    if date_value:
                        week_columns.append((col_idx, date_value))
                        print(f"🗓️  Found week starting {date_value} at column {col_idx}")
        
        return week_columns
    
    def _parse_date(self, value):
        """Try to parse a value as a date."""
        if isinstance(value, datetime):
            return value.date()
        
        # Handle pandas Timestamp objects
        if hasattr(value, 'date'):
            return value.date()
        
        if isinstance(value, str):
            # Try common date formats
            date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
            for fmt in date_formats:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        
        return None
    
    def classify_lift_type(self, exercise_name):
        """
        Classify exercises into lift types based on exercise name.
        """
        if pd.isna(exercise_name) or not isinstance(exercise_name, str):
            return "Accessory"
        
        exercise_lower = exercise_name.lower()
        
        # Squat variations
        if any(term in exercise_lower for term in ['squat']):
            return "Squat"
        
        # Deadlift variations  
        if any(term in exercise_lower for term in ['deadlift', 'rdl', 'romanian deadlift']):
            return "Deadlift"
        
        # Bench variations (excluding leg press)
        if any(term in exercise_lower for term in ['bench', 'press']) and 'leg' not in exercise_lower:
            return "Bench"
        
        return "Accessory"
    
    def parse_week_data(self, df, block_name, week_start_date, base_col):
        """
        Parse data for a single week starting at base_col.
        
        Expected structure:
        - Exercise → base_col + 0
        - Load → base_col + 6  
        - Sets → base_col + 7
        - Reps → base_col + 8
        - RPE → base_col + 9
        - Notes → base_col + 10
        """
        week_data = []
        current_day = None
        current_day_number = None
        
        # Look for data starting from row 2 (skip the date and week label rows)
        for row_idx in range(2, len(df)):
            row = df.iloc[row_idx]
            
            # Check if this row contains a DAY label
            day_label = self._extract_day_label(row, base_col)
            if day_label:
                current_day = day_label
                current_day_number = self._extract_day_number(day_label)
                continue
            
            # Extract exercise data if we're in a valid day
            if current_day and current_day_number is not None:
                # Calculate actual date: week_start_date + (day_number - 1) days
                actual_date = week_start_date + timedelta(days=current_day_number - 1)
                
                exercise_data = self._extract_exercise_data(row, base_col, block_name, actual_date, current_day)
                if exercise_data:
                    week_data.append(exercise_data)
        
        return week_data
    
    def _extract_day_label(self, row, base_col):
        """Extract DAY label from row if present."""
        # Check a few columns around base_col for DAY labels
        for col_offset in range(0, 12):
            col_idx = base_col + col_offset
            if col_idx < len(row):
                cell_value = row.iloc[col_idx] if col_idx < len(row) else None
                if pd.notna(cell_value) and isinstance(cell_value, str):
                    if re.match(r'day\s*\d+', str(cell_value).lower().strip()):
                        return str(cell_value).strip()
        return None
    
    def _extract_day_number(self, day_label):
        """Extract day number from day label (e.g., 'Day 1' -> 1, 'Day 2' -> 2)."""
        if not day_label:
            return None
        
        # Use regex to extract the number from labels like "Day 1", "DAY 2", etc.
        match = re.search(r'day\s*(\d+)', day_label.lower().strip())
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
    
    def _extract_exercise_data(self, row, base_col, block_name, week_start_date, day):
        """Extract exercise data from a row."""
        try:
            # Column mappings as per PRD
            exercise_col = base_col + 0
            load_col = base_col + 6
            sets_col = base_col + 7
            reps_col = base_col + 8
            rpe_col = base_col + 9
            notes_col = base_col + 10
            
            # Extract values safely
            exercise = row.iloc[exercise_col] if exercise_col < len(row) else None
            load = row.iloc[load_col] if load_col < len(row) else None
            sets = row.iloc[sets_col] if sets_col < len(row) else None
            reps = row.iloc[reps_col] if reps_col < len(row) else None
            rpe = row.iloc[rpe_col] if rpe_col < len(row) else None
            notes = row.iloc[notes_col] if notes_col < len(row) else None
            
            # Skip if no exercise name
            if pd.isna(exercise) or str(exercise).strip() == '':
                return None
            
            # Clean and convert values
            exercise = str(exercise).strip()
            load = self._clean_numeric(load)
            sets = self._clean_numeric(sets) 
            reps = self._clean_numeric(reps)
            rpe = self._clean_numeric(rpe)
            notes = str(notes).strip() if pd.notna(notes) else ''
            
            # Classify lift type
            lift_type = self.classify_lift_type(exercise)
            
            return {
                'Date': week_start_date,
                'Block': block_name,
                'Day': day,
                'Exercise': exercise,
                'Sets': sets,
                'Reps': reps,
                'Load': load,
                'RPE': rpe,
                'Notes': notes,
                'Lift_Type': lift_type
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting exercise data: {e}")
            return None
    
    def _clean_numeric(self, value):
        """Clean and convert numeric values."""
        if pd.isna(value):
            return None
        
        if isinstance(value, (int, float)):
            return value
        
        # Try to extract number from string
        if isinstance(value, str):
            # Remove common non-numeric characters
            cleaned = re.sub(r'[^\d.-]', '', value.strip())
            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                return None
        
        return None
    
    def normalize_sheet(self, sheet_name):
        """Normalize a single sheet into long-form data."""
        print(f"\n📋 Processing sheet: {sheet_name}")
        
        try:
            # Read sheet with all data as strings initially to preserve formatting
            df = pd.read_excel(self.workbook, sheet_name=sheet_name, header=None)
            
            if df.empty:
                print(f"⚠️  Sheet {sheet_name} is empty")
                return []
            
            # Detect week columns (every 12 columns as per PRD)
            week_columns = self.detect_week_columns(df)
            
            if not week_columns:
                print(f"⚠️  No week dates found in {sheet_name}")
                return []
            
            sheet_data = []
            
            # Process each week
            for col_idx, week_date in week_columns:
                print(f"  📅 Processing week {week_date} at column {col_idx}")
                week_data = self.parse_week_data(df, sheet_name, week_date, col_idx)
                sheet_data.extend(week_data)
                print(f"    ✅ Found {len(week_data)} exercise entries")
            
            print(f"📊 Sheet {sheet_name}: {len(sheet_data)} total entries")
            return sheet_data
            
        except Exception as e:
            print(f"❌ Error processing sheet {sheet_name}: {e}")
            return []
    
    def normalize_all_sheets(self):
        """Normalize all sheets in the workbook."""
        if not self.workbook:
            raise Exception("Workbook not loaded. Call load_workbook() first.")
        
        print(f"\n🚀 Starting normalization of {len(self.workbook.sheet_names)} sheets...")
        
        all_data = []
        for sheet_name in self.workbook.sheet_names:
            sheet_data = self.normalize_sheet(sheet_name)
            all_data.extend(sheet_data)
        
        # Convert to DataFrame
        if all_data:
            self.normalized_data = pd.DataFrame(all_data)
            print(f"\n✅ Normalization complete!")
            print(f"📊 Total entries: {len(self.normalized_data)}")
            print(f"🏋️  Unique exercises: {self.normalized_data['Exercise'].nunique()}")
            print(f"🗓️  Date range: {self.normalized_data['Date'].min()} to {self.normalized_data['Date'].max()}")
            print(f"🏷️  Lift types: {self.normalized_data['Lift_Type'].value_counts().to_dict()}")
        else:
            self.normalized_data = pd.DataFrame()
            print("⚠️  No data found to normalize")
        
        return self.normalized_data
    
    def export_csv(self, output_path=None):
        """Export normalized data to CSV."""
        if self.normalized_data.empty:
            print("❌ No data to export. Run normalize_all_sheets() first.")
            return
        
        if not output_path:
            output_path = self.excel_file_path.stem + "_normalized.csv"
        
        self.normalized_data.to_csv(output_path, index=False)
        print(f"💾 Exported to: {output_path}")
        return output_path
    
    def get_summary(self):
        """Get a summary of the normalized data."""
        if self.normalized_data.empty:
            return "No data available"
        
        summary = {
            'total_entries': len(self.normalized_data),
            'unique_exercises': self.normalized_data['Exercise'].nunique(),
            'date_range': f"{self.normalized_data['Date'].min()} to {self.normalized_data['Date'].max()}",
            'blocks': self.normalized_data['Block'].unique().tolist(),
            'lift_types': self.normalized_data['Lift_Type'].value_counts().to_dict()
        }
        return summary


def main():
    """Command line interface for the workout log normalizer."""
    parser = argparse.ArgumentParser(description='Normalize workout log Excel files')
    parser.add_argument('excel_file', help='Path to Excel file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--summary', '-s', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    try:
        # Initialize normalizer
        normalizer = WorkoutLogNormalizer(args.excel_file)
        
        # Load and process
        normalizer.load_workbook()
        df = normalizer.normalize_all_sheets()
        
        if args.summary:
            print("\n📈 SUMMARY:")
            summary = normalizer.get_summary()
            for key, value in summary.items():
                print(f"  {key}: {value}")
        else:
            # Export to CSV
            output_file = normalizer.export_csv(args.output)
            print(f"\n✅ Complete! Data saved to: {output_file}")
            
            # Show preview
            if not df.empty:
                print(f"\n👀 Preview (first 5 rows):")
                print(df.head().to_string())
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 