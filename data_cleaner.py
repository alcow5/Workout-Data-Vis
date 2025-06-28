#!/usr/bin/env python3
"""
Data Cleaner for Workout Log Normalizer
Analyzes and cleans blank or invalid entries from normalized data
"""

import pandas as pd
import argparse


class DataCleaner:
    """Tool to analyze and clean normalized workout data."""
    
    def __init__(self, csv_file):
        """Initialize with path to normalized CSV file."""
        self.csv_file = csv_file
        self.df = None
        
    def load_data(self):
        """Load the normalized CSV data."""
        print(f"📂 Loading data from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        self.df['Date'] = pd.to_datetime(self.df['Date']).dt.date
        print(f"✅ Loaded {len(self.df)} entries")
        
    def analyze_blank_entries(self):
        """Analyze different types of blank/incomplete entries."""
        print(f"\n🔍 BLANK ENTRY ANALYSIS")
        print(f"{'='*60}")
        
        df = self.df
        
        # Completely blank entries (no sets, reps, load)
        completely_blank = df[(df['Sets'].isna()) & (df['Reps'].isna()) & (df['Load'].isna())]
        print(f"\n📊 Completely blank entries: {len(completely_blank)} ({len(completely_blank)/len(df)*100:.1f}%)")
        
        # Analyze what these blank entries contain
        blank_exercises = completely_blank['Exercise'].value_counts()
        print(f"\n🏋️  Most common 'exercises' in blank entries:")
        for exercise, count in blank_exercises.head(10).items():
            print(f"  • '{exercise}': {count} times")
        
        # Partially blank (missing some but not all values)
        partially_blank = df[
            (df['Sets'].isna() | df['Reps'].isna() | df['Load'].isna()) &
            ~((df['Sets'].isna()) & (df['Reps'].isna()) & (df['Load'].isna()))
        ]
        print(f"\n📊 Partially blank entries: {len(partially_blank)} ({len(partially_blank)/len(df)*100:.1f}%)")
        
        # Apply validation rules
        validation_rules = {
            'RPE': {'min': 1, 'max': 10},
            'Sets': {'min': 1, 'max': 8}, 
            'Reps': {'min': 1, 'max': 50},
            'Load': {'min': 1, 'max': 700}
        }
        
        suspicious = pd.DataFrame()
        for col, rules in validation_rules.items():
            if col in df.columns:
                invalid_mask = (
                    (df[col] < rules['min']) | 
                    (df[col] > rules['max'])
                ) & df[col].notna()
                
                if invalid_mask.any():
                    invalid_entries = df[invalid_mask]
                    suspicious = pd.concat([suspicious, invalid_entries]).drop_duplicates()
                    print(f"\n⚠️  {col} outside {rules['min']}-{rules['max']}: {invalid_mask.sum()} entries")
        
        print(f"\n⚠️  Total suspicious entries: {len(suspicious)} ({len(suspicious)/len(df)*100:.1f}%)")
        
        if len(suspicious) > 0:
            print("Sample suspicious entries:")
            for _, row in suspicious.head(5).iterrows():
                print(f"  • {row['Exercise']}: {row['Sets']}×{row['Reps']} @ {row['Load']} lbs (RPE {row['RPE']})")
        
        # Header-like entries
        header_like = completely_blank[completely_blank['Exercise'].str.contains('Exercise|exercise', na=False)]
        print(f"\n📋 Header-like entries: {len(header_like)}")
        
        return {
            'completely_blank': completely_blank,
            'partially_blank': partially_blank,
            'suspicious': suspicious,
            'header_like': header_like
        }
    
    def apply_validation_rules(self, rules=None):
        """Apply validation rules and show detailed breakdown."""
        if rules is None:
            rules = {
                'RPE': {'min': 1, 'max': 10},
                'Sets': {'min': 1, 'max': 8}, 
                'Reps': {'min': 1, 'max': 50},
                'Load': {'min': 1, 'max': 700}
            }
        
        print(f"\n🔧 APPLYING VALIDATION RULES")
        print(f"{'='*60}")
        print("Rules:")
        for col, rule in rules.items():
            print(f"  • {col}: {rule['min']} - {rule['max']}")
        
        violations = {}
        total_violations = 0
        
        for col, rule in rules.items():
            if col in self.df.columns:
                invalid_mask = (
                    (self.df[col] < rule['min']) | 
                    (self.df[col] > rule['max'])
                ) & self.df[col].notna()
                
                count = invalid_mask.sum()
                if count > 0:
                    violations[col] = {
                        'count': count,
                        'entries': self.df[invalid_mask],
                        'rule': rule
                    }
                    total_violations += count
                    
                    print(f"\n⚠️  {col} violations: {count}")
                    
                    # Show examples of violations
                    examples = self.df[invalid_mask].head(3)
                    for _, row in examples.iterrows():
                        value = row[col]
                        print(f"    • {row['Exercise']}: {col}={value} (should be {rule['min']}-{rule['max']})")
        
        print(f"\n📊 Total entries violating rules: {total_violations}")
        return violations
    
    def create_clean_dataset(self, remove_completely_blank=True, remove_header_like=True, 
                           remove_suspicious=False, validation_rules=None):
        """Create a cleaned version of the dataset."""
        print(f"\n🧹 CREATING CLEAN DATASET")
        print(f"{'='*60}")
        
        df_clean = self.df.copy()
        original_count = len(df_clean)
        
        removed_counts = {}
        
        # Remove header-like entries
        if remove_header_like:
            header_mask = df_clean['Exercise'].str.contains('Exercise|exercise|notes:', na=False, case=False)
            removed_counts['header_like'] = header_mask.sum()
            df_clean = df_clean[~header_mask]
            print(f"🗑️  Removed {removed_counts['header_like']} header-like entries")
        
        # Remove completely blank entries
        if remove_completely_blank:
            blank_mask = (df_clean['Sets'].isna()) & (df_clean['Reps'].isna()) & (df_clean['Load'].isna())
            removed_counts['completely_blank'] = blank_mask.sum()
            df_clean = df_clean[~blank_mask]
            print(f"🗑️  Removed {removed_counts['completely_blank']} completely blank entries")
        
        # Remove entries that violate validation rules
        if remove_suspicious:
            if validation_rules is None:
                validation_rules = {
                    'RPE': {'min': 1, 'max': 10},
                    'Sets': {'min': 1, 'max': 8}, 
                    'Reps': {'min': 1, 'max': 50},
                    'Load': {'min': 1, 'max': 700}
                }
            
            suspicious_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
            for col, rules in validation_rules.items():
                if col in df_clean.columns:
                    invalid_mask = (
                        (df_clean[col] < rules['min']) | 
                        (df_clean[col] > rules['max'])
                    ) & df_clean[col].notna()
                    suspicious_mask = suspicious_mask | invalid_mask
            
            removed_counts['suspicious'] = suspicious_mask.sum()
            df_clean = df_clean[~suspicious_mask]
            print(f"🗑️  Removed {removed_counts['suspicious']} entries violating validation rules")
        
        total_removed = original_count - len(df_clean)
        print(f"\n📊 Cleaning Summary:")
        print(f"  Original entries: {original_count}")
        print(f"  Removed entries: {total_removed}")
        print(f"  Clean entries: {len(df_clean)} ({len(df_clean)/original_count*100:.1f}%)")
        
        return df_clean, removed_counts
    
    def save_clean_data(self, df_clean, output_file=None):
        """Save the cleaned dataset."""
        if output_file is None:
            output_file = self.csv_file.replace('.csv', '_clean.csv')
        
        df_clean.to_csv(output_file, index=False)
        print(f"💾 Saved clean dataset to: {output_file}")
        return output_file


def main():
    """Command line interface for data cleaning."""
    parser = argparse.ArgumentParser(description='Clean normalized workout data')
    parser.add_argument('--csv', default='AlexWOLog_normalized.csv', help='Normalized CSV file path')
    parser.add_argument('--analyze', '-a', action='store_true', help='Just analyze, don\'t clean')
    parser.add_argument('--clean', '-c', action='store_true', help='Create clean dataset with recommended settings')
    parser.add_argument('--validate', '-v', action='store_true', help='Create clean dataset with validation rules')
    parser.add_argument('--output', '-o', help='Output file for clean dataset')
    
    args = parser.parse_args()
    
    try:
        cleaner = DataCleaner(args.csv)
        cleaner.load_data()
        
        if args.analyze:
            cleaner.analyze_blank_entries()
            cleaner.apply_validation_rules()
        elif args.clean:
            df_clean, removed = cleaner.create_clean_dataset()
            cleaner.save_clean_data(df_clean, args.output)
        elif args.validate:
            df_clean, removed = cleaner.create_clean_dataset(
                remove_completely_blank=True,
                remove_header_like=True,
                remove_suspicious=True
            )
            output_file = args.output or cleaner.csv_file.replace('.csv', '_validated.csv')
            cleaner.save_clean_data(df_clean, output_file)
        else:
            # Default: show analysis
            cleaner.analyze_blank_entries()
            cleaner.apply_validation_rules()
            
            print(f"\nOptions:")
            print(f"  --clean: Create basic clean dataset")
            print(f"  --validate: Create dataset with validation rules (RPE 1-10, Sets 1-8, Reps 1-50, Load 1-700)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 