#!/usr/bin/env python3
"""
1RM Calculator for Workout Analysis
Estimates one-rep max using various formulas and RPE data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List


class OneRMCalculator:
    """Calculate estimated 1RM using multiple methods."""
    
    @staticmethod
    def epley_formula(weight: float, reps: float) -> float:
        """
        Epley Formula: 1RM = weight × (1 + reps/30)
        Most popular and widely used formula.
        """
        if pd.isna(weight) or pd.isna(reps) or reps <= 0:
            return np.nan
        return weight * (1 + reps / 30)
    
    @staticmethod
    def brzycki_formula(weight: float, reps: float) -> float:
        """
        Brzycki Formula: 1RM = weight × (36 / (37 - reps))
        Good for lower rep ranges (1-10 reps).
        """
        if pd.isna(weight) or pd.isna(reps) or reps <= 0 or reps >= 37:
            return np.nan
        return weight * (36 / (37 - reps))
    
    @staticmethod
    def lombardi_formula(weight: float, reps: float) -> float:
        """
        Lombardi Formula: 1RM = weight × reps^0.10
        Good for higher rep ranges.
        """
        if pd.isna(weight) or pd.isna(reps) or reps <= 0:
            return np.nan
        return weight * (reps ** 0.10)
    
    @staticmethod
    def rpe_percentage_table() -> Dict[tuple, float]:
        """
        RPE percentage table for calculating 1RM from RPE.
        Key: (reps, rpe), Value: percentage of 1RM
        """
        return {
            # 1 rep
            (1, 6): 0.86, (1, 7): 0.92, (1, 8): 0.95, (1, 9): 0.97, (1, 10): 1.00,
            # 2 reps  
            (2, 6): 0.84, (2, 7): 0.89, (2, 8): 0.92, (2, 9): 0.95, (2, 10): 0.97,
            # 3 reps
            (3, 6): 0.81, (3, 7): 0.86, (3, 8): 0.89, (3, 9): 0.92, (3, 10): 0.95,
            # 4 reps
            (4, 6): 0.79, (4, 7): 0.84, (4, 8): 0.86, (4, 9): 0.89, (4, 10): 0.92,
            # 5 reps
            (5, 6): 0.77, (5, 7): 0.81, (5, 8): 0.84, (5, 9): 0.86, (5, 10): 0.89,
            # 6 reps
            (6, 6): 0.75, (6, 7): 0.79, (6, 8): 0.81, (6, 9): 0.84, (6, 10): 0.86,
            # 7 reps
            (7, 6): 0.73, (7, 7): 0.77, (7, 8): 0.79, (7, 9): 0.81, (7, 10): 0.84,
            # 8 reps
            (8, 6): 0.71, (8, 7): 0.75, (8, 8): 0.77, (8, 9): 0.79, (8, 10): 0.81,
            # 9 reps
            (9, 6): 0.69, (9, 7): 0.73, (9, 8): 0.75, (9, 9): 0.77, (9, 10): 0.79,
            # 10 reps
            (10, 6): 0.67, (10, 7): 0.71, (10, 8): 0.73, (10, 9): 0.75, (10, 10): 0.77,
        }
    
    @staticmethod
    def rpe_based_1rm(weight: float, reps: float, rpe: float) -> float:
        """
        Calculate 1RM using RPE-based percentage table.
        Most accurate for powerlifting with RPE data.
        """
        if pd.isna(weight) or pd.isna(reps) or pd.isna(rpe):
            return np.nan
        
        # Round values for table lookup
        reps_rounded = int(round(reps))
        rpe_rounded = int(round(rpe))
        
        # Get percentage table
        rpe_table = OneRMCalculator.rpe_percentage_table()
        
        # Look up percentage
        if (reps_rounded, rpe_rounded) in rpe_table:
            percentage = rpe_table[(reps_rounded, rpe_rounded)]
            return weight / percentage
        
        # Fallback to interpolation or formula if not in table
        return OneRMCalculator.epley_formula(weight, reps)
    
    @staticmethod
    def calculate_all_methods(weight: float, reps: float, rpe: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate 1RM using all available methods.
        Returns dictionary with method names and estimated 1RMs.
        """
        methods = {
            'epley': OneRMCalculator.epley_formula(weight, reps),
            'brzycki': OneRMCalculator.brzycki_formula(weight, reps),
            'lombardi': OneRMCalculator.lombardi_formula(weight, reps),
        }
        
        # Add RPE-based calculation if RPE is available
        if rpe is not None and not pd.isna(rpe):
            methods['rpe_based'] = OneRMCalculator.rpe_based_1rm(weight, reps, rpe)
        
        return methods
    
    @staticmethod
    def best_estimate(weight: float, reps: float, rpe: Optional[float] = None) -> float:
        """
        Get the best 1RM estimate based on available data.
        Prioritizes RPE-based > Brzycki > Epley > Lombardi
        """
        if pd.isna(weight) or pd.isna(reps):
            return np.nan
        
        # Use RPE-based if available and reps are in good range
        if rpe is not None and not pd.isna(rpe) and 1 <= reps <= 10:
            rpe_est = OneRMCalculator.rpe_based_1rm(weight, reps, rpe)
            if not pd.isna(rpe_est):
                return rpe_est
        
        # Use Brzycki for low reps (most accurate)
        if 1 <= reps <= 10:
            brzycki_est = OneRMCalculator.brzycki_formula(weight, reps)
            if not pd.isna(brzycki_est):
                return brzycki_est
        
        # Fallback to Epley
        return OneRMCalculator.epley_formula(weight, reps)


class WorkoutAnalyzer:
    """Analyze workout data and calculate 1RM progressions."""
    
    def __init__(self, data_file: str):
        """Initialize with workout data file."""
        self.data_file = data_file
        self.df = None
        self.calculator = OneRMCalculator()
        
    def load_data(self):
        """Load and prepare workout data."""
        print(f"📂 Loading workout data from {self.data_file}...")
        self.df = pd.read_csv(self.data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        print(f"✅ Loaded {len(self.df)} entries")
        
        # Filter to main lifts
        main_lifts = ['Squat', 'Bench', 'Deadlift']
        self.df = self.df[self.df['Lift_Type'].isin(main_lifts)]
        print(f"📊 Found {len(self.df)} main lift entries")
        
    def calculate_1rm_estimates(self):
        """Calculate 1RM estimates for all entries."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("🧮 Calculating 1RM estimates...")
        
        # Calculate best estimate for each row
        self.df['estimated_1rm'] = self.df.apply(
            lambda row: self.calculator.best_estimate(
                row['Load'], row['Reps'], row.get('RPE')
            ), axis=1
        )
        
        # Calculate all methods for comparison
        all_methods = self.df.apply(
            lambda row: self.calculator.calculate_all_methods(
                row['Load'], row['Reps'], row.get('RPE')
            ), axis=1
        )
        
        # Expand methods into separate columns
        methods_df = pd.DataFrame(all_methods.tolist())
        self.df = pd.concat([self.df, methods_df.add_prefix('1rm_')], axis=1)
        
        print(f"✅ Calculated 1RM estimates for {len(self.df)} entries")
        
    def get_1rm_progression(self, lift_type: str = None, method: str = 'best') -> pd.DataFrame:
        """
        Get 1RM progression over time for specified lift.
        
        Args:
            lift_type: 'Squat', 'Bench', 'Deadlift', or None for all
            method: 'best', 'epley', 'brzycki', 'lombardi', 'rpe_based'
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df_filtered = self.df.copy()
        
        # Filter by lift type
        if lift_type:
            df_filtered = df_filtered[df_filtered['Lift_Type'] == lift_type]
        
        # Select 1RM method
        if method == 'best':
            rm_col = 'estimated_1rm'
        else:
            rm_col = f'1rm_{method}'
        
        # Remove invalid estimates
        df_filtered = df_filtered[df_filtered[rm_col].notna()]
        
        # Group by date and take max 1RM estimate per day
        progression = df_filtered.groupby(['Date', 'Lift_Type'])[rm_col].max().reset_index()
        progression = progression.rename(columns={rm_col: 'estimated_1rm'})
        progression = progression.sort_values('Date')
        
        return progression
    
    def get_max_1rm_by_lift(self) -> pd.DataFrame:
        """Get maximum estimated 1RM for each lift type."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        max_1rms = []
        for lift in ['Squat', 'Bench', 'Deadlift']:
            lift_data = self.df[self.df['Lift_Type'] == lift]
            if not lift_data.empty:
                max_row = lift_data.loc[lift_data['estimated_1rm'].idxmax()]
                max_1rms.append({
                    'Lift_Type': lift,
                    'Max_1RM': max_row['estimated_1rm'],
                    'Date': max_row['Date'],
                    'Weight_Used': max_row['Load'],
                    'Reps': max_row['Reps'],
                    'RPE': max_row.get('RPE', np.nan)
                })
        
        return pd.DataFrame(max_1rms)
    
    def get_recent_progress(self, days: int = 90) -> pd.DataFrame:
        """Get recent progress in last N days."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        cutoff_date = self.df['Date'].max() - pd.Timedelta(days=days)
        recent_data = self.df[self.df['Date'] >= cutoff_date]
        
        progress = []
        for lift in ['Squat', 'Bench', 'Deadlift']:
            lift_data = recent_data[recent_data['Lift_Type'] == lift]
            if not lift_data.empty:
                max_recent = lift_data['estimated_1rm'].max()
                progress.append({
                    'Lift_Type': lift,
                    'Recent_Max_1RM': max_recent,
                    'Entries': len(lift_data)
                })
        
        return pd.DataFrame(progress)
    
    def export_analysis(self, output_file: str = None):
        """Export analyzed data with 1RM estimates."""
        if output_file is None:
            output_file = self.data_file.replace('.csv', '_with_1rm.csv')
        
        self.df.to_csv(output_file, index=False)
        print(f"💾 Exported analysis to: {output_file}")
        return output_file


def main():
    """Command line interface for 1RM analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze workout data and calculate 1RM estimates')
    parser.add_argument('data_file', help='Path to normalized workout data CSV')
    parser.add_argument('--lift', choices=['Squat', 'Bench', 'Deadlift'], help='Analyze specific lift')
    parser.add_argument('--export', action='store_true', help='Export data with 1RM calculations')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = WorkoutAnalyzer(args.data_file)
        analyzer.load_data()
        analyzer.calculate_1rm_estimates()
        
        if args.summary:
            print("\n📈 1RM SUMMARY")
            print("=" * 50)
            
            # Max 1RMs
            max_1rms = analyzer.get_max_1rm_by_lift()
            print("\n🏆 All-Time Max Estimated 1RMs:")
            for _, row in max_1rms.iterrows():
                print(f"  {row['Lift_Type']}: {row['Max_1RM']:.1f} lbs ({row['Date'].strftime('%Y-%m-%d')})")
            
            # Recent progress
            recent = analyzer.get_recent_progress(90)
            print(f"\n📅 Recent Progress (Last 90 days):")
            for _, row in recent.iterrows():
                print(f"  {row['Lift_Type']}: {row['Recent_Max_1RM']:.1f} lbs ({row['Entries']} sessions)")
        
        if args.lift:
            progression = analyzer.get_1rm_progression(args.lift)
            print(f"\n📊 {args.lift} 1RM Progression:")
            print(progression.tail(10).to_string(index=False))
        
        if args.export:
            analyzer.export_analysis()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 