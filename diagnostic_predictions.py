#!/usr/bin/env python3
"""
Diagnostic script to investigate prediction issues
"""

import pandas as pd
import numpy as np
from predictive_analytics import PowerliftingPredictor
from one_rm_calculator import WorkoutAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import argparse


def analyze_recent_data(df, lift_type, months_back=6):
    """Analyze recent training data to understand what the model is seeing."""
    print(f"\n🔍 ANALYZING RECENT {lift_type.upper()} DATA")
    print("=" * 50)
    
    # Filter to lift and recent data
    recent_cutoff = datetime.now() - timedelta(days=months_back * 30)
    lift_data = df[
        (df['Lift_Type'] == lift_type) & 
        (df['Date'] >= recent_cutoff) &
        (df['estimated_1rm'].notna()) &
        (df['estimated_1rm'] > 0)
    ].copy()
    
    if lift_data.empty:
        print(f"❌ No recent {lift_type} data found")
        return
    
    # Show data summary
    print(f"📊 Found {len(lift_data)} recent {lift_type} entries")
    print(f"📅 Date range: {lift_data['Date'].min().strftime('%Y-%m-%d')} to {lift_data['Date'].max().strftime('%Y-%m-%d')}")
    
    # Group by date and get daily maxes
    daily_maxes = lift_data.groupby(lift_data['Date'].dt.date).agg({
        'estimated_1rm': 'max',
        'Load': 'max',
        'Exercise': 'first'
    }).reset_index()
    daily_maxes['Date'] = pd.to_datetime(daily_maxes['Date'])
    daily_maxes = daily_maxes.sort_values('Date')
    
    print(f"\n📈 RECENT PROGRESSION:")
    print("Date       | Est 1RM | Max Load | Exercise")
    print("-" * 45)
    for _, row in daily_maxes.tail(10).iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')} | {row['estimated_1rm']:6.1f}  | {row['Load']:7.1f}  | {row['Exercise']}")
    
    # Calculate trend
    if len(daily_maxes) >= 3:
        recent_trend = np.polyfit(np.arange(len(daily_maxes)), daily_maxes['estimated_1rm'], 1)[0]
        print(f"\n📊 TREND ANALYSIS:")
        print(f"  Recent trend: {recent_trend:+.2f} lbs per session")
        print(f"  Monthly equivalent: {recent_trend * 8:+.1f} lbs/month (assuming 2x/week)")
        
        # Show highest and lowest points
        max_idx = daily_maxes['estimated_1rm'].idxmax()
        min_idx = daily_maxes['estimated_1rm'].idxmin()
        print(f"  Highest: {daily_maxes.loc[max_idx, 'estimated_1rm']:.1f} lbs on {daily_maxes.loc[max_idx, 'Date'].strftime('%Y-%m-%d')}")
        print(f"  Lowest:  {daily_maxes.loc[min_idx, 'estimated_1rm']:.1f} lbs on {daily_maxes.loc[min_idx, 'Date'].strftime('%Y-%m-%d')}")
        
        # Check for potential data issues
        print(f"\n🔍 DATA QUALITY CHECKS:")
        
        # Check for exercise variations
        unique_exercises = lift_data['Exercise'].unique()
        if len(unique_exercises) > 1:
            print(f"⚠️  Multiple exercises found: {', '.join(unique_exercises)}")
            print("    This could affect 1RM calculations if exercises have different strength levels")
        
        # Check for unusual load patterns
        load_std = daily_maxes['Load'].std()
        load_mean = daily_maxes['Load'].mean()
        if load_std > load_mean * 0.3:  # High variability
            print(f"⚠️  High load variability: std={load_std:.1f}, mean={load_mean:.1f}")
            print("    This suggests mixed training phases (strength vs technique work)")
        
        # Check for rep range variations
        rep_analysis = lift_data.groupby('Reps')['estimated_1rm'].agg(['count', 'mean']).round(1)
        print(f"\n📋 REP RANGE BREAKDOWN:")
        print("Reps | Count | Avg 1RM")
        print("-" * 20)
        for reps, data in rep_analysis.iterrows():
            print(f"{int(reps):4d} | {int(data['count']):5d} | {data['mean']:7.1f}")
        
        # Check for recent deloads or technique work
        recent_week = daily_maxes.tail(7)  # Last 7 sessions
        earlier_week = daily_maxes.head(7) if len(daily_maxes) >= 14 else daily_maxes.head(len(daily_maxes)//2)
        
        if not recent_week.empty and not earlier_week.empty:
            recent_avg = recent_week['estimated_1rm'].mean()
            earlier_avg = earlier_week['estimated_1rm'].mean()
            change = recent_avg - earlier_avg
            
            print(f"\n🔄 RECENT vs EARLIER COMPARISON:")
            print(f"  Recent sessions avg: {recent_avg:.1f} lbs")
            print(f"  Earlier sessions avg: {earlier_avg:.1f} lbs")
            print(f"  Change: {change:+.1f} lbs")
            
            if change < -20:
                print("⚠️  Significant recent decrease detected - possibly deload/technique work")
            elif change > 20:
                print("✅ Recent improvement detected!")


def investigate_model_behavior(predictor, lift_type):
    """Investigate why the model is predicting decreases."""
    print(f"\n🤖 MODEL BEHAVIOR ANALYSIS - {lift_type.upper()}")
    print("=" * 50)
    
    # Try different time windows
    for days_back in [90, 180, 365]:
        print(f"\n📊 Using {days_back} days of data:")
        trajectory = predictor.trajectory_analysis(lift_type, forecast_days=180)
        
        if trajectory['success']:
            print(f"  Model: {trajectory['model_type']}")
            print(f"  R² Score: {trajectory['r2_score']:.3f}")
            print(f"  Current 1RM: {trajectory['current_1rm']:.1f} lbs")
            print(f"  Monthly gain: {trajectory['monthly_gain']:+.1f} lbs/month")
            print(f"  Data points: {len(trajectory['historical_data'])}")
        else:
            print(f"  ❌ {trajectory['message']}")


def suggest_fixes(df, lift_type):
    """Suggest potential fixes for the prediction issues."""
    print(f"\n💡 SUGGESTED FIXES FOR {lift_type.upper()}")
    print("=" * 50)
    
    lift_data = df[df['Lift_Type'] == lift_type].copy()
    
    print("1. **Check Exercise Consistency**")
    exercises = lift_data['Exercise'].value_counts()
    print(f"   Most common: {exercises.index[0]} ({exercises.iloc[0]} entries)")
    if len(exercises) > 1:
        print(f"   Also using: {', '.join(exercises.index[1:3])}")
        print("   → Consider filtering to main exercise only")
    
    print("\n2. **Check Recent Data Quality**")
    recent_data = lift_data[lift_data['Date'] >= (datetime.now() - timedelta(days=60))]
    if not recent_data.empty:
        recent_loads = recent_data['Load'].describe()
        print(f"   Recent max load: {recent_loads['max']:.1f} lbs")
        print(f"   Recent avg load: {recent_loads['mean']:.1f} lbs")
        
        # Check for low loads recently
        overall_max = lift_data['Load'].max()
        if recent_loads['max'] < overall_max * 0.85:
            print("   ⚠️  Recent max is <85% of all-time max")
            print("   → This suggests deload or technique work")
    
    print("\n3. **Recommended Model Adjustments**")
    print("   → Use only last 3-6 months of data")
    print("   → Filter to main exercise variant only") 
    print("   → Exclude obvious deload weeks")
    print("   → Focus on rep ranges 1-5 for strength assessment")


def main():
    """Run diagnostic analysis."""
    parser = argparse.ArgumentParser(description='Diagnose prediction issues')
    parser.add_argument('--file', '-f', help='Workout data file', 
                       default='AlexWOLog_normalized_fixed_validated.csv')
    parser.add_argument('--lift', '-l', help='Lift to analyze', 
                       choices=['Squat', 'Bench', 'Deadlift'], default='Squat')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"📊 Loading data from: {args.file}")
        df = pd.read_csv(args.file)
        
        # Initialize predictor (this will calculate 1RM estimates)
        predictor = PowerliftingPredictor(df)
        
        # Run analysis
        analyze_recent_data(predictor.df, args.lift, months_back=6)
        investigate_model_behavior(predictor, args.lift)
        suggest_fixes(predictor.df, args.lift)
        
        print(f"\n" + "=" * 50)
        print("🎯 SUMMARY")
        print("=" * 50)
        print("The predictions might be wrong due to:")
        print("• Recent deload or technique work phases")
        print("• Mixed exercise variations in the data")
        print("• Model using too much historical data")
        print("• Data quality issues or logging inconsistencies")
        print("\nLet's create a more accurate model with filters!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 