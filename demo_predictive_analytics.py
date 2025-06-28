#!/usr/bin/env python3
"""
Demo script showing the new predictive analytics features
"""

import pandas as pd
from predictive_analytics import PowerliftingPredictor
from datetime import datetime, timedelta
import argparse


def demo_progress_forecasting(predictor):
    """Demo progress forecasting capabilities."""
    print("\n🔮 PROGRESS FORECASTING DEMO")
    print("=" * 50)
    
    for lift in ['Squat', 'Bench', 'Deadlift']:
        print(f"\n📈 {lift} Trajectory Analysis:")
        
        trajectory = predictor.trajectory_analysis(lift, forecast_days=180)
        if trajectory['success']:
            print(f"  Current 1RM: {trajectory['current_1rm']:.1f} lbs")
            print(f"  6-Month Forecast: {trajectory['forecast_1rm']:.1f} lbs")
            print(f"  Projected Gain: {trajectory['projected_gain']:+.1f} lbs")
            print(f"  Monthly Rate: {trajectory['monthly_gain']:+.1f} lbs/month")
            print(f"  Trend: {trajectory['trend_direction']} (R² = {trajectory['r2_score']:.3f})")
        else:
            print(f"  ❌ {trajectory['message']}")


def demo_goal_timeline(predictor):
    """Demo goal timeline predictions."""
    print("\n🎯 GOAL TIMELINE DEMO")
    print("=" * 50)
    
    # Example goals for each lift
    goals = {
        'Squat': 600,
        'Bench': 450, 
        'Deadlift': 650
    }
    
    for lift, target in goals.items():
        print(f"\n🏆 {lift} Goal: {target} lbs")
        
        timeline = predictor.goal_timeline(lift, target)
        if timeline['success']:
            if timeline.get('already_achieved'):
                print(f"  🎉 Already achieved! Current: {timeline['current_1rm']:.1f} lbs")
            elif timeline.get('reachable'):
                print(f"  📅 Target Date: {timeline['target_date'].strftime('%B %Y')}")
                print(f"  ⏰ Time to Goal: {timeline['months_to_target']:.1f} months")
                print(f"  📈 Required Rate: {timeline['monthly_gain']:.1f} lbs/month")
            else:
                print(f"  ⚠️ Not reachable within 2 years")
                print(f"  💡 Current rate: {timeline['monthly_gain']:.1f} lbs/month")
        else:
            print(f"  ❌ Cannot calculate timeline")


def demo_meet_predictions(predictor):
    """Demo meet attempt predictions."""
    print("\n🏆 MEET PREDICTION DEMO")
    print("=" * 50)
    
    # Example meet in 12 weeks
    meet_date = datetime.now() + timedelta(days=84)
    print(f"📅 Meet Date: {meet_date.strftime('%B %d, %Y')} (12 weeks from now)")
    
    for lift in ['Squat', 'Bench', 'Deadlift']:
        print(f"\n💪 {lift} Attempt Suggestions:")
        
        prediction = predictor.meet_attempt_prediction(lift, meet_date)
        if prediction['success']:
            print(f"  Predicted 1RM: {prediction['predicted_1rm']:.1f} lbs")
            print(f"  🥉 Opener (90%): {prediction['opener']:.1f} lbs")
            print(f"  🥈 Second (97%): {prediction['second_attempt']:.1f} lbs")
            print(f"  🥇 Third (103%): {prediction['third_attempt']:.1f} lbs")
            print(f"  Confidence: {prediction['confidence']:.1%}")
        else:
            print(f"  ❌ Cannot predict attempts")


def demo_training_optimization(predictor):
    """Demo training optimization analysis."""
    print("\n🧠 TRAINING OPTIMIZATION DEMO")
    print("=" * 50)
    
    for lift in ['Squat', 'Bench', 'Deadlift']:
        print(f"\n🏋️ {lift} Optimization:")
        
        # Rep range effectiveness
        rep_analysis = predictor.optimal_rep_ranges(lift)
        if rep_analysis['success'] and rep_analysis['ranked_effectiveness']:
            print(f"  🎯 Most Effective Rep Range:")
            best_range = rep_analysis['ranked_effectiveness'][0]
            print(f"    {best_range[0]} (correlation: {best_range[1]['correlation_with_progress']:.3f})")
        
        # Optimal frequency
        freq_analysis = predictor.optimal_frequency(lift)
        if freq_analysis['success']:
            print(f"  📅 Optimal Frequency: {freq_analysis['optimal_frequency']}x per week")
            print(f"    Average: {freq_analysis['average_frequency']:.1f}x per week")
        
        # Volume analysis
        volume_analysis = predictor.volume_sweet_spot(lift)
        if volume_analysis['success']:
            print(f"  💪 Volume Recommendation: {volume_analysis['volume_recommendation']}")
            if volume_analysis.get('optimal_volume'):
                print(f"    Target: {volume_analysis['optimal_volume']:.0f} weekly volume")


def main():
    """Run the predictive analytics demo."""
    parser = argparse.ArgumentParser(description='Demo predictive analytics features')
    parser.add_argument('--file', '-f', help='Workout data file', 
                       default='AlexWOLog_normalized_fixed_validated.csv')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"📊 Loading data from: {args.file}")
        df = pd.read_csv(args.file)
        
        print(f"✅ Loaded {len(df)} workout entries")
        print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Initialize predictor
        predictor = PowerliftingPredictor(df)
        
        # Run demos
        demo_progress_forecasting(predictor)
        demo_goal_timeline(predictor)
        demo_meet_predictions(predictor)
        demo_training_optimization(predictor)
        
        print("\n" + "=" * 50)
        print("🚀 Try the full interactive experience in the web dashboard!")
        print("   Run: streamlit run workout_dashboard.py")
        print("   Then navigate to the '🔮 Predictive Analytics' section")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{args.file}'")
        print("Available files:")
        import glob
        csv_files = glob.glob("*normalized*.csv")
        for file in csv_files:
            print(f"  - {file}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 