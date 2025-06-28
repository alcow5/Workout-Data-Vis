#!/usr/bin/env python3
"""
Improved predictor that filters data for more accurate predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from predictive_analytics import PowerliftingPredictor


class ImprovedPowerliftingPredictor(PowerliftingPredictor):
    """Improved predictor with better data filtering."""
    
    def prepare_filtered_progression_data(self, lift_type: str, days_back: int = 180) -> pd.DataFrame:
        """Prepare clean progression data with intelligent filtering."""
        # Filter to specific lift and recent data
        cutoff_date = self.df['Date'].max() - timedelta(days=days_back)
        lift_data = self.df[
            (self.df['Lift_Type'] == lift_type) & 
            (self.df['Date'] >= cutoff_date) &
            (self.df['estimated_1rm'].notna()) &
            (self.df['estimated_1rm'] > 0)
        ].copy()
        
        if lift_data.empty:
            return pd.DataFrame()
        
        # Filter to main exercise only (most common exercise for this lift)
        main_exercise = lift_data['Exercise'].value_counts().index[0]
        lift_data = lift_data[lift_data['Exercise'] == main_exercise].copy()
        
        # Filter to strength-focused rep ranges (1-6 reps for more accurate 1RM)
        lift_data = lift_data[lift_data['Reps'] <= 6].copy()
        
        # Remove obvious outliers (1RM estimates more than 2 std dev from mean)
        mean_1rm = lift_data['estimated_1rm'].mean()
        std_1rm = lift_data['estimated_1rm'].std()
        lift_data = lift_data[
            abs(lift_data['estimated_1rm'] - mean_1rm) <= 2 * std_1rm
        ].copy()
        
        # Get best 1RM per day to smooth out daily variations
        daily_maxes = lift_data.groupby('Date')['estimated_1rm'].max().reset_index()
        daily_maxes = daily_maxes.sort_values('Date')
        
        # Add days since start for regression
        start_date = daily_maxes['Date'].min()
        daily_maxes['days_since_start'] = (daily_maxes['Date'] - start_date).dt.days
        
        print(f"🎯 Filtered to {len(daily_maxes)} training days using '{main_exercise}' with 1-6 reps")
        
        return daily_maxes
    
    def improved_trajectory_analysis(self, lift_type: str, forecast_days: int = 180) -> dict:
        """Improved trajectory analysis with better data filtering."""
        progression_data = self.prepare_filtered_progression_data(lift_type, days_back=180)
        
        if len(progression_data) < 5:
            return {
                'success': False,
                'message': f'Insufficient filtered data for {lift_type} (need 5+ data points, got {len(progression_data)})'
            }
        
        X = progression_data[['days_since_start']]
        y = progression_data['estimated_1rm']
        
        # Use simple linear regression first - it's often more reliable
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_score = linear_model.score(X, y)
        
        # Try polynomial only if linear fit is poor
        best_model = linear_model
        best_score = linear_score
        best_name = 'linear'
        
        if linear_score < 0.3:  # Only use polynomial if linear is really bad
            try:
                poly_model = Pipeline([
                    ('poly', PolynomialFeatures(2)),
                    ('linear', LinearRegression())
                ])
                poly_model.fit(X, y)
                poly_score = poly_model.score(X, y)
                
                if poly_score > linear_score + 0.1:  # Must be significantly better
                    best_model = poly_model
                    best_score = poly_score
                    best_name = 'polynomial_2'
            except:
                pass  # Stick with linear if polynomial fails
        
        # Generate forecast
        last_day = progression_data['days_since_start'].max()
        forecast_days_range = np.arange(last_day + 1, last_day + forecast_days + 1)
        forecast_X = forecast_days_range.reshape(-1, 1)
        forecast_y = best_model.predict(forecast_X)
        
        # Calculate trend metrics
        current_1rm = progression_data['estimated_1rm'].iloc[-1]
        forecast_1rm = forecast_y[-1]
        total_gain = forecast_1rm - current_1rm
        monthly_gain = total_gain / (forecast_days / 30.44)
        
        # Create forecast dates
        start_date = progression_data['Date'].min()
        forecast_dates = [start_date + timedelta(days=int(d)) for d in forecast_days_range]
        
        return {
            'success': True,
            'lift_type': lift_type,
            'model_type': best_name,
            'r2_score': best_score,
            'current_1rm': current_1rm,
            'forecast_1rm': forecast_1rm,
            'projected_gain': total_gain,
            'monthly_gain': monthly_gain,
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_y,
            'historical_data': progression_data,
            'trend_direction': 'increasing' if monthly_gain > 0 else 'decreasing',
            'data_points_used': len(progression_data),
            'main_exercise': progression_data.iloc[0] if len(progression_data) > 0 else None
        }
    
    def realistic_goal_timeline(self, lift_type: str, target_1rm: float) -> dict:
        """More realistic goal timeline based on filtered data."""
        trajectory = self.improved_trajectory_analysis(lift_type, forecast_days=730)
        
        if not trajectory['success']:
            return trajectory
        
        current_1rm = trajectory['current_1rm']
        
        if target_1rm <= current_1rm:
            return {
                'success': True,
                'message': f'Target {target_1rm} lbs already achieved! Current: {current_1rm:.1f} lbs',
                'target_1rm': target_1rm,
                'current_1rm': current_1rm,
                'already_achieved': True
            }
        
        # More conservative projection - use linear trend even if polynomial was selected
        X = trajectory['historical_data'][['days_since_start']]
        y = trajectory['historical_data']['estimated_1rm']
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        slope = linear_model.coef_[0]
        
        if slope <= 0:
            return {
                'success': True,
                'message': f'Based on recent trend, target {target_1rm} lbs may require training adjustments',
                'target_1rm': target_1rm,
                'current_1rm': current_1rm,
                'reachable': False,
                'monthly_gain': slope * 30.44,
                'recommendation': 'Consider increasing training frequency or intensity'
            }
        
        # Calculate days to reach target
        days_needed = (target_1rm - current_1rm) / slope
        target_date = datetime.now() + timedelta(days=days_needed)
        months_to_target = days_needed / 30.44
        
        return {
            'success': True,
            'target_1rm': target_1rm,
            'current_1rm': current_1rm,
            'target_date': target_date,
            'days_to_target': days_needed,
            'months_to_target': months_to_target,
            'reachable': True,
            'monthly_gain': slope * 30.44,
            'confidence': 'High' if trajectory['r2_score'] > 0.5 else 'Medium' if trajectory['r2_score'] > 0.2 else 'Low'
        }


def demo_improved_predictions():
    """Demo the improved predictions."""
    # Load data
    df = pd.read_csv('AlexWOLog_normalized_fixed_validated.csv')
    
    # Compare old vs new predictions
    print("🔄 COMPARING PREDICTIONS: OLD vs IMPROVED")
    print("=" * 60)
    
    old_predictor = PowerliftingPredictor(df)
    new_predictor = ImprovedPowerliftingPredictor(df)
    
    for lift in ['Squat', 'Bench', 'Deadlift']:
        print(f"\n💪 {lift.upper()} COMPARISON:")
        print("-" * 30)
        
        # Old prediction
        old_traj = old_predictor.trajectory_analysis(lift)
        if old_traj['success']:
            print(f"❌ OLD: {old_traj['current_1rm']:.1f} lbs → {old_traj['monthly_gain']:+.1f} lbs/month")
        else:
            print(f"❌ OLD: Failed to predict")
        
        # New prediction
        new_traj = new_predictor.improved_trajectory_analysis(lift)
        if new_traj['success']:
            print(f"✅ NEW: {new_traj['current_1rm']:.1f} lbs → {new_traj['monthly_gain']:+.1f} lbs/month")
            print(f"    Using {new_traj['data_points_used']} data points (R² = {new_traj['r2_score']:.3f})")
        else:
            print(f"✅ NEW: {new_traj['message']}")
    
    # Test goal timeline for squat
    print(f"\n🎯 GOAL TIMELINE TEST (600 lb Squat):")
    print("-" * 40)
    
    old_goal = old_predictor.goal_timeline('Squat', 600)
    new_goal = new_predictor.realistic_goal_timeline('Squat', 600)
    
    if old_goal['success'] and not old_goal.get('already_achieved'):
        if old_goal.get('reachable'):
            print(f"❌ OLD: {old_goal['months_to_target']:.1f} months")
        else:
            print(f"❌ OLD: Not reachable")
    
    if new_goal['success'] and not new_goal.get('already_achieved'):
        if new_goal.get('reachable'):
            print(f"✅ NEW: {new_goal['months_to_target']:.1f} months ({new_goal.get('confidence', 'Unknown')} confidence)")
        else:
            print(f"✅ NEW: {new_goal.get('message', 'Not reachable')}")


if __name__ == "__main__":
    demo_improved_predictions() 