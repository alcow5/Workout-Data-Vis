#!/usr/bin/env python3
"""
Predictive Analytics for Powerlifting Training
Provides forecasting, trajectory analysis, and training optimization insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


class PowerliftingPredictor:
    """Advanced predictive analytics for powerlifting training data."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with training data."""
        self.df = data.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Calculate 1RM estimates if not already present
        if 'estimated_1rm' not in self.df.columns:
            self._calculate_1rm_estimates()
    
    def _calculate_1rm_estimates(self):
        """Calculate 1RM estimates for the dataset."""
        from one_rm_calculator import OneRMCalculator
        
        calculator = OneRMCalculator()
        
        def calc_1rm(row):
            """Calculate best 1RM estimate for a row."""
            if pd.isna(row['Load']) or pd.isna(row['Reps']) or row['Load'] <= 0 or row['Reps'] <= 0:
                return None
            
            return calculator.best_estimate(
                weight=row['Load'], 
                reps=row['Reps'], 
                rpe=row.get('RPE')
            )
        
        # Calculate 1RM estimates
        self.df['estimated_1rm'] = self.df.apply(calc_1rm, axis=1)
        
        # Filter out invalid estimates
        self.df = self.df[
            (self.df['estimated_1rm'].notna()) & 
            (self.df['estimated_1rm'] > 0) &
            (self.df['estimated_1rm'] < 1000)  # Remove unrealistic estimates
        ].copy()
        
    def prepare_progression_data(self, lift_type: str, days_back: int = 365) -> pd.DataFrame:
        """Prepare clean progression data for analysis."""
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
        
        # Get best 1RM per day
        daily_maxes = lift_data.groupby('Date')['estimated_1rm'].max().reset_index()
        daily_maxes = daily_maxes.sort_values('Date')
        
        # Add days since start for regression
        start_date = daily_maxes['Date'].min()
        daily_maxes['days_since_start'] = (daily_maxes['Date'] - start_date).dt.days
        
        return daily_maxes
    
    def trajectory_analysis(self, lift_type: str, forecast_days: int = 180) -> Dict:
        """Analyze trajectory and forecast future performance."""
        progression_data = self.prepare_progression_data(lift_type)
        
        if len(progression_data) < 5:  # Need minimum data points
            return {
                'success': False,
                'message': f'Insufficient data for {lift_type} trajectory analysis'
            }
        
        X = progression_data[['days_since_start']]
        y = progression_data['estimated_1rm']
        
        # Try multiple models and pick the best
        models = {
            'linear': LinearRegression(),
            'polynomial_2': Pipeline([
                ('poly', PolynomialFeatures(2)),
                ('linear', LinearRegression())
            ]),
            'polynomial_3': Pipeline([
                ('poly', PolynomialFeatures(3)),
                ('linear', LinearRegression())
            ])
        }
        
        best_model = None
        best_score = -np.inf
        best_name = 'linear'
        
        for name, model in models.items():
            try:
                model.fit(X, y)
                score = model.score(X, y)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
            except:
                continue
        
        if best_model is None:
            return {
                'success': False,
                'message': f'Could not fit model for {lift_type}'
            }
        
        # Generate forecast
        last_day = progression_data['days_since_start'].max()
        forecast_days_range = np.arange(last_day + 1, last_day + forecast_days + 1)
        forecast_X = forecast_days_range.reshape(-1, 1)
        forecast_y = best_model.predict(forecast_X)
        
        # Create forecast dates
        start_date = progression_data['Date'].min()
        forecast_dates = [start_date + timedelta(days=int(d)) for d in forecast_days_range]
        
        # Calculate trend metrics
        current_1rm = progression_data['estimated_1rm'].iloc[-1]
        forecast_1rm = forecast_y[-1]
        total_gain = forecast_1rm - current_1rm
        monthly_gain = total_gain / (forecast_days / 30.44)  # Average days per month
        
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
            'trend_direction': 'increasing' if monthly_gain > 0 else 'decreasing'
        }
    
    def goal_timeline(self, lift_type: str, target_1rm: float) -> Dict:
        """Estimate when you'll reach a target 1RM."""
        trajectory = self.trajectory_analysis(lift_type, forecast_days=730)  # 2 years
        
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
        
        # Find when forecast crosses target
        forecast_values = trajectory['forecast_values']
        forecast_dates = trajectory['forecast_dates']
        
        # Find first date where forecast >= target
        target_indices = np.where(forecast_values >= target_1rm)[0]
        
        if len(target_indices) == 0:
            return {
                'success': True,
                'message': f'Target {target_1rm} lbs not reachable within 2-year forecast',
                'target_1rm': target_1rm,
                'current_1rm': current_1rm,
                'reachable': False,
                'monthly_gain': trajectory['monthly_gain']
            }
        
        target_date = forecast_dates[target_indices[0]]
        days_to_target = (target_date - datetime.now()).days
        months_to_target = days_to_target / 30.44
        
        return {
            'success': True,
            'target_1rm': target_1rm,
            'current_1rm': current_1rm,
            'target_date': target_date,
            'days_to_target': days_to_target,
            'months_to_target': months_to_target,
            'reachable': True,
            'monthly_gain': trajectory['monthly_gain']
        }
    
    def meet_attempt_prediction(self, lift_type: str, meet_date: str = None) -> Dict:
        """Predict competition attempts based on training."""
        if meet_date is None:
            meet_date = datetime.now() + timedelta(days=84)  # 12 weeks from now
        else:
            meet_date = pd.to_datetime(meet_date)
        
        days_to_meet = (meet_date - datetime.now()).days
        
        if days_to_meet < 0:
            return {
                'success': False,
                'message': 'Meet date is in the past'
            }
        
        # Get trajectory forecast
        trajectory = self.trajectory_analysis(lift_type, forecast_days=days_to_meet + 30)
        
        if not trajectory['success']:
            return trajectory
        
        # Find predicted 1RM at meet date
        forecast_dates = trajectory['forecast_dates']
        forecast_values = trajectory['forecast_values']
        
        # Find closest forecast date to meet
        meet_date_differences = [abs((date - meet_date).days) for date in forecast_dates]
        closest_idx = np.argmin(meet_date_differences)
        predicted_1rm = forecast_values[closest_idx]
        
        # Conservative attempt selection (typical powerlifting strategy)
        opener = predicted_1rm * 0.90  # 90% for opener
        second = predicted_1rm * 0.97  # 97% for second
        third = predicted_1rm * 1.03   # 103% for third (aggressive)
        
        return {
            'success': True,
            'meet_date': meet_date,
            'days_to_meet': days_to_meet,
            'predicted_1rm': predicted_1rm,
            'opener': opener,
            'second_attempt': second,
            'third_attempt': third,
            'confidence': min(trajectory['r2_score'], 0.95)  # Cap confidence at 95%
        }
    
    def optimal_rep_ranges(self, lift_type: str, window_days: int = 90) -> Dict:
        """Analyze which rep ranges correlate with best progress."""
        lift_data = self.df[self.df['Lift_Type'] == lift_type].copy()
        
        if lift_data.empty:
            return {'success': False, 'message': f'No data for {lift_type}'}
        
        # Get recent data and calculate progress windows
        lift_data = lift_data.sort_values('Date')
        
        # Define rep range categories
        rep_ranges = {
            '1-3 (Strength)': (1, 3),
            '4-6 (Power)': (4, 6), 
            '7-10 (Hypertrophy)': (7, 10),
            '11+ (Endurance)': (11, 50)
        }
        
        rep_analysis = {}
        
        for range_name, (min_reps, max_reps) in rep_ranges.items():
            range_data = lift_data[
                (lift_data['Reps'] >= min_reps) & 
                (lift_data['Reps'] <= max_reps) &
                (lift_data['estimated_1rm'].notna())
            ]
            
            if len(range_data) < 5:  # Need minimum data
                continue
            
            # Calculate progress correlation
            range_data['days_since_start'] = (range_data['Date'] - range_data['Date'].min()).dt.days
            
            if range_data['days_since_start'].var() > 0:  # Need time variation
                correlation = np.corrcoef(range_data['days_since_start'], range_data['estimated_1rm'])[0,1]
            else:
                correlation = 0
            
            # Calculate average 1RM and volume
            avg_1rm = range_data['estimated_1rm'].mean()
            total_volume = (range_data['Sets'] * range_data['Reps'] * range_data['Load']).sum()
            session_count = len(range_data)
            
            rep_analysis[range_name] = {
                'correlation_with_progress': correlation,
                'average_1rm': avg_1rm,
                'total_volume': total_volume,
                'session_count': session_count,
                'avg_volume_per_session': total_volume / session_count if session_count > 0 else 0
            }
        
        # Rank rep ranges by effectiveness
        ranked_ranges = sorted(rep_analysis.items(), 
                             key=lambda x: x[1]['correlation_with_progress'], 
                             reverse=True)
        
        return {
            'success': True,
            'lift_type': lift_type,
            'rep_analysis': rep_analysis,
            'ranked_effectiveness': ranked_ranges,
            'recommendation': ranked_ranges[0][0] if ranked_ranges else None
        }
    
    def optimal_frequency(self, lift_type: str) -> Dict:
        """Analyze optimal training frequency for each lift."""
        lift_data = self.df[self.df['Lift_Type'] == lift_type].copy()
        
        if lift_data.empty:
            return {'success': False, 'message': f'No data for {lift_type}'}
        
        # Group by week and count sessions
        lift_data['Week'] = lift_data['Date'].dt.to_period('W')
        weekly_stats = lift_data.groupby('Week').agg({
            'Date': 'nunique',  # Sessions per week
            'estimated_1rm': 'max'  # Best 1RM that week
        }).reset_index()
        
        weekly_stats.columns = ['Week', 'Sessions', 'Best_1RM']
        weekly_stats = weekly_stats[weekly_stats['Sessions'] > 0]
        
        if len(weekly_stats) < 4:
            return {'success': False, 'message': 'Insufficient weekly data'}
        
        # Analyze progress by frequency
        frequency_analysis = {}
        
        for freq in range(1, min(8, weekly_stats['Sessions'].max() + 1)):  # Up to 7x per week
            freq_weeks = weekly_stats[weekly_stats['Sessions'] == freq]
            
            if len(freq_weeks) < 2:
                continue
            
            # Calculate average 1RM for this frequency
            avg_1rm = freq_weeks['Best_1RM'].mean()
            week_count = len(freq_weeks)
            
            # Calculate progress trend within this frequency
            if len(freq_weeks) >= 3:
                weeks_numeric = np.arange(len(freq_weeks))
                correlation = np.corrcoef(weeks_numeric, freq_weeks['Best_1RM'])[0,1]
            else:
                correlation = 0
            
            frequency_analysis[freq] = {
                'average_1rm': avg_1rm,
                'week_count': week_count,
                'progress_correlation': correlation,
                'sample_size': 'Good' if week_count >= 4 else 'Limited'
            }
        
        # Find optimal frequency
        if frequency_analysis:
            optimal_freq = max(frequency_analysis.keys(), 
                             key=lambda x: frequency_analysis[x]['average_1rm'])
        else:
            optimal_freq = None
        
        return {
            'success': True,
            'lift_type': lift_type,
            'frequency_analysis': frequency_analysis,
            'optimal_frequency': optimal_freq,
            'average_frequency': weekly_stats['Sessions'].mean(),
            'frequency_range': f"{weekly_stats['Sessions'].min()}-{weekly_stats['Sessions'].max()}"
        }
    
    def volume_sweet_spot(self, lift_type: str) -> Dict:
        """Find the ideal weekly volume for each lift."""
        lift_data = self.df[self.df['Lift_Type'] == lift_type].copy()
        
        if lift_data.empty:
            return {'success': False, 'message': f'No data for {lift_type}'}
        
        # Calculate volume per session
        lift_data['Volume'] = lift_data['Sets'] * lift_data['Reps'] * lift_data['Load']
        lift_data['Week'] = lift_data['Date'].dt.to_period('W')
        
        # Group by week
        weekly_volume = lift_data.groupby('Week').agg({
            'Volume': 'sum',
            'estimated_1rm': 'max'
        }).reset_index()
        
        weekly_volume.columns = ['Week', 'Weekly_Volume', 'Best_1RM']
        weekly_volume = weekly_volume[weekly_volume['Weekly_Volume'] > 0]
        
        if len(weekly_volume) < 4:
            return {'success': False, 'message': 'Insufficient weekly volume data'}
        
        # Create volume buckets for analysis
        volume_percentiles = np.percentile(weekly_volume['Weekly_Volume'], [25, 50, 75])
        
        volume_buckets = {
            'Low (Bottom 25%)': (0, volume_percentiles[0]),
            'Moderate (25-50%)': (volume_percentiles[0], volume_percentiles[1]),
            'High (50-75%)': (volume_percentiles[1], volume_percentiles[2]),
            'Very High (Top 25%)': (volume_percentiles[2], np.inf)
        }
        
        bucket_analysis = {}
        
        for bucket_name, (min_vol, max_vol) in volume_buckets.items():
            bucket_data = weekly_volume[
                (weekly_volume['Weekly_Volume'] >= min_vol) & 
                (weekly_volume['Weekly_Volume'] < max_vol)
            ]
            
            if len(bucket_data) < 2:
                continue
            
            bucket_analysis[bucket_name] = {
                'average_1rm': bucket_data['Best_1RM'].mean(),
                'week_count': len(bucket_data),
                'avg_volume': bucket_data['Weekly_Volume'].mean(),
                'volume_range': f"{bucket_data['Weekly_Volume'].min():.0f}-{bucket_data['Weekly_Volume'].max():.0f}"
            }
        
        # Find optimal volume bucket
        if bucket_analysis:
            optimal_bucket = max(bucket_analysis.keys(), 
                               key=lambda x: bucket_analysis[x]['average_1rm'])
            optimal_volume = bucket_analysis[optimal_bucket]['avg_volume']
        else:
            optimal_bucket = None
            optimal_volume = None
        
        # Calculate volume-performance correlation
        if len(weekly_volume) >= 3:
            volume_correlation = np.corrcoef(weekly_volume['Weekly_Volume'], 
                                           weekly_volume['Best_1RM'])[0,1]
        else:
            volume_correlation = 0
        
        return {
            'success': True,
            'lift_type': lift_type,
            'volume_analysis': bucket_analysis,
            'optimal_volume_bucket': optimal_bucket,
            'optimal_volume': optimal_volume,
            'volume_correlation': volume_correlation,
            'average_weekly_volume': weekly_volume['Weekly_Volume'].mean(),
            'volume_recommendation': 'Increase' if volume_correlation > 0.3 else 'Maintain' if volume_correlation > -0.3 else 'Decrease'
        }


def main():
    """Command line interface for predictive analytics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictive analytics for powerlifting data')
    parser.add_argument('data_file', help='Path to workout data CSV')
    parser.add_argument('--lift', choices=['Squat', 'Bench', 'Deadlift'], required=True)
    parser.add_argument('--forecast', action='store_true', help='Show trajectory forecast')
    parser.add_argument('--goal', type=float, help='Target 1RM for timeline analysis')
    parser.add_argument('--meet-date', help='Meet date for attempt prediction (YYYY-MM-DD)')
    parser.add_argument('--optimize', action='store_true', help='Show training optimization analysis')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = pd.read_csv(args.data_file)
        predictor = PowerliftingPredictor(df)
        
        print(f"\n🔮 PREDICTIVE ANALYTICS - {args.lift}")
        print("=" * 50)
        
        if args.forecast:
            trajectory = predictor.trajectory_analysis(args.lift)
            if trajectory['success']:
                print(f"\n📈 TRAJECTORY FORECAST:")
                print(f"  Current 1RM: {trajectory['current_1rm']:.1f} lbs")
                print(f"  6-Month Forecast: {trajectory['forecast_1rm']:.1f} lbs")
                print(f"  Projected Gain: {trajectory['projected_gain']:+.1f} lbs")
                print(f"  Monthly Rate: {trajectory['monthly_gain']:+.1f} lbs/month")
                print(f"  Model: {trajectory['model_type']} (R² = {trajectory['r2_score']:.3f})")
        
        if args.goal:
            timeline = predictor.goal_timeline(args.lift, args.goal)
            if timeline['success']:
                if timeline.get('already_achieved'):
                    print(f"\n🎯 GOAL TIMELINE: {timeline['message']}")
                elif timeline.get('reachable'):
                    print(f"\n🎯 GOAL TIMELINE ({args.goal} lbs):")
                    print(f"  Target Date: {timeline['target_date'].strftime('%Y-%m-%d')}")
                    print(f"  Time to Goal: {timeline['months_to_target']:.1f} months")
                else:
                    print(f"\n🎯 GOAL TIMELINE: {timeline['message']}")
        
        if args.meet_date:
            prediction = predictor.meet_attempt_prediction(args.lift, args.meet_date)
            if prediction['success']:
                print(f"\n🏆 MEET PREDICTION:")
                print(f"  Meet Date: {prediction['meet_date'].strftime('%Y-%m-%d')}")
                print(f"  Predicted 1RM: {prediction['predicted_1rm']:.1f} lbs")
                print(f"  Opener: {prediction['opener']:.1f} lbs")
                print(f"  Second: {prediction['second_attempt']:.1f} lbs")
                print(f"  Third: {prediction['third_attempt']:.1f} lbs")
        
        if args.optimize:
            print(f"\n🧠 TRAINING OPTIMIZATION:")
            
            # Rep ranges
            rep_analysis = predictor.optimal_rep_ranges(args.lift)
            if rep_analysis['success']:
                print(f"\n  📊 Most Effective Rep Ranges:")
                for i, (range_name, data) in enumerate(rep_analysis['ranked_effectiveness'][:3]):
                    print(f"    {i+1}. {range_name} (correlation: {data['correlation_with_progress']:.3f})")
            
            # Frequency
            freq_analysis = predictor.optimal_frequency(args.lift)
            if freq_analysis['success']:
                print(f"\n  📅 Optimal Frequency: {freq_analysis['optimal_frequency']}x per week")
                print(f"    Average: {freq_analysis['average_frequency']:.1f}x per week")
            
            # Volume
            volume_analysis = predictor.volume_sweet_spot(args.lift)
            if volume_analysis['success']:
                print(f"\n  💪 Volume Sweet Spot: {volume_analysis['optimal_volume_bucket']}")
                if volume_analysis['optimal_volume']:
                    print(f"    Target: {volume_analysis['optimal_volume']:.0f} total volume/week")
                print(f"    Recommendation: {volume_analysis['volume_recommendation']} volume")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 