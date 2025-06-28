#!/usr/bin/env python3
"""
Workout Analysis Dashboard
Beautiful web interface for analyzing workout data and 1RM progression
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

from one_rm_calculator import WorkoutAnalyzer, OneRMCalculator
from meet_data_integrator import OpenPowerliftingIntegrator
from predictive_analytics import PowerliftingPredictor
from improved_predictor import ImprovedPowerliftingPredictor


# Page configuration
st.set_page_config(
    page_title="💪 Workout Analysis Dashboard",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        padding: 0.75rem !important;
        border-radius: 0.5rem !important;
        margin: 0.25rem 0 !important;
    }
    .stMetric > div {
        color: #212529 !important;
    }
    .stMetric label {
        color: #495057 !important;
        font-weight: 600 !important;
    }
    .stMetric .metric-value {
        color: #212529 !important;
        font-weight: bold !important;
    }
    .prediction-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 1px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #0d47a1;
    }
    .prediction-card h4 {
        color: #0d47a1 !important;
        margin: 0 0 0.5rem 0;
    }
    .prediction-card p {
        color: #1565c0 !important;
        margin: 0.25rem 0;
    }
    .optimization-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border: 1px solid #9c27b0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #4a148c;
    }
    .optimization-card h4 {
        color: #4a148c !important;
        margin: 0 0 0.5rem 0;
    }
    .optimization-card p {
        color: #6a1b9a !important;
        margin: 0.25rem 0;
    }
    .lift-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        margin: 0.25rem 0;
        text-align: center;
    }
    .meet-point {
        background: gold;
        border: 2px solid #FFD700;
        border-radius: 50%;
    }
    .compact-header {
        margin-bottom: 0.5rem !important;
    }
    /* Fix for tab content readability */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #212529 !important;
    }
    /* Ensure text in expandable sections is readable */
    .streamlit-expanderHeader {
        color: #212529 !important;
    }
    .streamlit-expanderContent {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_workout_data(file_path):
    """Load and analyze workout data with caching."""
    analyzer = WorkoutAnalyzer(file_path)
    analyzer.load_data()
    analyzer.calculate_1rm_estimates()
    return analyzer


@st.cache_data
def load_meet_data(lifter_name: str, force_update: bool = False):
    """Load and cache meet data for a lifter."""
    if not lifter_name.strip():
        return pd.DataFrame()
    
    try:
        integrator = OpenPowerliftingIntegrator()
        
        # Download data if needed
        if not integrator.download_data(force_update=force_update):
            return pd.DataFrame()
        
        # Load full database
        opl_df = integrator.load_data()
        if opl_df is None:
            return pd.DataFrame()
        
        # Get lifter's meet results
        meet_results = integrator.get_lifter_meets(opl_df, lifter_name)
        
        if meet_results.empty:
            return pd.DataFrame()
        
        # Format for dashboard integration
        formatted_meets = integrator.format_for_dashboard(meet_results)
        return formatted_meets
        
    except Exception as e:
        st.error(f"Error loading meet data: {e}")
        return pd.DataFrame()


def find_data_files():
    """Find available workout data files."""
    current_dir = Path(".")
    patterns = ["*normalized_fixed_validated*.csv", "*normalized_validated*.csv", "*normalized_clean*.csv", "*with_1rm*.csv"]
    
    files = []
    for pattern in patterns:
        files.extend(list(current_dir.glob(pattern)))
    
    # Remove duplicates while preserving order (prioritizes "fixed" files)
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    # Sort by modification time (newest first), but "fixed" files are already prioritized by pattern order
    unique_files.sort(key=lambda x: (0 if "fixed" in x.name else 1, -x.stat().st_mtime))
    return [str(f) for f in unique_files]


def calculate_workout_stats(analyzer):
    """Calculate comprehensive workout statistics."""
    df = analyzer.df.copy()
    
    if df.empty:
        return {
            'total_workouts': 0,
            'total_entries': 0,
            'date_range_days': 0,
            'first_workout': None,
            'last_workout': None,
            'workouts_per_week': 0,
            'main_lift_entries': 0,
            'accessory_entries': 0
        }
    
    # Count unique workout dates
    unique_dates = df['Date'].dt.date.nunique()
    
    # Date range
    first_date = df['Date'].min()
    last_date = df['Date'].max()
    date_range_days = (last_date - first_date).days
    
    # Training frequency (workouts per week)
    weeks_covered = max(1, date_range_days / 7)  # Avoid division by zero
    workouts_per_week = unique_dates / weeks_covered if weeks_covered > 0 else 0
    
    # Entry breakdown
    main_lifts = df[df['Lift_Type'].isin(['Squat', 'Bench', 'Deadlift'])]
    accessory = df[df['Lift_Type'] == 'Accessory']
    
    return {
        'total_workouts': unique_dates,
        'total_entries': len(df),
        'date_range_days': date_range_days,
        'first_workout': first_date.strftime('%Y-%m-%d'),
        'last_workout': last_date.strftime('%Y-%m-%d'),
        'workouts_per_week': workouts_per_week,
        'main_lift_entries': len(main_lifts),
        'accessory_entries': len(accessory)
    }


def display_workout_stats_sidebar(stats):
    """Display workout statistics in the sidebar."""
    st.sidebar.markdown("### 📊 Training Log Stats")
    
    # Create a nice info box with stats
    st.sidebar.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>🏋️ Workouts Logged:</strong></span>
            <span><strong>{stats['total_workouts']}</strong></span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>📝 Total Entries:</strong></span>
            <span><strong>{stats['total_entries']}</strong></span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>📅 Training Days:</strong></span>
            <span><strong>{stats['date_range_days']}</strong></span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>📈 Frequency:</strong></span>
            <span><strong>{stats['workouts_per_week']:.1f}/week</strong></span>
        </div>
        <hr style="margin: 0.5rem 0; border: 1px solid #ddd;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span>💪 Main Lifts:</span>
            <span>{stats['main_lift_entries']}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span>🔧 Accessories:</span>
            <span>{stats['accessory_entries']}</span>
        </div>
        <hr style="margin: 0.5rem 0; border: 1px solid #ddd;">
        <div style="font-size: 0.8rem; color: #666;">
            <div><strong>First:</strong> {stats['first_workout']}</div>
            <div><strong>Latest:</strong> {stats['last_workout']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_workout_stats_main(stats):
    """Display workout statistics on the main page."""
    # Create 4 columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Custom styled metric cards
    metrics = [
        ("🏋️ Workouts Logged", f"{stats['total_workouts']}", "Total unique training days", "#FF6B6B"),
        ("📝 Exercise Entries", f"{stats['total_entries']}", "All logged exercises", "#4ECDC4"),
        ("📈 Training Frequency", f"{stats['workouts_per_week']:.1f}/week", "Average workouts per week", "#45B7D1"),
        ("📅 Training Span", f"{stats['date_range_days']} days", f"From {stats['first_workout']} to {stats['last_workout']}", "#96CEB4")
    ]
    
    for col, (label, value, help_text, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color} 0%, {color}CC 100%);
                padding: 1rem;
                border-radius: 8px;
                color: white;
                text-align: center;
                margin: 0.25rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0; color: white; font-size: 0.9rem; font-weight: 500;">{label}</h4>
                <h2 style="margin: 0.25rem 0; color: white; font-size: 1.8rem; font-weight: 700;">{value}</h2>
                <p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 0.75rem;">{help_text}</p>
            </div>
            """, unsafe_allow_html=True)


def get_actual_prs_by_rep_range(analyzer):
    """Get actual PRs by rep range (not estimated 1RM)."""
    df = analyzer.df.copy()
    
    # Filter to main lifts only
    main_lifts = df[df['Lift_Type'].isin(['Squat', 'Bench', 'Deadlift'])].copy()
    
    if main_lifts.empty:
        return pd.DataFrame()
    
    # Find max weight for each rep count for each lift type
    pr_data = main_lifts.groupby(['Lift_Type', 'Reps'])['Load'].agg(['max', 'idxmax']).reset_index()
    pr_data.columns = ['Lift_Type', 'Reps', 'Max_Weight', 'idx']
    
    # Get the date for each PR
    pr_data['Date'] = pr_data['idx'].map(main_lifts['Date'])
    pr_data['Exercise'] = pr_data['idx'].map(main_lifts['Exercise'])
    
    # Only keep rep ranges 1-10 for clarity
    pr_data = pr_data[pr_data['Reps'] <= 10]
    
    return pr_data[['Lift_Type', 'Reps', 'Max_Weight', 'Date', 'Exercise']].sort_values(['Lift_Type', 'Reps'])


def plot_pr_heatmap(pr_data):
    """Create PR heatmap by rep range."""
    if pr_data.empty:
        return None
    
    # Create pivot table for heatmap
    heatmap_data = pr_data.pivot(index='Reps', columns='Lift_Type', values='Max_Weight')
    
    # Fill missing values with 0 for display
    heatmap_data = heatmap_data.fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        text=heatmap_data.values,
        texttemplate="%{text:.0f}",
        textfont={"size": 12},
        hovertemplate='<b>%{x}</b><br>' +
                      '%{y} Rep Max: %{z:.0f} lbs<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="🏆 Actual PRs by Rep Range",
            font=dict(size=16, color='#2E4057')
        ),
        xaxis_title="Lift Type",
        yaxis_title="Reps",
        height=300,
        margin=dict(t=40, b=20, l=40, r=20)
    )
    
    return fig


def plot_1rm_progression_with_meets(analyzer, meet_data, selected_lifts, date_range, method='best'):
    """Create 1RM progression chart - each lift only shows days when that lift was actually performed."""
    fig = go.Figure()
    
    colors = {'Squat': '#FF6B6B', 'Bench': '#4ECDC4', 'Deadlift': '#45B7D1'}
    
    # Filter data by date range first
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    
    for lift in selected_lifts:
        # Step 1: Get ONLY this specific lift type data
        lift_data = analyzer.df[
            (analyzer.df['Lift_Type'] == lift) & 
            (analyzer.df['Date'] >= start_date) & 
            (analyzer.df['Date'] <= end_date) &
            (analyzer.df['Load'].notna()) &  # Must have valid weight
            (analyzer.df['Reps'].notna()) &  # Must have valid reps
            (analyzer.df['Load'] > 0) &      # Weight must be positive
            (analyzer.df['Reps'] > 0)        # Reps must be positive
        ].copy()
        
        if not lift_data.empty:
            # Step 2: Calculate 1RM estimates using the selected method
            calculator = analyzer.calculator
            
            if method == 'best':
                lift_data['estimated_1rm'] = lift_data.apply(
                    lambda row: calculator.best_estimate(
                        weight=row['Load'], 
                        reps=row['Reps'], 
                        rpe=row.get('RPE')
                    ), axis=1
                )
            elif method == 'rpe_based':
                lift_data['estimated_1rm'] = lift_data.apply(
                    lambda row: calculator.rpe_based_1rm(
                        weight=row['Load'], 
                        reps=row['Reps'], 
                        rpe=row.get('RPE')
                    ), axis=1
                )
            elif method == 'epley':
                lift_data['estimated_1rm'] = lift_data.apply(
                    lambda row: calculator.epley_formula(
                        weight=row['Load'], 
                        reps=row['Reps']
                    ), axis=1
                )
            elif method == 'brzycki':
                lift_data['estimated_1rm'] = lift_data.apply(
                    lambda row: calculator.brzycki_formula(
                        weight=row['Load'], 
                        reps=row['Reps']
                    ), axis=1
                )
            elif method == 'lombardi':
                lift_data['estimated_1rm'] = lift_data.apply(
                    lambda row: calculator.lombardi_formula(
                        weight=row['Load'], 
                        reps=row['Reps']
                    ), axis=1
                )
            else:
                lift_data['estimated_1rm'] = lift_data.apply(
                    lambda row: calculator.best_estimate(
                        weight=row['Load'], 
                        reps=row['Reps'], 
                        rpe=row.get('RPE')
                    ), axis=1
                )
            
            # Step 3: Remove invalid 1RM estimates
            lift_data = lift_data[
                (lift_data['estimated_1rm'].notna()) & 
                (lift_data['estimated_1rm'] > 0) &
                (lift_data['estimated_1rm'] < 1000)  # Remove unrealistic estimates
            ]
            
            if not lift_data.empty:
                # Step 4: Get the best 1RM estimate per day
                daily_maxes = lift_data.groupby('Date')['estimated_1rm'].max().reset_index()
                daily_maxes = daily_maxes.sort_values('Date')
                
                # Step 5: Add additional filtering to remove outliers
                if len(daily_maxes) > 1:
                    # Remove estimates that are less than 50% of the median (likely errors)
                    median_1rm = daily_maxes['estimated_1rm'].median()
                    min_threshold = median_1rm * 0.5
                    daily_maxes = daily_maxes[daily_maxes['estimated_1rm'] >= min_threshold]
                
                if not daily_maxes.empty:
                    fig.add_trace(go.Scatter(
                        x=daily_maxes['Date'],
                        y=daily_maxes['estimated_1rm'],
                        mode='markers+lines',  # Markers first to show individual points clearly
                        name=f'{lift} Training',
                        line=dict(color=colors.get(lift, '#333'), width=2),
                        marker=dict(size=5, color=colors.get(lift, '#333')),
                        connectgaps=False,
                        hovertemplate=f'<b>{lift} Training</b><br>' +
                                      'Date: %{x}<br>' +
                                      'Est. 1RM: %{y:.1f} lbs<br>' +
                                      '<extra></extra>'
                    ))
        
        # Meet results
        if not meet_data.empty:
            meet_lifts = meet_data[meet_data['Lift_Type'] == lift].copy()
            if not meet_lifts.empty:
                meet_lifts_filtered = meet_lifts[
                    (meet_lifts['Date'] >= start_date) & 
                    (meet_lifts['Date'] <= end_date)
                ]
                
                if not meet_lifts_filtered.empty:
                    fig.add_trace(go.Scatter(
                        x=meet_lifts_filtered['Date'],
                        y=meet_lifts_filtered['Load'],
                        mode='markers',
                        name=f'{lift} Meets',
                        marker=dict(
                            size=12,
                            color='gold',
                            symbol='star',
                            line=dict(width=2, color=colors.get(lift, '#333'))
                        ),
                        hovertemplate=f'<b>{lift} Competition</b><br>' +
                                      'Date: %{x}<br>' +
                                      'Weight: %{y:.1f} lbs<br>' +
                                      'Place: %{customdata}<br>' +
                                      '<extra></extra>',
                        customdata=meet_lifts_filtered['Place']
                    ))
    
    fig.update_layout(
        title=dict(
            text="📈 Training vs Competition Performance",
            font=dict(size=16, color='#2E4057')
        ),
        xaxis_title="Date",
        yaxis_title="Weight (lbs)",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=350,
        margin=dict(t=60, b=40, l=40, r=20)
    )
    
    return fig


def plot_rpe_trends(analyzer, selected_lifts, date_range):
    """Plot RPE trends over time - much more useful than redundant 1RM bars."""
    df = analyzer.df.copy()
    
    # Filter by lifts and date range
    df = df[df['Lift_Type'].isin(selected_lifts)]
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask]
    
    # Filter out missing RPE values
    df = df[df['RPE'].notna() & (df['RPE'] > 0)]
    
    if df.empty:
        # Return empty chart if no RPE data
        fig = go.Figure()
        fig.add_annotation(
            text="No RPE data available for selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title="📊 RPE Trends Over Time",
            template="plotly_white",
            height=300,
            margin=dict(t=40, b=20, l=40, r=20)
        )
        return fig
    
    # Group by week and lift type for smoother visualization
    df['Week'] = df['Date'].dt.to_period('W').dt.start_time
    
    # Calculate average RPE by week and lift type
    rpe_data = df.groupby(['Week', 'Lift_Type'])['RPE'].mean().reset_index()
    rpe_data['RPE'] = rpe_data['RPE'].round(1)
    
    fig = px.line(
        rpe_data,
        x='Week',
        y='RPE',
        color='Lift_Type',
        title="📊 RPE Trends Over Time",
        color_discrete_map={'Squat': '#FF6B6B', 'Bench': '#4ECDC4', 'Deadlift': '#45B7D1'}
    )
    
    # Add RPE zone references
    fig.add_hline(y=6, line_dash="dash", line_color="green", 
                  annotation_text="RPE 6 (Easy)", annotation_position="bottom right")
    fig.add_hline(y=8, line_dash="dash", line_color="orange", 
                  annotation_text="RPE 8 (Hard)", annotation_position="bottom right")
    fig.add_hline(y=9, line_dash="dash", line_color="red", 
                  annotation_text="RPE 9 (Very Hard)", annotation_position="bottom right")
    
    fig.update_layout(
        template="plotly_white",
        height=300,
        yaxis_title="Average RPE",
        yaxis=dict(range=[5, 10]),
        margin=dict(t=40, b=20, l=40, r=20)
    )
    
    return fig


def plot_training_volume_compact(analyzer, selected_lifts, date_range):
    """Plot compact training volume over time."""
    df = analyzer.df.copy()
    
    # Filter by lifts and date range
    df = df[df['Lift_Type'].isin(selected_lifts)]
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask]
    
    # Calculate volume (sets * reps * weight)
    df['volume'] = df['Sets'] * df['Reps'] * df['Load']
    
    # Group by week and lift type for smoother visualization
    df['Week'] = df['Date'].dt.to_period('W').dt.start_time
    volume_data = df.groupby(['Week', 'Lift_Type'])['volume'].sum().reset_index()
    
    fig = px.line(
        volume_data,
        x='Week',
        y='volume',
        color='Lift_Type',
        title="📊 Weekly Training Volume",
        color_discrete_map={'Squat': '#FF6B6B', 'Bench': '#4ECDC4', 'Deadlift': '#45B7D1'}
    )
    
    fig.update_layout(
        template="plotly_white",
        height=300,
        yaxis_title="Volume (Sets × Reps × Weight)",
        margin=dict(t=40, b=20, l=40, r=20)
    )
    
    return fig


def display_summary_cards_compact(analyzer):
    """Display compact summary metric cards for training estimates."""
    max_1rms = analyzer.get_max_1rm_by_lift()
    recent_progress = analyzer.get_recent_progress(90)
    
    st.markdown("**📈 Training Estimates** *(calculated from recent sessions)*")
    col1, col2, col3 = st.columns(3)
    
    for i, (col, lift) in enumerate(zip([col1, col2, col3], ['Squat', 'Bench', 'Deadlift'])):
        with col:
            max_data = max_1rms[max_1rms['Lift_Type'] == lift]
            recent_data = recent_progress[recent_progress['Lift_Type'] == lift]
            
            if not max_data.empty:
                max_1rm = max_data.iloc[0]['Max_1RM']
                max_date = max_data.iloc[0]['Date'].strftime('%m/%d')
                
                recent_1rm = recent_data.iloc[0]['Recent_Max_1RM'] if not recent_data.empty else 0
                
                # Calculate progress
                progress = recent_1rm - max_1rm if not recent_data.empty else 0
                progress_str = f"+{progress:.1f}" if progress > 0 else f"{progress:.1f}"
                
                st.markdown(f"""
                <div class="lift-card">
                    <h4 style="margin: 0; color: white; font-size: 0.9rem;">{lift}</h4>
                    <h3 style="margin: 0.2rem 0; color: white; font-size: 1.3rem;">~{max_1rm:.0f} lbs</h3>
                    <p style="margin: 0; color: #f0f0f0; font-size: 0.7rem;">Est. Max: {max_date} | 90d: {progress_str}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="lift-card">
                    <h4 style="margin: 0; color: white; font-size: 0.9rem;">{lift}</h4>
                    <h3 style="margin: 0.2rem 0; color: white; font-size: 1.3rem;">No Data</h3>
                </div>
                """, unsafe_allow_html=True)


def display_competition_bests(meet_data):
    """Display best competition lifts from OpenPowerlifting data."""
    if meet_data.empty:
        return
    
    # Get best competition lift for each type
    comp_bests = meet_data.groupby('Lift_Type')['Load'].agg(['max', 'idxmax']).reset_index()
    comp_bests.columns = ['Lift_Type', 'Best_Weight', 'idx']
    
    # Get the date and meet info for each best
    comp_bests['Date'] = comp_bests['idx'].map(meet_data['Date'])
    comp_bests['Place'] = comp_bests['idx'].map(meet_data['Place'])
    
    col1, col2, col3 = st.columns(3)
    
    for col, lift in zip([col1, col2, col3], ['Squat', 'Bench', 'Deadlift']):
        with col:
            lift_data = comp_bests[comp_bests['Lift_Type'] == lift]
            
            if not lift_data.empty:
                best_weight = lift_data.iloc[0]['Best_Weight']
                best_date = lift_data.iloc[0]['Date'].strftime('%m/%d/%Y')
                place = lift_data.iloc[0]['Place']
                
                # Different color scheme for competition results (gold theme)
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
                    padding: 0.8rem;
                    border-radius: 8px;
                    color: #1a1a1a;
                    text-align: center;
                    margin: 0.25rem 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin: 0; color: #1a1a1a; font-size: 0.9rem; font-weight: 500;">{lift}</h4>
                    <h3 style="margin: 0.2rem 0; color: #1a1a1a; font-size: 1.3rem; font-weight: 700;">{best_weight:.0f} lbs</h3>
                    <p style="margin: 0; color: #444; font-size: 0.7rem;">Meet: {best_date} | Place: {place}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #D3D3D3 0%, #A9A9A9 100%);
                    padding: 0.8rem;
                    border-radius: 8px;
                    color: #666;
                    text-align: center;
                    margin: 0.25rem 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin: 0; color: #666; font-size: 0.9rem; font-weight: 500;">{lift}</h4>
                    <h3 style="margin: 0.2rem 0; color: #666; font-size: 1.3rem; font-weight: 700;">No Data</h3>
                </div>
                """, unsafe_allow_html=True)


def plot_trajectory_forecast(predictor, lift_type, forecast_days=180):
    """Plot trajectory analysis with forecast."""
    # Use improved method if available, fallback to regular method
    if hasattr(predictor, 'improved_trajectory_analysis'):
        trajectory = predictor.improved_trajectory_analysis(lift_type, forecast_days)
    else:
        trajectory = predictor.trajectory_analysis(lift_type, forecast_days)
    
    if not trajectory['success']:
        fig = go.Figure()
        fig.add_annotation(
            text=trajectory['message'],
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title=f"📈 {lift_type} Trajectory Forecast",
            template="plotly_white",
            height=300,
            margin=dict(t=40, b=20, l=40, r=20)
        )
        return fig
    
    fig = go.Figure()
    
    # Historical data
    hist_data = trajectory['historical_data']
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data['estimated_1rm'],
        mode='markers+lines',
        name='Historical',
        line=dict(color='#45B7D1', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=trajectory['forecast_dates'],
        y=trajectory['forecast_values'],
        mode='lines',
        name='Forecast',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Current point
    fig.add_trace(go.Scatter(
        x=[hist_data['Date'].iloc[-1]],
        y=[trajectory['current_1rm']],
        mode='markers',
        name='Current',
        marker=dict(size=10, color='gold', symbol='star')
    ))
    
    fig.update_layout(
        title=f"📈 {lift_type} Trajectory Forecast ({trajectory['trend_direction']})",
        xaxis_title="Date",
        yaxis_title="Estimated 1RM (lbs)",
        template="plotly_white",
        height=300,
        margin=dict(t=40, b=20, l=40, r=20),
        showlegend=True
    )
    
    return fig


def plot_rep_range_effectiveness(predictor, selected_lifts):
    """Plot rep range effectiveness analysis."""
    fig = go.Figure()
    
    colors = {'Squat': '#FF6B6B', 'Bench': '#4ECDC4', 'Deadlift': '#45B7D1'}
    
    for lift in selected_lifts:
        rep_analysis = predictor.optimal_rep_ranges(lift)
        
        if not rep_analysis['success']:
            continue
        
        rep_data = rep_analysis['rep_analysis']
        
        # Extract data for plotting
        rep_ranges = list(rep_data.keys())
        correlations = [rep_data[r]['correlation_with_progress'] for r in rep_ranges]
        
        fig.add_trace(go.Bar(
            x=rep_ranges,
            y=correlations,
            name=f'{lift}',
            marker_color=colors.get(lift, '#666'),
            opacity=0.8
        ))
    
    fig.update_layout(
        title="🎯 Rep Range Effectiveness (Progress Correlation)",
        xaxis_title="Rep Range",
        yaxis_title="Correlation with Progress",
        template="plotly_white",
        height=300,
        margin=dict(t=40, b=20, l=40, r=20),
        barmode='group'
    )
    
    return fig


def plot_volume_analysis(predictor, selected_lifts):
    """Plot volume sweet spot analysis."""
    fig = go.Figure()
    
    colors = {'Squat': '#FF6B6B', 'Bench': '#4ECDC4', 'Deadlift': '#45B7D1'}
    
    for i, lift in enumerate(selected_lifts):
        volume_analysis = predictor.volume_sweet_spot(lift)
        
        if not volume_analysis['success']:
            continue
        
        volume_data = volume_analysis['volume_analysis']
        
        # Extract data for plotting
        buckets = list(volume_data.keys())
        avg_1rms = [volume_data[b]['average_1rm'] for b in buckets]
        
        fig.add_trace(go.Scatter(
            x=buckets,
            y=avg_1rms,
            mode='lines+markers',
            name=f'{lift}',
            line=dict(color=colors.get(lift, '#666'), width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="💪 Volume Sweet Spot Analysis",
        xaxis_title="Volume Category",
        yaxis_title="Average 1RM (lbs)",
        template="plotly_white",
        height=300,
        margin=dict(t=40, b=20, l=40, r=20)
    )
    
    return fig


def display_goal_timeline(predictor, lift_type, target_1rm):
    """Display goal timeline analysis."""
    # Use improved method if available, fallback to regular method
    if hasattr(predictor, 'realistic_goal_timeline'):
        timeline = predictor.realistic_goal_timeline(lift_type, target_1rm)
    else:
        timeline = predictor.goal_timeline(lift_type, target_1rm)
    
    if not timeline['success']:
        st.error(f"❌ {timeline.get('message', 'Could not calculate timeline')}")
        return
    
    if timeline.get('already_achieved'):
        st.success(f"🎉 {timeline['message']}")
        return
    
    if not timeline.get('reachable'):
        st.warning(f"⚠️ {timeline['message']}")
        st.info(f"💡 Current monthly gain: {timeline['monthly_gain']:.1f} lbs/month")
        return
    
    # Display timeline info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <h4>🎯 Target</h4>
            <p><strong>{timeline['target_1rm']:.0f} lbs</strong></p>
            <p>{timeline['target_1rm'] - timeline['current_1rm']:+.1f} lbs to go</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h4>📅 Target Date</h4>
            <p><strong>{timeline['target_date'].strftime('%b %Y')}</strong></p>
            <p>{timeline['months_to_target']:.1f} months</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="prediction-card">
            <h4>📈 Required Rate</h4>
            <p><strong>{timeline['monthly_gain']:.1f} lbs/month</strong></p>
            <p>Based on current trend</p>
        </div>
        """, unsafe_allow_html=True)


def display_meet_predictions(predictor, selected_lifts, meet_date=None):
    """Display meet attempt predictions."""
    st.markdown("#### 🏆 Competition Attempt Predictions")
    
    if meet_date is None:
        meet_date = datetime.now() + timedelta(days=84)  # 12 weeks default
    
    cols = st.columns(len(selected_lifts))
    
    for i, lift in enumerate(selected_lifts):
        with cols[i]:
            prediction = predictor.meet_attempt_prediction(lift, meet_date)
            
            if prediction['success']:
                confidence_color = "#4caf50" if prediction['confidence'] > 0.8 else "#ff9800" if prediction['confidence'] > 0.6 else "#f44336"
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>{lift}</h4>
                    <p><strong>Predicted 1RM:</strong> {prediction['predicted_1rm']:.0f} lbs</p>
                    <hr style="margin: 0.5rem 0; border-color: #2196f3;">
                    <p><strong>Suggested Attempts:</strong></p>
                    <p>🥉 <strong>Opener:</strong> {prediction['opener']:.0f} lbs (90%)</p>
                    <p>🥈 <strong>Second:</strong> {prediction['second_attempt']:.0f} lbs (97%)</p>
                    <p>🥇 <strong>Third:</strong> {prediction['third_attempt']:.0f} lbs (103%)</p>
                    <hr style="margin: 0.5rem 0; border-color: #2196f3;">
                    <p><strong>Confidence:</strong> <span style="color: {confidence_color}; font-weight: bold;">{prediction['confidence']:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Could not predict {lift}")


def plot_workout_frequency(analyzer, date_range):
    """Plot workout frequency over time to show training consistency and breaks."""
    df = analyzer.df.copy()
    
    # Filter by date range
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask]
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No workout data available for selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title="📅 Workout Frequency Over Time",
            template="plotly_white",
            height=300,
            margin=dict(t=40, b=20, l=40, r=20)
        )
        return fig
    
    # Count unique workout dates by month
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # Group by month and count unique dates within each month
    monthly_workouts = df.groupby('YearMonth').agg({
        'Date': lambda x: x.dt.date.nunique()
    }).reset_index()
    monthly_workouts.columns = ['YearMonth', 'Workouts']
    
    # Convert YearMonth back to datetime for plotting
    monthly_workouts['Date'] = monthly_workouts['YearMonth'].dt.start_time
    
    # Create the line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_workouts['Date'],
        y=monthly_workouts['Workouts'],
        mode='lines+markers',
        name='Workouts per Month',
        line=dict(color='#45B7D1', width=3),
        marker=dict(size=6, color='#45B7D1'),
        fill='tonexty',
        fillcolor='rgba(69, 183, 209, 0.1)',
        hovertemplate='<b>%{x|%B %Y}</b><br>' +
                      'Workouts: %{y}<br>' +
                      '<extra></extra>'
    ))
    
    # Add average line
    avg_workouts = monthly_workouts['Workouts'].mean()
    fig.add_hline(
        y=avg_workouts, 
        line_dash="dash", 
        line_color="orange",
        annotation_text=f"Average: {avg_workouts:.1f} workouts/month",
        annotation_position="top right"
    )
    
    # Highlight low activity periods (< 50% of average)
    low_activity_threshold = avg_workouts * 0.5
    low_months = monthly_workouts[monthly_workouts['Workouts'] < low_activity_threshold]
    
    if not low_months.empty:
        fig.add_trace(go.Scatter(
            x=low_months['Date'],
            y=low_months['Workouts'],
            mode='markers',
            name='Low Activity',
            marker=dict(size=10, color='red', symbol='x'),
            hovertemplate='<b>%{x|%B %Y}</b><br>' +
                          'Workouts: %{y} (Low Activity)<br>' +
                          '<extra></extra>'
        ))
    
    avg_workouts = monthly_workouts['Workouts'].mean()
    
    fig.update_layout(
        title=dict(
            text=f"📅 Logged Workouts Over Time (Avg: {avg_workouts:.1f}/month)",
            font=dict(size=16, color='#2E4057')
        ),
        xaxis_title="Month",
        yaxis_title="Workouts per Month",
        template="plotly_white",
        hovermode='x unified',
        height=300,
        margin=dict(t=40, b=20, l=40, r=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def main():
    """Main dashboard application."""
    
    # Compact Header
    st.title("💪 Workout Analysis Dashboard")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # File selection
    data_files = find_data_files()
    if not data_files:
        st.error("❌ No workout data files found!")
        st.info("Make sure you have normalized workout data files (CSV) in the current directory.")
        st.stop()
    
    selected_file = st.sidebar.selectbox(
        "📁 Data File",
        options=data_files,
        index=0,
        help="Choose your workout data file"
    )
    
    # Load data
    try:
        with st.spinner("Loading workout data..."):
            analyzer = load_workout_data(selected_file)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
    
    # Calculate workout statistics (will display on main page)
    workout_stats = calculate_workout_stats(analyzer)
    
    # Meet data integration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Meet Data")
    
    lifter_name = st.sidebar.text_input(
        "Your Name (for OpenPowerlifting)",
        help="Enter your name as it appears in competition results",
        placeholder="e.g. Alex Smith"
    )
    
    force_update_meets = st.sidebar.button("🔄 Update Meet Data")
    
    # Load meet data if name provided
    meet_data = pd.DataFrame()
    if lifter_name.strip():
        with st.spinner("Loading meet results..."):
            meet_data = load_meet_data(lifter_name, force_update=force_update_meets)
            
        if not meet_data.empty:
            st.sidebar.success(f"✅ Found {len(meet_data)} meet results")
        else:
            st.sidebar.info("ℹ️ No meet results found")
    
    # Lift selection
    available_lifts = ['Squat', 'Bench', 'Deadlift']
    selected_lifts = st.sidebar.multiselect(
        "🏋️ Select Lifts",
        options=available_lifts,
        default=available_lifts,
        help="Choose which lifts to analyze"
    )
    
    # Date range selection
    min_date = analyzer.df['Date'].min().date()
    max_date = analyzer.df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select date range for analysis"
    )
    
    if len(date_range) != 2:
        st.warning("Please select both start and end dates.")
        st.stop()
    
    # 1RM calculation method
    method = st.sidebar.selectbox(
        "🧮 1RM Method",
        options=['best', 'rpe_based', 'epley', 'brzycki', 'lombardi'],
        index=0,
        help="Choose 1RM calculation method"
    )
    
    # === MAIN DASHBOARD LAYOUT ===
    
    # Row 1: Workout Statistics
    st.markdown("## 📊 Training Overview")
    display_workout_stats_main(workout_stats)
    
    # Row 2: Workout Frequency Chart and Current 1RM Status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Workout Frequency Over Time
        frequency_fig = plot_workout_frequency(analyzer, date_range)
        st.plotly_chart(frequency_fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Training vs Competition")
        display_summary_cards_compact(analyzer)
        
        # Add competition bests if meet data is available
        if not meet_data.empty:
            st.markdown("#### 🥇 Competition Bests")
            display_competition_bests(meet_data)
        else:
            st.info("💡 Enter your name in the sidebar to load competition results from OpenPowerlifting")
    
    # Row 3: PR Heatmap and 1RM Progression
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # PR Heatmap
        pr_data = get_actual_prs_by_rep_range(analyzer)
        pr_fig = plot_pr_heatmap(pr_data)
        if pr_fig:
            st.plotly_chart(pr_fig, use_container_width=True)
        else:
            st.info("No PR data available")
    
    with col2:
        # 1RM Progression with Meet Results
        if selected_lifts:
            if not meet_data.empty:
                progression_fig = plot_1rm_progression_with_meets(analyzer, meet_data, selected_lifts, date_range, method)
            else:
                # Fallback to training-only chart
                progression_fig = plot_1rm_progression_with_meets(analyzer, pd.DataFrame(), selected_lifts, date_range, method)
            st.plotly_chart(progression_fig, use_container_width=True)
        else:
            st.warning("Select lifts to view progression")
    
    # Row 4: RPE Trends and Training Volume
    col1, col2 = st.columns(2)
    
    with col1:
        # RPE Trends (much more useful than redundant 1RM bars!)
        if selected_lifts:
            rpe_fig = plot_rpe_trends(analyzer, selected_lifts, date_range)
            st.plotly_chart(rpe_fig, use_container_width=True)
    
    with col2:
        # Training volume
        if selected_lifts:
            volume_fig = plot_training_volume_compact(analyzer, selected_lifts, date_range)
            st.plotly_chart(volume_fig, use_container_width=True)
    
    # Row 5: Recent PRs and Meet Results
    col1, col2 = st.columns(2)
    
    with col1:
        if not pr_data.empty:
            st.markdown("#### 📋 Recent Training PRs")
            
            # Get most recent PRs (last 6 months)
            recent_cutoff = pd.Timestamp(datetime.now() - timedelta(days=180))
            recent_prs = pr_data[pr_data['Date'] >= recent_cutoff].copy()
            
            if not recent_prs.empty:
                recent_prs['Date'] = recent_prs['Date'].dt.strftime('%Y-%m-%d')
                recent_prs = recent_prs.sort_values('Date', ascending=False)
                
                st.dataframe(
                    recent_prs[['Date', 'Lift_Type', 'Reps', 'Max_Weight']].rename(columns={
                        'Date': 'Date',
                        'Lift_Type': 'Lift',
                        'Reps': 'Reps',
                        'Max_Weight': 'Weight (lbs)'
                    }).head(8),
                    use_container_width=True,
                    height=200
                )
            else:
                st.info("No recent PRs found.")
    
    with col2:
        if not meet_data.empty:
            st.markdown("#### 🏆 Competition Results")
            
            # Show recent meet results
            meet_display = meet_data.copy()
            meet_display['Date'] = meet_display['Date'].dt.strftime('%Y-%m-%d')
            meet_display = meet_display.sort_values('Date', ascending=False)
            
            display_cols = ['Date', 'Lift_Type', 'Load', 'Place']
            if 'Total' in meet_display.columns:
                display_cols.append('Total')
            
            st.dataframe(
                meet_display[display_cols].rename(columns={
                    'Date': 'Date',
                    'Lift_Type': 'Lift',
                    'Load': 'Weight (lbs)',
                    'Place': 'Place',
                    'Total': 'Total (lbs)'
                }).head(8),
                use_container_width=True,
                height=200
            )
        else:
            # Method explanation
            st.markdown("#### 🧮 1RM Method Info")
            method_explanations = {
                'best': "Uses the most appropriate formula based on available data",
                'rpe_based': "Uses RPE-based percentage tables - most accurate",
                'epley': "1RM = weight × (1 + reps/30) - most popular",
                'brzycki': "1RM = weight × (36/(37-reps)) - good for 1-10 reps",
                'lombardi': "1RM = weight × reps^0.10 - good for higher reps"
            }
            
            st.info(f"**{method.title()} Method**\n\n{method_explanations[method]}")
    
    # === PREDICTIVE ANALYTICS SECTION ===
    st.markdown("---")
    st.markdown("## 🔮 Predictive Analytics")
    
    # Information about improved predictions
    st.info("🎯 **Using Improved Predictions**: This dashboard now filters data to your main exercises and strength-focused rep ranges (1-6 reps) for more accurate forecasting.")
    
    # Initialize improved predictor for accurate results
    predictor = ImprovedPowerliftingPredictor(analyzer.df)
    
    # Tabs for different analytics
    forecast_tab, optimization_tab, goals_tab, meet_tab = st.tabs([
        "📊 Progress Forecasting", 
        "🧠 Training Optimization", 
        "🎯 Goal Timeline", 
        "🏆 Meet Planning"
    ])
    
    with forecast_tab:
        st.markdown("### 📈 Trajectory Analysis & Forecasting")
        
        # Controls
        col1, col2 = st.columns([2, 1])
        with col1:
            forecast_lift = st.selectbox(
                "Select lift for forecast:",
                options=selected_lifts,
                key="forecast_lift"
            )
        with col2:
            forecast_days = st.slider(
                "Forecast period (days):",
                min_value=30, max_value=365, 
                value=180, step=30,
                key="forecast_days"
            )
        
        # Display trajectory forecast
        if forecast_lift:
            fig_trajectory = plot_trajectory_forecast(predictor, forecast_lift, forecast_days)
            st.plotly_chart(fig_trajectory, use_container_width=True)
            
            # Show trajectory metrics
            trajectory = predictor.improved_trajectory_analysis(forecast_lift, forecast_days)
            if trajectory['success']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>📊 Current 1RM</h4>
                        <p><strong>{trajectory['current_1rm']:.1f} lbs</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>🔮 Forecast 1RM</h4>
                        <p><strong>{trajectory['forecast_1rm']:.1f} lbs</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    gain_color = "#4caf50" if trajectory['projected_gain'] > 0 else "#f44336"
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>📈 Projected Gain</h4>
                        <p><strong style="color: {gain_color};">{trajectory['projected_gain']:+.1f} lbs</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    rate_color = "#4caf50" if trajectory['monthly_gain'] > 0 else "#f44336"
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>📅 Monthly Rate</h4>
                        <p><strong style="color: {rate_color};">{trajectory['monthly_gain']:+.1f} lbs/month</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model info
                st.info(f"📊 Model: {trajectory['model_type']} | R² Score: {trajectory['r2_score']:.3f} | Trend: {trajectory['trend_direction']}")
        else:
            st.warning("Please select a lift for forecasting")
    
    with optimization_tab:
        st.markdown("### 🧠 Training Optimization Analysis")
        
        if selected_lifts:
            # Rep Range Effectiveness
            st.markdown("#### 🎯 Rep Range Effectiveness")
            fig_rep_ranges = plot_rep_range_effectiveness(predictor, selected_lifts)
            st.plotly_chart(fig_rep_ranges, use_container_width=True)
            
            # Volume Analysis
            st.markdown("#### 💪 Volume Sweet Spot")
            fig_volume = plot_volume_analysis(predictor, selected_lifts)
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # Frequency Analysis
            st.markdown("#### 📅 Optimal Training Frequency")
            
            freq_cols = st.columns(len(selected_lifts))
            for i, lift in enumerate(selected_lifts):
                with freq_cols[i]:
                    freq_analysis = predictor.optimal_frequency(lift)
                    if freq_analysis['success']:
                        st.markdown(f"""
                        <div class="optimization-card">
                            <h4>{lift}</h4>
                            <p><strong>Optimal:</strong> {freq_analysis['optimal_frequency']}x/week</p>
                            <p><strong>Current Avg:</strong> {freq_analysis['average_frequency']:.1f}x/week</p>
                            <p><strong>Range:</strong> {freq_analysis['frequency_range']}x/week</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Insufficient data for {lift}")
            
            # Optimization Recommendations
            st.markdown("#### 💡 Optimization Recommendations")
            for lift in selected_lifts:
                with st.expander(f"{lift} Recommendations"):
                    # Rep ranges
                    rep_analysis = predictor.optimal_rep_ranges(lift)
                    if rep_analysis['success'] and rep_analysis['ranked_effectiveness']:
                        best_rep_range = rep_analysis['ranked_effectiveness'][0][0]
                        st.success(f"🎯 Most effective rep range: **{best_rep_range}**")
                    
                    # Volume
                    volume_analysis = predictor.volume_sweet_spot(lift)
                    if volume_analysis['success']:
                        recommendation = volume_analysis['volume_recommendation']
                        optimal_volume = volume_analysis.get('optimal_volume')
                        if optimal_volume:
                            st.info(f"💪 Volume recommendation: **{recommendation}** (target: {optimal_volume:.0f} weekly volume)")
                        else:
                            st.info(f"💪 Volume recommendation: **{recommendation}**")
        else:
            st.warning("Please select lifts to view optimization analysis")
    
    with goals_tab:
        st.markdown("### 🎯 Goal Timeline Analysis")
        
        if selected_lifts:
            col1, col2 = st.columns([1, 1])
            with col1:
                goal_lift = st.selectbox(
                    "Select lift:",
                    options=selected_lifts,
                    key="goal_lift"
                )
            with col2:
                # Get current max as a reasonable default
                current_1rms = analyzer.get_max_1rm_by_lift()
                current_max = 400  # default
                if not current_1rms.empty:
                    lift_data = current_1rms[current_1rms['Lift_Type'] == goal_lift]
                    if not lift_data.empty:
                        current_max = int(lift_data.iloc[0]['Max_1RM']) + 25  # Add 25 lbs as target
                
                target_1rm = st.number_input(
                    "Target 1RM (lbs):",
                    min_value=100, max_value=1000,
                    value=current_max,
                    step=5,
                    key="target_1rm"
                )
            
            if st.button("🎯 Calculate Timeline", type="primary"):
                display_goal_timeline(predictor, goal_lift, target_1rm)
        else:
            st.warning("Please select lifts for goal analysis")
    
    with meet_tab:
        st.markdown("### 🏆 Competition Meet Planning")
        
        if selected_lifts:
            col1, col2 = st.columns([1, 1])
            with col1:
                meet_date = st.date_input(
                    "Meet Date:",
                    value=datetime.now() + timedelta(days=84),
                    min_value=datetime.now().date(),
                    key="meet_date"
                )
            with col2:
                st.write("")  # Spacing
                show_predictions = st.button("🏆 Generate Predictions", type="primary")
            
            if show_predictions:
                # Convert date to datetime for consistency
                meet_datetime = pd.to_datetime(meet_date)
                display_meet_predictions(predictor, selected_lifts, meet_datetime)
        else:
            st.warning("Please select lifts for meet planning")
    
    # Footer
    st.markdown("---")
    
    # Data info note
    total_workouts = len(analyzer.df['Date'].dt.date.unique())
    st.info(f"📊 **Data Note**: This dashboard shows {total_workouts} logged workout sessions from your training data. The workout frequency reflects detailed logging sessions rather than total training frequency.")
    
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 0.5rem;">
        <p>💪 Track your strength journey • 🏆 Meet data from OpenPowerlifting • 📊 Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 