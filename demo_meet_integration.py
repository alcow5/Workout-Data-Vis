#!/usr/bin/env python3
"""
Demo: OpenPowerlifting Meet Data Integration
Test the meet data integration with the dashboard
"""

from meet_data_integrator import OpenPowerliftingIntegrator
import pandas as pd

def demo_integration():
    """Demo the OpenPowerlifting integration."""
    
    print("🏋️ OpenPowerlifting Meet Data Integration Demo")
    print("=" * 50)
    
    # Initialize integrator
    integrator = OpenPowerliftingIntegrator()
    
    # Download data (this will cache it)
    print("\n📥 Downloading OpenPowerlifting database...")
    if not integrator.download_data():
        print("❌ Failed to download data")
        return
    
    # Load data
    print("\n📊 Loading data...")
    df = integrator.load_data()
    if df is None:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(df):,} competition entries")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"🏛️ Federations: {df['Federation'].nunique()} unique")
    
    # Demo search functionality
    print("\n🔍 Testing search functionality...")
    
    # Search for some well-known lifters
    test_names = ["John Haack", "Julius Maddox", "Amanda Lawrence", "Jennifer Thompson"]
    
    for name in test_names:
        print(f"\n🔎 Searching for '{name}'...")
        results = integrator.search_lifter(df, name)
        
        if not results.empty:
            print(f"✅ Found {len(results)} entries")
            
            # Get meet results
            meet_results = integrator.get_lifter_meets(df, name)
            if not meet_results.empty:
                latest = meet_results.iloc[-1]
                print(f"   Last meet: {latest['Date'].strftime('%Y-%m-%d')} - {latest['MeetName']}")
                print(f"   Latest total: {latest.get('TotalKg', 'N/A')} kg")
            
            # Format for dashboard
            formatted = integrator.format_for_dashboard(meet_results)
            print(f"   Dashboard entries: {len(formatted)}")
        else:
            print(f"❌ No results found")
    
    print("\n✅ Demo completed! OpenPowerlifting integration is working.")
    print("\n💡 To use with your dashboard:")
    print("   1. Run: streamlit run workout_dashboard.py")
    print("   2. Enter your name in the 'Meet Data' section")
    print("   3. Your competition results will be plotted with training data!")

if __name__ == "__main__":
    demo_integration() 