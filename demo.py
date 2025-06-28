#!/usr/bin/env python3
"""
Demo script for the Workout Log Normalizer
Shows how to use the tool programmatically
"""

from workout_log_normalizer import WorkoutLogNormalizer


def main():
    """Demo the workout log normalizer with your Excel file."""
    
    print("🏋️  Workout Log Normalizer Demo")
    print("=" * 50)
    
    # Initialize the normalizer with your Excel file
    excel_file = "AlexWOLog.xlsx"
    
    try:
        print(f"\n📂 Loading: {excel_file}")
        normalizer = WorkoutLogNormalizer(excel_file)
        
        # Load the workbook
        normalizer.load_workbook()
        
        # Normalize all sheets
        df = normalizer.normalize_all_sheets()
        
        if not df.empty:
            print("\n" + "="*50)
            print("📈 RESULTS SUMMARY")
            print("="*50)
            
            summary = normalizer.get_summary()
            print(f"📊 Total entries: {summary['total_entries']}")
            print(f"🏋️  Unique exercises: {summary['unique_exercises']}")
            print(f"🗓️  Date range: {summary['date_range']}")
            print(f"📋 Training blocks: {', '.join(summary['blocks'])}")
            print(f"🏷️  Lift type breakdown:")
            for lift_type, count in summary['lift_types'].items():
                print(f"   • {lift_type}: {count} entries")
            
            print("\n" + "="*50)
            print("👀 DATA PREVIEW (First 10 rows)")
            print("="*50)
            print(df.head(10).to_string(index=False))
            
            # Export to CSV
            output_file = normalizer.export_csv()
            print(f"\n💾 Full data exported to: {output_file}")
            
            print("\n🎉 Demo complete! Check the CSV file for your normalized workout data.")
            
        else:
            print("⚠️  No data was found or parsed from the Excel file.")
            print("   This might happen if:")
            print("   • The sheet structure doesn't match the expected format")
            print("   • No dates were found in the first row")
            print("   • The column offsets don't align with the data")
            
    except FileNotFoundError:
        print(f"❌ Excel file '{excel_file}' not found in current directory.")
        print("   Make sure the file exists and try again.")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("   Check that your Excel file follows the expected format:")
        print("   • Dates in row 0 (weeks every 12 columns)")
        print("   • DAY labels followed by exercise data")
        print("   • Exercise, Load, Sets, Reps, RPE in expected column offsets")


if __name__ == "__main__":
    main() 