#!/usr/bin/env python3
"""
Google Sheets Sync for Workout Log Normalizer
Automatically pulls new workout data from shared Google Sheets
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import argparse
from pathlib import Path

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

from workout_log_normalizer import WorkoutLogNormalizer
from data_cleaner import DataCleaner


class GoogleSheetsSync:
    """Sync workout data from Google Sheets to local files."""
    
    def __init__(self, credentials_path=None, spreadsheet_url=None, config_file="sheets_config.json"):
        """Initialize Google Sheets sync."""
        self.credentials_path = credentials_path
        self.spreadsheet_url = spreadsheet_url
        self.config_file = config_file
        self.gc = None
        self.spreadsheet = None
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def setup_authentication(self, credentials_path):
        """Set up Google Sheets authentication."""
        if not GSPREAD_AVAILABLE:
            raise Exception("Google Sheets integration requires: pip install gspread google-auth")
        
        print("🔐 Setting up Google Sheets authentication...")
        
        # Define the scopes
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        
        try:
            # Load credentials from service account file
            credentials = Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )
            
            # Initialize gspread client
            self.gc = gspread.authorize(credentials)
            print("✅ Authentication successful!")
            
            # Save credentials path to config
            self.config['credentials_path'] = credentials_path
            self.save_config()
            
            return True
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("\n💡 To set up Google Sheets access:")
            print("1. Go to Google Cloud Console (console.cloud.google.com)")
            print("2. Create a new project or select existing")
            print("3. Enable Google Sheets API and Google Drive API")
            print("4. Create a Service Account")
            print("5. Download the JSON credentials file")
            print("6. Share your spreadsheet with the service account email")
            return False
    
    def connect_spreadsheet(self, spreadsheet_url):
        """Connect to the Google Spreadsheet."""
        if not self.gc:
            if not self.setup_authentication(self.credentials_path):
                return False
        
        try:
            print(f"📊 Connecting to spreadsheet...")
            
            # Extract spreadsheet ID from URL
            if '/d/' in spreadsheet_url:
                sheet_id = spreadsheet_url.split('/d/')[1].split('/')[0]
            else:
                sheet_id = spreadsheet_url
            
            self.spreadsheet = self.gc.open_by_key(sheet_id)
            print(f"✅ Connected to: {self.spreadsheet.title}")
            
            # Save spreadsheet info to config
            self.config['spreadsheet_url'] = spreadsheet_url
            self.config['spreadsheet_title'] = self.spreadsheet.title
            self.config['last_sync'] = None
            self.save_config()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to spreadsheet: {e}")
            print("\n💡 Make sure:")
            print("1. The spreadsheet URL is correct")
            print("2. The spreadsheet is shared with your service account email")
            print("3. The service account has 'Viewer' permissions")
            return False
    
    def list_worksheets(self):
        """List all worksheets in the spreadsheet."""
        if not self.spreadsheet:
            print("❌ Not connected to spreadsheet")
            return []
        
        worksheets = self.spreadsheet.worksheets()
        print(f"\n📋 Found {len(worksheets)} worksheets:")
        for i, ws in enumerate(worksheets):
            print(f"  {i+1}. {ws.title}")
        
        return worksheets
    
    def download_worksheet_as_excel(self, output_file="downloaded_workout_log.xlsx"):
        """Download all worksheets as an Excel file."""
        if not self.spreadsheet:
            print("❌ Not connected to spreadsheet")
            return None
        
        print(f"📥 Downloading all worksheets to {output_file}...")
        
        try:
            # Create Excel writer
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                worksheets = self.spreadsheet.worksheets()
                
                for ws in worksheets:
                    print(f"  📋 Downloading: {ws.title}")
                    
                    # Get all values from worksheet
                    data = ws.get_all_values()
                    
                    if data:
                        # Convert to DataFrame
                        df = pd.DataFrame(data)
                        
                        # Write to Excel
                        sheet_name = ws.title[:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
            print(f"✅ Downloaded to: {output_file}")
            
            # Update last sync time
            self.config['last_sync'] = datetime.now()
            self.config['last_download_file'] = output_file
            self.save_config()
            
            return output_file
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def sync_and_normalize(self, apply_validation=True, output_suffix=""):
        """Download, normalize, and clean the data in one step."""
        print(f"\n🔄 STARTING FULL SYNC AND NORMALIZATION")
        print(f"{'='*60}")
        
        # Step 1: Download from Google Sheets
        excel_file = self.download_worksheet_as_excel()
        if not excel_file:
            return None
        
        # Step 2: Normalize the data
        print(f"\n🔧 Normalizing workout data...")
        normalizer = WorkoutLogNormalizer(excel_file)
        normalizer.load_workbook()
        df = normalizer.normalize_all_sheets()
        
        if df.empty:
            print("⚠️  No data was normalized")
            return None
        
        # Step 3: Save normalized data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        normalized_file = f"workout_log_normalized_{timestamp}{output_suffix}.csv"
        df.to_csv(normalized_file, index=False)
        print(f"💾 Saved normalized data: {normalized_file}")
        
        # Step 4: Apply validation and cleaning if requested
        if apply_validation:
            print(f"\n🧹 Applying validation and cleaning...")
            cleaner = DataCleaner(normalized_file)
            cleaner.load_data()
            
            # Create validated dataset
            df_clean, removed = cleaner.create_clean_dataset(
                remove_completely_blank=True,
                remove_header_like=True,
                remove_suspicious=True
            )
            
            validated_file = f"workout_log_validated_{timestamp}{output_suffix}.csv"
            cleaner.save_clean_data(df_clean, validated_file)
            
            print(f"✅ Pipeline complete!")
            print(f"📊 Raw normalized: {normalized_file}")
            print(f"🎯 Validated data: {validated_file}")
            
            return {
                'raw': normalized_file,
                'validated': validated_file,
                'entries': len(df_clean)
            }
        
        return {
            'raw': normalized_file,
            'entries': len(df)
        }
    
    def check_for_updates(self):
        """Check if there are new entries since last sync."""
        if not self.config.get('last_sync'):
            print("🆕 No previous sync found - full sync recommended")
            return True
        
        last_sync = datetime.fromisoformat(str(self.config['last_sync']))
        days_since = (datetime.now() - last_sync).days
        
        print(f"📅 Last sync: {last_sync.strftime('%Y-%m-%d %H:%M:%S')} ({days_since} days ago)")
        
        if days_since > 0:
            print("🔄 Updates likely available")
            return True
        else:
            print("✅ Recently synced")
            return False
    
    def setup_scheduled_sync(self):
        """Provide instructions for setting up scheduled syncing."""
        print(f"\n⏰ SCHEDULED SYNC SETUP")
        print(f"{'='*60}")
        
        script_path = os.path.abspath(__file__)
        
        print("To automatically sync your workout data:")
        print("\n🪟 Windows (Task Scheduler):")
        print("1. Open Task Scheduler")
        print("2. Create Basic Task")
        print("3. Set trigger (e.g., Daily at 6 PM)")
        print(f"4. Action: Start a program")
        print(f"   Program: python")
        print(f"   Arguments: \"{script_path}\" --sync")
        print(f"   Start in: \"{os.path.dirname(script_path)}\"")
        
        print("\n🐧 Linux/Mac (Cron):")
        print("1. Run: crontab -e")
        print("2. Add line:")
        print(f"   0 18 * * * cd {os.path.dirname(script_path)} && python {script_path} --sync")
        print("   (Runs daily at 6 PM)")
        
        print("\n📋 Manual sync anytime:")
        print(f"   python {script_path} --sync")


def main():
    """Command line interface for Google Sheets sync."""
    parser = argparse.ArgumentParser(description='Sync workout data from Google Sheets')
    parser.add_argument('--setup', action='store_true', help='Set up Google Sheets authentication')
    parser.add_argument('--credentials', help='Path to Google service account JSON file')
    parser.add_argument('--spreadsheet', help='Google Sheets URL or ID')
    parser.add_argument('--sync', action='store_true', help='Download and normalize data')
    parser.add_argument('--check', action='store_true', help='Check for updates since last sync')
    parser.add_argument('--schedule', action='store_true', help='Show scheduled sync setup instructions')
    parser.add_argument('--no-validation', action='store_true', help='Skip validation rules')
    
    args = parser.parse_args()
    
    # Initialize sync manager
    sync = GoogleSheetsSync()
    
    if not GSPREAD_AVAILABLE:
        print("❌ Google Sheets integration not available")
        print("📦 Install required packages:")
        print("   pip install gspread google-auth")
        return 1
    
    try:
        if args.setup:
            if not args.credentials:
                args.credentials = input("Enter path to service account JSON file: ").strip()
            if not args.spreadsheet:
                args.spreadsheet = input("Enter Google Sheets URL: ").strip()
            
            # Set up authentication and connection
            if sync.setup_authentication(args.credentials):
                if sync.connect_spreadsheet(args.spreadsheet):
                    sync.list_worksheets()
                    print(f"\n✅ Setup complete! You can now run:")
                    print(f"   python {__file__} --sync")
        
        elif args.sync:
            # Load config
            if 'credentials_path' in sync.config:
                sync.setup_authentication(sync.config['credentials_path'])
            if 'spreadsheet_url' in sync.config:
                sync.connect_spreadsheet(sync.config['spreadsheet_url'])
            
            if sync.spreadsheet:
                apply_validation = not args.no_validation
                result = sync.sync_and_normalize(apply_validation=apply_validation)
                if result:
                    print(f"\n🎉 Sync successful! {result['entries']} entries processed")
            else:
                print("❌ Run --setup first to configure Google Sheets access")
        
        elif args.check:
            sync.check_for_updates()
        
        elif args.schedule:
            sync.setup_scheduled_sync()
        
        else:
            # Default: show status and options
            print("🏋️  Google Sheets Workout Log Sync")
            print("=" * 50)
            
            if sync.config:
                print("📊 Configuration:")
                if 'spreadsheet_title' in sync.config:
                    print(f"  Spreadsheet: {sync.config['spreadsheet_title']}")
                if 'last_sync' in sync.config and sync.config['last_sync']:
                    print(f"  Last sync: {sync.config['last_sync']}")
                
                print(f"\n🔄 Quick sync: python {__file__} --sync")
            else:
                print("⚙️  Not configured yet")
                print(f"🚀 Setup: python {__file__} --setup")
            
            print(f"\n📖 All options:")
            print(f"  --setup: Configure Google Sheets access")
            print(f"  --sync: Download and process latest data")
            print(f"  --check: Check for updates")
            print(f"  --schedule: Show automatic sync setup")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 