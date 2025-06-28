# 🔗 Google Sheets Integration Setup

This guide will help you set up automatic syncing from your shared workout Google Sheet.

## 📋 Prerequisites

1. **Google Account** with access to your workout spreadsheet
2. **Python environment** with required packages

## 🚀 Quick Setup

### Step 1: Install Required Packages
```bash
pip install gspread google-auth
```

### Step 2: Set Up Google Cloud Project

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create or Select Project**:
   - Click the project dropdown
   - Create new project or select existing
   - Name it something like "Workout Log Sync"

3. **Enable APIs**:
   - Go to "APIs & Services" → "Library"
   - Search and enable: **Google Sheets API**
   - Search and enable: **Google Drive API**

### Step 3: Create Service Account

1. **Go to "APIs & Services" → "Credentials"**
2. **Click "Create Credentials" → "Service Account"**
3. **Fill out details**:
   - Name: `workout-log-sync`
   - Description: `Service account for workout log data sync`
   - Click "Create and Continue"
4. **Skip roles** (click "Continue")
5. **Skip user access** (click "Done")

### Step 4: Download Credentials

1. **Find your service account** in the credentials list
2. **Click the email address** to open details
3. **Go to "Keys" tab**
4. **Click "Add Key" → "Create New Key"**
5. **Choose JSON format** and click "Create"
6. **Save the file** as `workout_sync_credentials.json` in your project folder

### Step 5: Share Your Spreadsheet

1. **Open your workout Google Sheet**
2. **Click "Share" button**
3. **Copy the service account email** from the JSON file
4. **Add the email** with "Viewer" permissions
5. **Copy the spreadsheet URL**

## 🔧 Configuration

```bash
# First-time setup
python google_sheets_sync.py --setup

# Test the connection
python google_sheets_sync.py --sync
```

## 📊 Usage Commands

```bash
# Manual sync
python google_sheets_sync.py --sync

# Check for updates
python google_sheets_sync.py --check

# Setup automatic syncing
python google_sheets_sync.py --schedule
```

---

🎉 **You're all set!** Your workout data will now automatically sync. 