# 💪 Workout Analysis Dashboard Guide

Your comprehensive strength training dashboard with **meet data integration**! Track training progression alongside competition performance.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch dashboard:**
   ```bash
   streamlit run workout_dashboard.py
   ```

3. **Access at:** `http://localhost:8501`

## 🏆 NEW: Meet Data Integration

### **Competition Results Integration**
- **Automatic Download**: Downloads OpenPowerlifting database (3.6M+ entries)
- **Name Search**: Search by your competition name
- **Visual Integration**: Meet results plotted as **gold stars** alongside training data
- **Performance Comparison**: See how training translates to competition

### **How to Use Meet Data**
1. In the sidebar, find "🏆 Meet Data" section
2. Enter your name as it appears in competition results
3. Click "🔄 Update Meet Data" to refresh
4. Your meet results will automatically appear in charts

### **Supported Federations**
- All IPF affiliates (USAPL, CPU, etc.)
- USPA, RPS, SPF, and 400+ other federations
- **Data Range**: 1964 to present (updated nightly)

## 📊 Dashboard Features

### **Row 1: Current Status Cards**
- **Max 1RMs**: Your estimated peak performance
- **90-Day Progress**: Recent strength gains
- **Compact Design**: All key metrics at a glance

### **Row 2: PR Analysis & Progression**
- **🏆 Real PR Heatmap**: Actual personal records by rep range (1-10 reps)
  - Based on real training data, not estimates
  - Color-coded by weight lifted
  - Easy to spot strength patterns
  
- **📈 Training vs Competition**: Side-by-side comparison
  - Training progression (estimated 1RM lines)
  - Competition results (gold stars)
  - Hover for detailed meet information

### **Row 3: Comparisons & Volume**
- **🎯 Max 1RM Bars**: Quick comparison across lifts
- **📊 Weekly Volume**: Training load over time
  - Smoothed weekly averages
  - Volume = Sets × Reps × Weight

### **Row 4: Recent Data**
- **📋 Training PRs**: Last 6 months of personal records
- **🏆 Competition Results**: Recent meet performances
- **📖 Method Explanations**: 1RM calculation info

## 🔧 Configuration Options

### **Sidebar Controls**
- **📁 Data File**: Choose your workout CSV
- **🏋️ Lift Selection**: Focus on specific movements
- **📅 Date Range**: Filter by time period  
- **🧮 1RM Method**: Choose calculation approach
- **🏆 Meet Data**: Enter competition name

### **1RM Calculation Methods**
- **Best**: Auto-selects optimal formula
- **RPE-Based**: Uses RPE percentage tables (most accurate)
- **Epley**: weight × (1 + reps/30) 
- **Brzycki**: weight × (36/(37-reps))
- **Lombardi**: weight × reps^0.10

## 🎯 Key Benefits

### **Real Performance Tracking**
- **Actual PRs**: Based on real lifts, not calculations
- **Training Effectiveness**: See how gym work translates to meets
- **Pattern Recognition**: Identify your strongest rep ranges
- **Progress Validation**: Confirm estimates with competition results

### **Comprehensive Analysis**
- **3+ Years of Data**: Complete training history
- **Multiple Metrics**: 1RM, volume, frequency, RPE trends
- **Visual Clarity**: Color-coded, interactive charts
- **Professional Quality**: Publication-ready visualizations

## 📁 File Requirements

### **Training Data Format**
Your CSV should include:
- `Date`, `Exercise`, `Sets`, `Reps`, `Load`, `RPE`
- Automatically categorizes into Squat/Bench/Deadlift/Accessory
- Supports multiple naming conventions

### **Meet Data Format**
- **Automatic**: Downloaded from OpenPowerlifting
- **No Manual Entry**: Just provide your competition name
- **Global Coverage**: 400+ federations worldwide

## 🔍 Troubleshooting

### **No Meet Results Found**
- Try name variations (first name only, nickname, etc.)
- Check if your federation reports to OpenPowerlifting
- Verify spelling matches competition records
- Some smaller/local meets may not be tracked

### **Slow Performance**
- First-time download is ~145MB (may take 2-3 minutes)
- Subsequent loads use cached data
- Use date filters to improve chart performance

### **Missing Training Data**
- Ensure CSV has required columns
- Check date format (YYYY-MM-DD preferred)
- Verify lift names match common patterns

## 🚀 Advanced Features

### **Data Caching**
- Meet data cached locally in `opl_cache/` folder
- Updates nightly from OpenPowerlifting
- Force refresh with "🔄 Update Meet Data" button

### **Performance Optimization**
- Streamlit caching for fast reloads
- Compressed data storage
- Efficient date filtering

### **Export Options**
- All charts support PNG/SVG export
- Data tables can be downloaded as CSV
- Right-click charts for save options

## 💡 Pro Tips

1. **Use RPE-Based Method**: Most accurate for powerlifting
2. **Filter Date Ranges**: Focus on specific training blocks
3. **Check Meet Integration**: Validates your 1RM estimates
4. **Monitor Volume Trends**: Avoid overtraining patterns
5. **Track Rep Range PRs**: Find your strength sweet spots

## 🎪 What's Next?

Your dashboard now provides a complete picture of your strength journey:
- **Training progression** from daily workouts
- **Competition validation** from official meets  
- **Pattern analysis** across multiple years
- **Performance prediction** for future goals

Train smart, compete strong! 💪

---

*Dashboard powered by your training data and OpenPowerlifting's comprehensive competition database.* 