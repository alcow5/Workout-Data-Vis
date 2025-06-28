# 💪 Powerlifting Data Analysis & Visualization Dashboard

A comprehensive powerlifting training data analysis system with predictive analytics, competition integration, and beautiful visualizations.
![workout_vis_ex3](https://github.com/user-attachments/assets/ea17da60-34e8-4437-9c83-4dfb311b808f)
![workout_vis_ex2](https://github.com/user-attachments/assets/824dea45-b5c7-4718-bfd3-d0ed6d2d0976)
![workout_vis_ex1](https://github.com/user-attachments/assets/45e9d256-2438-48b3-8fbd-b3d69bd4ea24)

## 🌟 Features

### 📊 **Data Processing & Normalization**
- **Excel Workout Log Parser**: Converts multi-week training blocks from Excel into normalized CSV format
- **Smart Data Validation**: Configurable rules for RPE, sets, reps, and load validation
- **1RM Calculations**: Multiple formulas (Epley, Brzycki, Lombardi, RPE-based) for strength estimation

### 🔮 **Predictive Analytics**
- **Trajectory Forecasting**: ML-powered strength progression predictions
- **Goal Timeline Analysis**: Realistic timelines for achieving target 1RMs
- **Competition Attempt Suggestions**: AI-powered meet planning with opener/second/third recommendations
- **Training Optimization**: Analysis of optimal rep ranges, frequency, and volume

### 🏆 **Competition Integration**
- **OpenPowerlifting Database**: Integration with 3.6M+ competition results
- **Meet vs Training Comparison**: Side-by-side analysis of competition performance vs training estimates
- **Competition History Tracking**: Personal competition results visualization

### 📈 **Interactive Dashboard**
- **Beautiful Streamlit Interface**: Professional web-based dashboard
- **Real-time Visualizations**: Interactive charts with Plotly
- **PR Tracking**: Personal record heatmaps by rep range
- **Training Consistency Analysis**: Workout frequency and volume trends
- **RPE Trend Analysis**: Training intensity tracking over time

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Dashboard
```bash
streamlit run workout_dashboard.py
```

### Data Processing Pipeline
1. **Normalize Excel Data**:
   ```bash
   python workout_log_normalizer.py your_workout_log.xlsx
   ```

2. **Clean & Validate Data**:
   ```bash
   python data_cleaner.py
   ```

3. **Launch Dashboard**:
   ```bash
   streamlit run workout_dashboard.py
   ```

## 📁 Project Structure

```
powerlifting-analysis/
├── 📊 Core Analysis
│   ├── workout_log_normalizer.py    # Excel → CSV converter
│   ├── one_rm_calculator.py         # 1RM estimation engine
│   ├── data_cleaner.py             # Data validation & cleaning
│   └── spot_checker.py             # Accuracy verification tool
├── 🔮 Predictive Analytics
│   ├── predictive_analytics.py      # Original ML predictor
│   ├── improved_predictor.py        # Enhanced predictor with filtering
│   ├── diagnostic_predictions.py    # Model debugging tools
│   └── all_lifts_reality_check.py  # Prediction validation
├── 🏆 Competition Integration
│   ├── meet_data_integrator.py     # OpenPowerlifting integration
│   └── google_sheets_sync.py       # Google Sheets automation
├── 📈 Visualization
│   ├── workout_dashboard.py        # Main Streamlit dashboard
│   └── demo.py                     # Feature demonstrations
├── 📋 Documentation
│   ├── README.md                   # This file
│   ├── dashboard_guide.md          # Dashboard usage guide
│   └── google_sheets_setup_guide.md # Setup instructions
└── ⚙️ Configuration
    ├── requirements.txt            # Python dependencies
    └── .gitignore                 # Git exclusions
```

## 🎯 Key Capabilities

### **Strength Progression Analysis**
- Track estimated 1RM over time for squat, bench, deadlift
- Compare training performance vs competition results
- Identify training patterns that drive progress

### **Predictive Modeling**
- **Improved Algorithm**: Filters to main exercises and strength-focused rep ranges (1-6 reps)
- **Realistic Forecasting**: Monthly strength gain predictions
- **Goal Planning**: Timeline estimation for target 1RMs

### **Training Optimization**
- **Rep Range Effectiveness**: Which rep ranges drive your progress
- **Optimal Frequency**: How often to train each lift
- **Volume Sweet Spot**: Ideal weekly volume recommendations

### **Competition Planning**
- **Meet Attempt Calculator**: Conservative opener/second/aggressive third suggestions
- **Performance Prediction**: Confidence-rated competition forecasts
- **Historical Analysis**: Competition trends and improvements

## 📊 Example Outputs

The system provides insights like:
- **"Squat progressing at +4.2 lbs/month"**
- **"600 lb goal achievable in 14.9 months"**
- **"Most effective rep range: 1-3 reps (Strength)"**
- **"Optimal training frequency: 3x per week"**

## 🛠️ Technical Features

### **Data Processing**
- Handles complex Excel layouts with multi-week training blocks
- Intelligent date parsing and exercise categorization
- Robust error handling and data validation

### **Machine Learning**
- Scikit-learn powered regression models
- Outlier detection and removal
- Multiple model comparison and selection

### **Visualization**
- Interactive Plotly charts
- Custom CSS styling for professional appearance
- Responsive design for various screen sizes

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- streamlit, plotly
- openpyxl (Excel support)
- gspread, google-auth (Google Sheets integration)

## 🎓 Usage Examples

### Basic Analysis
```python
from one_rm_calculator import WorkoutAnalyzer

analyzer = WorkoutAnalyzer('your_data.csv')
analyzer.load_data()
analyzer.calculate_1rm_estimates()

# Get max 1RMs by lift
max_1rms = analyzer.get_max_1rm_by_lift()
print(max_1rms)
```

### Predictive Analytics
```python
from improved_predictor import ImprovedPowerliftingPredictor

predictor = ImprovedPowerliftingPredictor(data)
trajectory = predictor.improved_trajectory_analysis('Squat')
print(f"Monthly gain: {trajectory['monthly_gain']:.1f} lbs/month")
```

### Competition Integration
```python
from meet_data_integrator import OpenPowerliftingIntegrator

integrator = OpenPowerliftingIntegrator()
meets = integrator.get_lifter_meets(opl_data, "Your Name")
```

## 🏆 Success Stories

This system has been used to:
- ✅ Accurately predict strength progressions
- ✅ Optimize training programs based on data
- ✅ Plan successful competition attempts
- ✅ Identify the most effective training methods

## 🤝 Contributing

Contributions welcome! This project demonstrates:
- Advanced data processing techniques
- Machine learning applications in sports
- Interactive dashboard development
- Integration with external APIs

## 📄 License

Open source - feel free to use and modify for your own training analysis!

---

**Built with ❤️ for the powerlifting community**

*Transform your training data into actionable insights and achieve your strength goals!* 💪 
