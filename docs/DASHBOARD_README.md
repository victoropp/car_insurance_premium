# Car Insurance Premium Analytics Dashboard

## Overview
A state-of-the-art interactive dashboard for analyzing car insurance premiums, visualizing model performance, and testing predictions with real-time interactive features.

## Features

### 1. Data Overview Section
- **Dataset Distribution Table**: Summary statistics for all features
- **Correlation Matrix**: Interactive heatmap showing feature relationships
- **Premium Distribution**: Histogram of insurance premium values
- **Feature Statistics**: Bar chart of mean values across features
- **Data Quality Indicator**: Gauge showing data completeness percentage
- **Feature Relationships**: Scatter plot with multi-dimensional visualization

### 2. Feature Analysis Section
- **Distribution Plots**: Violin plots for each feature showing distribution patterns
- **3D Scatter Plot**: Interactive 3D visualization of key relationships
- **Feature Importance**: Horizontal bar chart of feature rankings
- **Risk Segments**: Sunburst chart showing hierarchical risk categorization

### 3. Model Performance Dashboard
- **Model Comparison**: Side-by-side RMSE and R² comparisons
- **Training vs Validation**: Scatter plot showing overfitting analysis
- **Cross-Validation Performance**: Box plots of CV results
- **Model Rankings**: Top performing models by validation R²
- **Error Distribution**: Histogram of prediction errors
- **Performance Radar**: Multi-metric comparison for top 5 models
- **Complexity vs Performance**: Trade-off analysis

### 4. Prediction Analysis
- **Actual vs Predicted**: Scatter plot with perfect prediction line
- **Residual Plot**: Error analysis across predictions
- **Error Distribution**: Statistical distribution of residuals
- **Q-Q Plot**: Normality assessment of residuals
- **Prediction Intervals**: 95% confidence intervals
- **Error Metrics**: Interactive gauge showing R², MAE, and RMSE

### 5. Risk Profiling
- **Risk Distribution by Age**: Box plots across age groups
- **Risk Heatmap**: 2D heat matrix of age vs accidents
- **Risk Segments Pie**: Distribution of risk categories
- **Driver Profiles**: 3D visualization of driver characteristics
- **Premium Clusters**: K-means clustering visualization
- **Risk Factors**: Impact analysis of different risk factors

### 6. Interactive Model Testing Interface
- **Input Features**:
  - Driver Age (18-80 years)
  - Driver Experience (0-60 years)
  - Previous Accidents (0-10)
  - Annual Mileage (1-100k km)
  - Car Manufacturing Year (1990-2024)
  - Car Age (0-30 years)
  
- **Model Selection**: Choose from available trained models
- **Prediction Results**:
  - Predicted premium amount
  - Risk category classification
  - Percentile ranking
  - Key insights and comparisons
  
- **Analysis Tools**:
  - Feature Impact Analysis: Compare input values to averages
  - Sensitivity Analysis: See how changes in features affect predictions

## Installation

### Prerequisites
```bash
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn scipy joblib
```

### Files Required
1. `dashboard.py` - Main dashboard application
2. `visualizations.py` - Visualization engine module
3. `insurance_tranining_dataset.csv` - Training dataset
4. `model_results.csv` - Model performance metrics
5. `feature_importance.csv` - Feature importance rankings (optional)
6. Model files (`.pkl`) - Trained model files (optional)

## Usage

### Method 1: Direct Launch
```bash
python dashboard.py
```

### Method 2: Using Launch Script
```bash
python run_dashboard.py
```

### Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8050
```

## Dashboard Navigation

1. **Top Navigation Bar**: Quick links to different sections
2. **Summary Cards**: Key metrics at a glance
3. **Interactive Plots**: 
   - Hover for details
   - Zoom and pan capabilities
   - Download plot images
   - Reset view button

## Model Testing Guide

1. **Enter Input Values**: Adjust the sliders/inputs for each feature
2. **Select Model**: Choose from available trained models
3. **Click "Predict Premium"**: Generate prediction
4. **Review Results**:
   - Premium amount
   - Risk classification
   - Comparative analysis
   - Feature impact visualization

## Technical Details

### Architecture
- **Frontend**: Plotly Dash with Bootstrap components
- **Backend**: Python with scikit-learn models
- **Visualization**: Plotly Graph Objects
- **Data Processing**: Pandas and NumPy

### Performance Optimizations
- Efficient data caching
- Lazy loading of visualizations
- Optimized callback functions
- Responsive design for all screen sizes

## Customization

### Adding New Visualizations
Edit `visualizations.py` and add new methods to the `InsuranceVisualizationEngine` class.

### Modifying Dashboard Layout
Edit `dashboard.py` to change the layout structure or add new sections.

### Updating Color Schemes
Modify the `color_palette` and `template` attributes in the visualization engine.

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change port in `dashboard.py`: `app.run(debug=True, port=8051)`

2. **Missing Dependencies**
   - Install all required packages: `pip install -r requirements.txt`

3. **Model Files Not Found**
   - Ensure `.pkl` files are in the same directory as `dashboard.py`

4. **Data File Errors**
   - Verify CSV files are properly formatted and in the correct location

## Features Highlights

### Advanced Analytics
- Real-time prediction with multiple models
- Sensitivity analysis for feature impact
- Risk segmentation and profiling
- Performance benchmarking

### Interactive Elements
- Dynamic filtering and selection
- Responsive hover information
- Downloadable visualizations
- Customizable input parameters

### Professional Design
- Clean, modern interface
- Consistent color schemes
- Intuitive navigation
- Mobile-responsive layout

## Stop the Dashboard

Press `CTRL+C` in the terminal where the dashboard is running.

## Support

For issues or questions, check the console output for error messages and ensure all dependencies are properly installed.