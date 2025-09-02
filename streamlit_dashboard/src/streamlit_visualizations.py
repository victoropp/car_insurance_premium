"""
Streamlit-Optimized Visualization Engine for Insurance Premium Analytics
Videbimus AI - https://www.videbimusai.com

This module provides optimized visualizations for Streamlit deployment with memory efficiency
and caching support. All charts are designed to work seamlessly with Streamlit's architecture.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st


class StreamlitVisualizationEngine:
    """Production-grade visualization engine optimized for Streamlit deployment"""
    
    def __init__(self):
        """Initialize the visualization engine with Streamlit-optimized settings"""
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#73AB84',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C91C2',
            'light': '#F5F5F5',
            'dark': '#2D3436'
        }
        
        self.template = 'plotly_white'
        self.font_family = 'Arial, sans-serif'
        
        # Load and cache data
        self._load_data()
    
    @st.cache_data
    def _load_data(_self):
        """Load and cache all required data with memory optimization"""
        try:
            # Load training dataset with optimizations
            training_data = pd.read_csv('data/insurance_tranining_dataset.csv')
            
            # Load model results
            model_results = pd.read_csv('data/model_results.csv')
            
            # Load test results
            test_results = pd.read_csv('data/final_test_results.csv')
            
            # Load feature importance
            feature_importance = pd.read_csv('data/feature_importance.csv')
            
            # Load holdout predictions
            predictions_holdout = pd.read_csv('data/predictions_holdout_test.csv')
            
            # Memory optimization - convert numeric columns to appropriate types
            if 'Driver Age' in training_data.columns:
                training_data['Driver Age'] = training_data['Driver Age'].astype('int32')
            if 'Car Age' in training_data.columns:
                training_data['Car Age'] = training_data['Car Age'].astype('int32')
            
            # Store data as instance variables
            _self.training_data = training_data
            _self.model_results = model_results
            _self.test_results = test_results  
            _self.feature_importance = feature_importance
            _self.predictions_holdout = predictions_holdout
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def create_executive_summary(self):
        """Create executive summary with 6 high-level visualizations"""
        
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '<b>💰 Premium Distribution Overview</b>',
                '<b>🎯 Model Performance (Test Set)</b>',
                '<b>🔥 Premium vs Age Analysis</b>',
                '<b>⚠️ Accident Risk Impact on Premium</b>',
                '<b>⚡ Experience Impact Analysis</b>',
                '<b>🏆 Feature Importance Ranking</b>'
            ),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # 1. Premium Distribution Overview
        fig.add_trace(
            go.Histogram(
                x=self.training_data['Insurance Premium ($)'],
                nbinsx=30,
                name='Premium Distribution',
                marker=dict(
                    color=self.colors['primary'],
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>Premium Range:</b> $%{x}<br><b>Count:</b> %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Model Performance Summary (Top 3 models - Test Set)
        top3_models = self.test_results.nlargest(3, 'Test_R2')
        fig.add_trace(
            go.Bar(
                x=top3_models['Model'],
                y=top3_models['Test_R2'],
                name='Model Performance',
                marker=dict(
                    color=['#2E86AB', '#A23B72', '#F18F01'],  # Distinct colors for each model
                    line=dict(color='white', width=2)
                ),
                text=[f'{x:.4f}' for x in top3_models['Test_R2']],
                textposition='outside',
                textfont=dict(size=12),
                hovertemplate='<b>%{x}</b><br>Test R²: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Premium vs Age Analysis
        fig.add_trace(
            go.Scatter(
                x=self.training_data['Driver Age'],
                y=self.training_data['Insurance Premium ($)'],
                mode='markers',
                name='Age vs Premium',
                marker=dict(
                    color=self.training_data['Insurance Premium ($)'],
                    colorscale='Viridis',
                    showscale=False,
                    size=6,
                    opacity=0.6,
                    line=dict(color='white', width=0.5)
                ),
                hovertemplate='<b>Age:</b> %{x}<br><b>Premium:</b> $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Risk Factor Analysis - Accidents Impact on Premium
        # Show how accidents affect premiums - much more meaningful
        accident_analysis = self.training_data.groupby('Previous Accidents')['Insurance Premium ($)'].agg(['mean', 'std', 'count']).reset_index()
        accident_analysis.columns = ['Accidents', 'avg_premium', 'std', 'count']
        
        # Calculate percentage increase from baseline (0 accidents)
        baseline_premium = accident_analysis.loc[accident_analysis['Accidents'] == 0, 'avg_premium'].values[0]
        accident_analysis['pct_increase'] = ((accident_analysis['avg_premium'] - baseline_premium) / baseline_premium * 100)
        
        # Create gradient colors based on risk level
        colors = ['#2E86AB', '#73AB84', '#F18F01', '#C73E1D', '#8B0000', '#4B0082'][:len(accident_analysis)]
        
        fig.add_trace(
            go.Bar(
                x=accident_analysis['Accidents'].astype(str),
                y=accident_analysis['avg_premium'],
                name='Premium by Accidents',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=[f'${p:.0f}<br>+{inc:.1f}%' if inc > 0 else f'${p:.0f}<br>Base' 
                      for p, inc in zip(accident_analysis['avg_premium'], accident_analysis['pct_increase'])],
                textposition='outside',
                textfont=dict(size=11, color='black'),
                hovertemplate='<b>%{x} Previous Accidents</b><br>Avg Premium: $%{y:.2f}<br>Increase: %{text}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Experience Impact Analysis  
        fig.add_trace(
            go.Scatter(
                x=self.training_data['Driver Experience'],
                y=self.training_data['Insurance Premium ($)'],
                mode='markers',
                name='Experience Impact',
                marker=dict(
                    color=self.colors['info'],
                    size=6,
                    opacity=0.6,
                    line=dict(color='white', width=0.5)
                ),
                hovertemplate='<b>Experience:</b> %{x} years<br><b>Premium:</b> $%{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Feature Importance Ranking (Top 8)
        top_features = self.feature_importance.nlargest(8, 'Importance')
        fig.add_trace(
            go.Bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                name='Feature Importance',
                marker=dict(
                    color=top_features['Importance'],
                    colorscale=[[0, self.colors['light']], [1, self.colors['primary']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=[f'{x:.3f}' for x in top_features['Importance']],
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            template=self.template,
            font=dict(family=self.font_family, size=11),
            margin=dict(t=50, b=80, l=80, r=80),
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Premium ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2, tickangle=0)
        fig.update_yaxes(title_text="Test R² Score", row=1, col=2, range=[0.99, 1.0])
        fig.update_xaxes(title_text="Age", row=2, col=1)
        fig.update_yaxes(title_text="Premium ($)", row=2, col=1)
        fig.update_xaxes(title_text="Number of Previous Accidents", row=2, col=2)
        fig.update_yaxes(title_text="Average Premium ($)", row=2, col=2, range=[485, 500])
        fig.update_xaxes(title_text="Years of Experience", row=3, col=1)
        fig.update_yaxes(title_text="Premium ($)", row=3, col=1)
        fig.update_xaxes(title_text="Importance Score", row=3, col=2)
        fig.update_yaxes(title_text="Feature", row=3, col=2)
        
        return fig
    
    def create_detailed_analysis(self):
        """Create detailed analysis with 6 in-depth visualizations"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '<b>👥 Experience Level Distribution</b>',
                '<b>🚗 Vehicle Age Impact</b>',
                '<b>⚠️ Accident History Analysis</b>',
                '<b>🛣️ Annual Mileage Patterns</b>',
                '<b>📈 Premium Correlation Matrix</b>',
                '<b>🎯 Top Feature Insights</b>'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'box'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # 1. Experience Level Distribution
        exp_groups = pd.cut(self.training_data['Driver Experience'], 
                           bins=[0, 2, 5, 10, float('inf')], 
                           labels=['New (0-2)', 'Junior (2-5)', 'Mid (5-10)', 'Senior (10+)'])
        exp_analysis = self.training_data.groupby(exp_groups)['Insurance Premium ($)'].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=self.training_data['Driver Experience'],
                y=self.training_data['Insurance Premium ($)'],
                mode='markers',
                marker=dict(
                    color=self.training_data['Driver Experience'],
                    colorscale='RdYlBu_r',
                    showscale=False,
                    size=6,
                    opacity=0.6
                ),
                name='Experience Analysis',
                hovertemplate='<b>Experience:</b> %{x} years<br><b>Avg Premium:</b> $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Vehicle Age Impact
        vehicle_age_analysis = self.training_data.groupby('Car Age').agg({
            'Insurance Premium ($)': ['mean', 'count']
        }).round(2)
        vehicle_age_analysis.columns = ['avg_premium', 'count']
        vehicle_age_analysis = vehicle_age_analysis.reset_index()
        
        # Filter for reasonable sample sizes
        vehicle_age_analysis = vehicle_age_analysis[vehicle_age_analysis['count'] >= 5]
        
        fig.add_trace(
            go.Scatter(
                x=vehicle_age_analysis['Car Age'],
                y=vehicle_age_analysis['avg_premium'],
                mode='markers+lines',
                marker=dict(
                    color=vehicle_age_analysis['avg_premium'],
                    colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                    size=8,
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                line=dict(color=self.colors['primary'], width=2),
                text=[f'n={int(x)}' for x in vehicle_age_analysis['count']],
                textposition='top center',
                name='Vehicle Age Impact',
                hovertemplate='<b>Vehicle Age:</b> %{x} years<br><b>Avg Premium:</b> $%{y:.2f}<br><b>Sample Size:</b> %{text}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Accident History Analysis - Enhanced visualization
        accident_analysis = self.training_data.groupby('Previous Accidents').agg({
            'Insurance Premium ($)': ['mean', 'std', 'count']
        }).round(2)
        accident_analysis.columns = ['avg_premium', 'std_premium', 'count']
        accident_analysis = accident_analysis.reset_index()
        
        # Calculate percentage increase from zero accidents baseline
        baseline = accident_analysis.loc[accident_analysis['Previous Accidents'] == 0, 'avg_premium'].values[0]
        accident_analysis['pct_increase'] = ((accident_analysis['avg_premium'] - baseline) / baseline * 100).round(2)
        
        # Create descriptive labels with all information
        accident_analysis['label_text'] = [
            f'${avg:.0f}<br>Base Rate<br>(n={int(n)})' if acc == 0 
            else f'${avg:.0f}<br>+{pct:.1f}%<br>(n={int(n)})'
            for acc, avg, pct, n in zip(
                accident_analysis['Previous Accidents'],
                accident_analysis['avg_premium'],
                accident_analysis['pct_increase'],
                accident_analysis['count']
            )
        ]
        
        # Create risk-based colors
        risk_colors = ['#2E86AB', '#73AB84', '#FFB700', '#FF6B6B', '#DC143C', '#8B0000']
        
        fig.add_trace(
            go.Bar(
                x=accident_analysis['Previous Accidents'].astype(str),
                y=accident_analysis['avg_premium'],
                marker=dict(
                    color=risk_colors[:len(accident_analysis)],
                    line=dict(color='white', width=2)
                ),
                text=accident_analysis['label_text'],
                textposition='outside',
                textfont=dict(size=10, color='black'),
                name='Accident Impact',
                hovertemplate='<b>%{x} Previous Accidents</b><br>Average Premium: $%{y:.2f}<br>Increase from base: +%{customdata:.1f}%<br>Sample size: %{text}<extra></extra>',
                customdata=accident_analysis['pct_increase']
            ),
            row=2, col=1
        )
        
        # 4. Annual Mileage Patterns (Box Plot)
        # Create mileage ranges
        mileage_ranges = pd.cut(self.training_data['Annual Mileage (x1000 km)'], 
                               bins=[0, 10, 15, 20, 25, float('inf')],
                               labels=['Low (0-10)', 'Moderate (10-15)', 'High (15-20)', 'Very High (20-25)', 'Extreme (25+)'])
        
        for i, range_name in enumerate(['Low (0-10)', 'Moderate (10-15)', 'High (15-20)', 'Very High (20-25)', 'Extreme (25+)']):
            if range_name in mileage_ranges.values:
                data = self.training_data[mileage_ranges == range_name]['Insurance Premium ($)']
                if len(data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=data,
                            name=range_name,
                            marker=dict(color=self.colors['info']),
                            boxmean=True,
                            hovertemplate='<b>Mileage: %{x}</b><br>Premium: $%{y:.2f}<extra></extra>'
                        ),
                        row=2, col=2
                    )
        
        # 5. Premium Correlation Matrix (Heatmap)
        numeric_columns = ['Driver Age', 'Driver Experience', 'Car Age', 'Previous Accidents', 
                          'Annual Mileage (x1000 km)', 'Insurance Premium ($)']
        correlation_matrix = self.training_data[numeric_columns].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                showscale=True,
                text=correlation_matrix.round(3).values,
                texttemplate='%{text}',
                textfont=dict(size=10),
                name='Correlation Matrix',
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Top Feature Insights (Top 6 features)
        top6_features = self.feature_importance.nlargest(6, 'Importance')
        
        fig.add_trace(
            go.Bar(
                x=top6_features['Importance'],
                y=top6_features['Feature'],
                orientation='h',
                marker=dict(
                    color=top6_features['Importance'],
                    colorscale=[[0, self.colors['light']], [1, self.colors['primary']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=[f'{x:.3f}' for x in top6_features['Importance']],
                textposition='outside',
                name='Feature Importance',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=False,
            template=self.template,
            font=dict(family=self.font_family, size=11),
            margin=dict(t=50, b=80, l=80, r=80),
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Years of Experience", row=1, col=1)
        fig.update_yaxes(title_text="Insurance Premium ($)", row=1, col=1)
        fig.update_xaxes(title_text="Vehicle Age (Years)", row=1, col=2)
        fig.update_yaxes(title_text="Average Premium ($)", row=1, col=2)
        fig.update_xaxes(title_text="Number of Previous Accidents", row=2, col=1)
        fig.update_yaxes(title_text="Average Premium ($)", row=2, col=1, range=[485, 502])
        fig.update_yaxes(title_text="Insurance Premium ($)", row=2, col=2)
        fig.update_xaxes(title_text="Feature Importance Score", row=3, col=2)
        
        return fig
    
    def create_model_performance(self):
        """Create model performance analysis with 6 comprehensive metrics"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '<b>🏆 Top 10 Models - Validation</b>',
                '<b>🏆 Top Models - Test Results</b>',
                '<b>⚖️ Overfitting Check (Best & Worst)</b>',
                '<b>🔬 Model Complexity - Test Results</b>',
                '<b>💯 Prediction Error Range</b>',
                '<b>⭐ Best Model Indicator</b>'
            ),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'indicator'}]],
            vertical_spacing=0.18,
            horizontal_spacing=0.15,
            row_heights=[0.32, 0.32, 0.36]
        )
        
        # 1. Top 10 Models - Validation Results (from actual data)
        top10_val = self.model_results.nlargest(10, 'Val_R2').sort_values('Val_R2')
        
        # Create cleaner model names
        model_icons = {'1': '🥇', '2': '🥈', '3': '🥉'}
        short_names = []
        for idx, name in enumerate(top10_val['Model'], 1):
            icon = model_icons.get(str(11-idx), '')  # Since sorted ascending, top is at the end
            if 'Regression' in name:
                clean_name = name.replace(' Regression', '')
            elif 'Random Forest' in name:
                clean_name = 'Random Forest'
            elif 'Gradient Boosting' in name:
                clean_name = 'Gradient Boost'
            elif 'Extra Trees' in name:
                clean_name = 'Extra Trees'
            elif 'Stacking' in name:
                clean_name = name.replace('Stacking ', 'Stack ')
            else:
                clean_name = name[:15]
            short_names.append(f'{icon} {clean_name}' if icon else clean_name)
        
        # Professional monochrome gradient
        colors = []
        for i in range(len(top10_val)):
            if i >= 7:  # Top 3 models
                colors.append('#2E86AB')  # Primary blue
            else:
                gray_value = int(180 - i * 10)
                colors.append(f'rgb({gray_value}, {gray_value}, {gray_value})')
        
        fig.add_trace(
            go.Bar(
                x=top10_val['Val_R2'],
                y=short_names,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=[f'{x:.4f}' for x in top10_val['Val_R2']],
                textposition='outside',
                textfont=dict(size=10, color='black'),
                name='Validation R²',
                hovertemplate='<b>%{customdata}</b><br>Val R²: %{x:.4f}<extra></extra>',
                customdata=top10_val['Model']
            ),
            row=1, col=1
        )
        
        # 2. Top Models - Test Results (actual test data)
        if hasattr(self, 'test_results') and len(self.test_results) > 0:
            # Sort test results by Test_R2
            test_sorted = self.test_results.sort_values('Test_R2')
            
            # Clean model names
            test_names = []
            for idx, name in enumerate(test_sorted['Model'], 1):
                if idx == len(test_sorted):
                    icon = '🥇'
                elif idx == len(test_sorted) - 1:
                    icon = '🥈'
                elif idx == len(test_sorted) - 2:
                    icon = '🥉'
                else:
                    icon = ''
                
                if 'Stacking' in name:
                    clean_name = name.replace('Stacking ', 'Stack ').replace('(', '').replace(')', '')
                elif 'Voting' in name:
                    clean_name = 'Voting Ens.'
                else:
                    clean_name = name
                test_names.append(f'{icon} {clean_name}' if icon else clean_name)
            
            # Colors for test results
            test_colors = ['#999999', '#5BA0C3', '#2E86AB']  # Gray to blue gradient
            
            fig.add_trace(
                go.Bar(
                    x=test_sorted['Test_R2'],
                    y=test_names,
                    orientation='h',
                    marker=dict(
                        color=test_colors[:len(test_sorted)],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{x:.4f}' for x in test_sorted['Test_R2']],
                    textposition='outside',
                    textfont=dict(size=10, color='black'),
                    name='Test R²',
                    hovertemplate='<b>%{customdata}</b><br>Test R²: %{x:.4f}<br>RMSE: %{meta:.4f}<extra></extra>',
                    customdata=test_sorted['Model'],
                    meta=test_sorted['Test_RMSE']
                ),
                row=1, col=2
            )
        else:
            # Fallback if no test results
            fig.add_annotation(
                text="No test results available",
                xref="x2", yref="y2",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
                row=1, col=2
            )
        
        # 3. Overfitting Analysis - Simplified Bar Chart
        # Calculate overfitting as percentage difference
        overfit_data = self.model_results.copy()
        overfit_data['Overfit_Pct'] = ((overfit_data['Train_R2'] - overfit_data['Val_R2']) / overfit_data['Train_R2'] * 100).round(3)
        
        # Get top 5 most overfit and top 5 least overfit models
        most_overfit = overfit_data.nlargest(5, 'Overfit_Pct')
        least_overfit = overfit_data.nsmallest(5, 'Overfit_Pct')
        combined = pd.concat([least_overfit, most_overfit]).sort_values('Overfit_Pct')
        
        # Simplify model names
        model_names = []
        for name in combined['Model']:
            if 'Regression' in name:
                model_names.append(name.replace(' Regression', '').replace('Polynomial', 'Poly'))
            elif 'Random Forest' in name:
                model_names.append('Random Forest')
            elif 'Gradient Boosting' in name:
                model_names.append('Gradient Boost')
            elif 'Stacking' in name:
                model_names.append(name.replace('Stacking ', 'Stack '))
            else:
                model_names.append(name[:15])
        
        # Professional gradient - blue for good, gray for bad
        colors = []
        for pct in combined['Overfit_Pct']:
            if pct < 0.01:  # Excellent - no overfitting
                colors.append('#2E86AB')  # Primary blue
            elif pct < 0.05:  # Good - minimal overfitting  
                colors.append('#5BA0C3')  # Light blue
            else:  # Overfitting
                colors.append('#999999')  # Gray
        
        fig.add_trace(
            go.Bar(
                x=combined['Overfit_Pct'],
                y=model_names,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=[f'{x:.3f}%' if x >= 0 else f'{x:.3f}%' for x in combined['Overfit_Pct']],
                textposition='outside',
                textfont=dict(size=10),
                name='Overfitting %',
                hovertemplate='<b>%{y}</b><br>Overfitting: %{x:.3f}%<br>Train R²: %{customdata[0]:.4f}<br>Val R²: %{customdata[1]:.4f}<extra></extra>',
                customdata=[[train, val] for train, val in zip(combined['Train_R2'], combined['Val_R2'])]
            ),
            row=2, col=1
        )
        
        # Add reference line at 0
        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=-0.5, y1=len(combined)-0.5,
            line=dict(color="gray", width=2, dash="dash"),
            row=2, col=1
        )
        
        # 4. Model Complexity Analysis - Based on Test Results
        if hasattr(self, 'test_results') and len(self.test_results) > 0:
            # Use actual test results for complexity analysis
            # Assign complexity scores based on model type
            complexity_scores = {
                'Stacking (Linear)': 9,
                'Stacking (Ridge)': 9,
                'Voting Ensemble': 8
            }
            
            test_complexity = self.test_results.copy()
            test_complexity['Complexity'] = test_complexity['Model'].map(complexity_scores)
            
            # Also add some validation models for comparison
            val_models_for_complexity = pd.DataFrame({
                'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting'],
                'Test_R2': [self.model_results[self.model_results['Model'] == 'Linear Regression']['Val_R2'].values[0],
                           self.model_results[self.model_results['Model'] == 'Ridge Regression']['Val_R2'].values[0],
                           self.model_results[self.model_results['Model'] == 'Random Forest']['Val_R2'].values[0],
                           self.model_results[self.model_results['Model'] == 'Gradient Boosting']['Val_R2'].values[0]],
                'Complexity': [1, 2, 7, 8],
                'Type': ['Val', 'Val', 'Val', 'Val']
            })
            test_complexity['Type'] = 'Test'
            
            combined_complexity = pd.concat([test_complexity[['Model', 'Test_R2', 'Complexity', 'Type']], 
                                            val_models_for_complexity])
            
            # Create bubble chart
            colors = ['#2E86AB' if t == 'Test' else '#e0e0e0' for t in combined_complexity['Type']]
            sizes = [50 if t == 'Test' else 30 for t in combined_complexity['Type']]
            
            fig.add_trace(
                go.Scatter(
                    x=combined_complexity['Complexity'],
                    y=combined_complexity['Test_R2'],
                    mode='markers+text',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        line=dict(color='white', width=2)
                    ),
                    text=[m.replace(' Regression', '').replace('Stacking ', 'Stack ').replace('(', '').replace(')', '') 
                          for m in combined_complexity['Model']],
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertemplate='<b>%{text}</b><br>Complexity: %{x}<br>R²: %{y:.4f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 5. Prediction Error Range - More intuitive error visualization
        if hasattr(self, 'test_results') and len(self.test_results) > 0:
            # Show error ranges in a more intuitive way
            models = self.test_results['Model'].str.replace('Stacking ', '').str.replace('(', '').str.replace(')', '')
            
            # Create a combined metric showing RMSE and MAE ranges
            rmse_vals = self.test_results['Test_RMSE'].values
            mae_vals = self.test_results['Test_MAE'].values
            
            # Create grouped bar chart showing both error metrics
            fig.add_trace(
                go.Bar(
                    x=['RMSE', 'RMSE', 'RMSE'],
                    y=rmse_vals,
                    name='RMSE',
                    marker=dict(
                        color=['#2E86AB', '#5BA0C3', '#999999'],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{models.iloc[i]}<br>${v:.2f}' for i, v in enumerate(rmse_vals)],
                    textposition='outside',
                    textfont=dict(size=9),
                    hovertemplate='<b>%{text}</b><br>RMSE: $%{y:.2f}<extra></extra>',
                    showlegend=False,
                    offsetgroup=0
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=['MAE', 'MAE', 'MAE'],
                    y=mae_vals,
                    name='MAE',
                    marker=dict(
                        color=['#2E86AB', '#5BA0C3', '#999999'],
                        pattern=dict(shape="/", size=3, solidity=0.5),
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{models.iloc[i]}<br>${v:.2f}' for i, v in enumerate(mae_vals)],
                    textposition='outside',
                    textfont=dict(size=9),
                    hovertemplate='<b>%{text}</b><br>MAE: $%{y:.2f}<extra></extra>',
                    showlegend=False,
                    offsetgroup=1
                ),
                row=3, col=1
            )
            
            # Add annotations to explain metrics
            fig.add_annotation(
                x=0.5, y=0.45,
                text="Lower is Better →",
                showarrow=False,
                font=dict(size=11, color="#666"),
                xref="x5", yref="y5",
                row=3, col=1
            )
        
        # 6. Best Model Indicator
        best_r2 = self.test_results['Test_R2'].max() if hasattr(self, 'test_results') and len(self.test_results) > 0 else 0.9978
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=best_r2,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Best Test R² Score</b>", 'font': {'size': 16}},
                delta={'reference': 0.95, 'increasing': {'color': self.colors['success']}},
                gauge={
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': '#2E86AB'},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.8], 'color': '#e0e0e0'},
                        {'range': [0.8, 0.95], 'color': '#b0b0b0'},
                        {'range': [0.95, 1], 'color': '#2E86AB'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.99
                    }
                }
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=950,
            showlegend=False,
            template=self.template,
            font=dict(family=self.font_family, size=11),
            margin=dict(t=50, b=80, l=80, r=80),
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Validation R² Score", row=1, col=1, range=[0.985, 1.002], tickformat='.4f')
        fig.update_yaxes(title_text="", row=1, col=1, tickfont=dict(size=10))
        fig.update_xaxes(title_text="Model Complexity", row=1, col=2, range=[0, 10])
        fig.update_yaxes(title_text="Validation R² Score", row=1, col=2, range=[0.990, 0.999])
        fig.update_xaxes(title_text="Overfitting Percentage", row=2, col=1, range=[-0.1, 0.5])
        fig.update_yaxes(title_text="", row=2, col=1, tickfont=dict(size=10))
        fig.update_xaxes(title_text="Test RMSE", row=2, col=2, range=[0.25, 0.45])
        fig.update_yaxes(title_text="Test MAE", row=2, col=2, range=[0.18, 0.32])
        
        return fig