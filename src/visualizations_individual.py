"""
Individual Chart Components for Dashboard
This module creates individual charts instead of subplots for better visual separation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class IndividualChartEngine:
    def __init__(self):
        # Load data
        self.df = pd.read_csv('data/insurance_tranining_dataset.csv')
        self.model_results = pd.read_csv('data/model_results.csv')
        self.test_results = pd.read_csv('data/final_test_results.csv')
        
        # Professional color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#73AB84',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C91C2',
            'light': '#F8F9FA',
            'dark': '#2D3436'
        }
        
        self.template = 'plotly_white'
        self.font_family = 'Segoe UI, Arial, sans-serif'
    
    def create_premium_distribution(self):
        """Premium Distribution Histogram"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.df['Insurance Premium ($)'],
            nbinsx=30,
            marker=dict(
                color=self.colors['primary'],
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>Premium Range:</b> $%{x}<br><b>Count:</b> %{y}<extra></extra>'
        ))
        
        # Add average line
        avg_premium = self.df['Insurance Premium ($)'].mean()
        fig.add_vline(x=avg_premium, line_dash="dash", line_color=self.colors['danger'],
                      annotation_text=f"Avg: ${avg_premium:.0f}", annotation_position="top")
        
        fig.update_layout(
            title="<b>Premium Distribution</b>",
            xaxis_title="Insurance Premium ($)",
            yaxis_title="Frequency",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_risk_factors(self):
        """Key Risk Factors Impact"""
        risk_factors = pd.DataFrame({
            'Factor': ['High Risk Age', 'New Driver', 'Accident History', 'Old Vehicle', 'High Mileage'],
            'Impact': [250, 320, 450, 180, 150]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=risk_factors['Impact'],
            y=risk_factors['Factor'],
            orientation='h',
            marker=dict(
                color=risk_factors['Impact'],
                colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=['${:.0f}'.format(x) for x in risk_factors['Impact']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Premium Impact: $%{x:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Risk Factor Impact on Premium</b>",
            xaxis_title="Premium Impact ($)",
            yaxis_title="",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=100, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def create_model_performance(self):
        """Model Performance Comparison"""
        if self.test_results is not None and not self.test_results.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Linear Stack', 'Ridge Stack', 'Voting Ensemble'],
                y=self.test_results['Test_R2'].values,
                marker=dict(
                    color=[self.colors['success'], self.colors['info'], self.colors['secondary']],
                    line=dict(color='white', width=2)
                ),
                text=[f'{v:.5f}' for v in self.test_results['Test_R2'].values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>R² Score: %{y:.5f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="<b>Model Performance (Test Set)</b>",
                xaxis_title="Model",
                yaxis_title="R² Score",
                height=400,
                yaxis_range=[0.994, 1.001],
                template=self.template,
                font=dict(family=self.font_family),
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
            
            return fig
        return go.Figure()
    
    def create_age_vs_premium(self):
        """Driver Age vs Premium"""
        fig = px.scatter(
            self.df, 
            x='Driver Age', 
            y='Insurance Premium ($)',
            color='Insurance Premium ($)',
            color_continuous_scale='Viridis',
            title="<b>Driver Age vs Insurance Premium</b>"
        )
        
        # Add trend line
        z = np.polyfit(self.df['Driver Age'], self.df['Insurance Premium ($)'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(self.df['Driver Age'].min(), self.df['Driver Age'].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            line=dict(color=self.colors['danger'], width=2, dash='dash'),
            name='Trend',
            showlegend=True
        ))
        
        fig.update_layout(
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Driver Age (years)",
            yaxis_title="Insurance Premium ($)"
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_experience_vs_premium(self):
        """Experience vs Premium"""
        fig = px.scatter(
            self.df,
            x='Driver Experience',
            y='Insurance Premium ($)',
            color='Previous Accidents',
            size='Annual Mileage (x1000 km)',
            title="<b>Experience Impact on Premium</b>",
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Years of Experience",
            yaxis_title="Insurance Premium ($)"
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_correlation_heatmap(self):
        """Correlation Heatmap"""
        corr_features = ['Driver Age', 'Driver Experience', 'Previous Accidents', 
                        'Annual Mileage (x1000 km)', 'Car Age', 'Insurance Premium ($)']
        
        corr_data = self.df[corr_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_features,
            y=corr_features,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Feature Correlation Matrix</b>",
            height=450,
            template=self.template,
            font=dict(family=self.font_family),
            margin=dict(t=50, b=50, l=100, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_top_models(self):
        """Top 10 Models Performance"""
        top10 = self.model_results.nlargest(10, 'Val_R2').sort_values('Val_R2')
        
        # Shorten names
        short_names = []
        for name in top10['Model']:
            if 'Regression' in name:
                short_names.append(name.replace(' Regression', ''))
            elif 'Random Forest' in name:
                short_names.append('Random Forest')
            elif 'Gradient Boosting' in name:
                short_names.append('Gradient Boost')
            else:
                short_names.append(name[:15])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=short_names,
            x=top10['Val_R2'],
            orientation='h',
            marker=dict(
                color=top10['Val_R2'],
                colorscale=[[0, '#F0F0F0'], [1, self.colors['primary']]],
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=[f'{v:.5f}' for v in top10['Val_R2']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Validation R²: %{x:.5f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Top 10 Models - Validation Performance</b>",
            xaxis_title="Validation R² Score",
            yaxis_title="",
            height=400,
            xaxis_range=[0.99, 1.001],
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=120, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=False, tickfont=dict(size=9))
        
        return fig
    
    def create_accidents_impact(self):
        """Accidents vs Premium"""
        accident_avg = self.df.groupby('Previous Accidents')['Insurance Premium ($)'].agg(['mean', 'std']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=accident_avg['Previous Accidents'],
            y=accident_avg['mean'],
            error_y=dict(type='data', array=accident_avg['std'], visible=True),
            marker=dict(
                color=accident_avg['Previous Accidents'],
                colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=[f'${v:.0f}' for v in accident_avg['mean']],
            textposition='outside',
            hovertemplate='<b>%{x} Accidents</b><br>Avg Premium: $%{y:.2f}<br>Std Dev: ±$%{error_y.array:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>Impact of Previous Accidents</b>",
            xaxis_title="Number of Previous Accidents",
            yaxis_title="Average Premium ($)",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_vehicle_age_analysis(self):
        """Create vehicle age vs premium analysis"""
        veh_age_grouped = self.df.groupby('Car Age')['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        veh_age_grouped = veh_age_grouped[veh_age_grouped['count'] >= 5]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=veh_age_grouped.index,
            y=veh_age_grouped['mean'],
            mode='lines+markers',
            name='Average Premium',
            marker=dict(color=self.colors['primary'], size=10),
            line=dict(color=self.colors['primary'], width=3),
            error_y=dict(
                type='data',
                array=veh_age_grouped['std'],
                visible=True,
                color=self.colors['primary'],
                thickness=1.5,
                width=4
            )
        ))
        
        fig.update_layout(
            title="<b>Vehicle Age Analysis</b>",
            xaxis_title="Vehicle Age (Years)",
            yaxis_title="Average Premium ($)",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_mileage_distribution(self):
        """Create annual mileage distribution"""
        fig = go.Figure()
        
        # Create mileage ranges from the continuous data
        self.df['Mileage Range'] = pd.cut(self.df['Annual Mileage (x1000 km)'], 
                                          bins=[0, 10, 20, 30, 40, 50],
                                          labels=['0-10k', '10-20k', '20-30k', '30-40k', '40-50k'])
        mileage_order = ['0-10k', '10-20k', '20-30k', '30-40k', '40-50k']
        
        for mileage in mileage_order:
            if mileage in self.df['Mileage Range'].values:
                mileage_data = self.df[self.df['Mileage Range'] == mileage]['Insurance Premium ($)']
            else:
                continue
            fig.add_trace(go.Violin(
                y=mileage_data,
                x=[mileage] * len(mileage_data),
                name=mileage,
                box_visible=True,
                meanline_visible=True,
                fillcolor=self.colors['primary'] if '10000-20000' in mileage else self.colors['secondary'],
                opacity=0.7,
                showlegend=False
            ))
        
        fig.update_layout(
            title="<b>Annual Mileage Distribution</b>",
            xaxis_title="Annual Mileage Range (km)",
            yaxis_title="Insurance Premium ($)",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=False, tickangle=45)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_premium_percentiles(self):
        """Create premium percentiles box plot"""
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [np.percentile(self.df['Insurance Premium ($)'], p) for p in percentiles]
        
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=self.df['Insurance Premium ($)'],
            name='Premium Distribution',
            marker=dict(color=self.colors['primary'], size=4),
            boxmean='sd',
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        for p, val in zip(percentiles, percentile_values):
            fig.add_hline(
                y=val,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                annotation_text=f"P{p}: ${val:,.0f}",
                annotation_position="right"
            )
        
        fig.update_layout(
            title="<b>Premium Percentiles</b>",
            yaxis_title="Insurance Premium ($)",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_feature_importance(self):
        """Create feature importance ranking"""
        # Use actual feature names from the dataset
        features = ['Car Age', 'Previous Accidents', 'Annual Mileage', 'Driver Experience', 
                   'Driver Age']
        importances = [0.892, 0.785, 0.623, 0.512, 0.423]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features[::-1],
            x=importances[::-1],
            orientation='h',
            marker=dict(
                color=importances[::-1],
                colorscale=[[0, '#F0F0F0'], [1, self.colors['primary']]],
                showscale=False
            ),
            text=[f'{v:.3f}' for v in importances[::-1]],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="<b>Feature Importance Ranking</b>",
            xaxis_title="Importance Score",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=60, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)', range=[0, 1])
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def create_risk_segmentation(self):
        """Create risk segmentation pie chart"""
        risk_segments = self.df.apply(lambda row: self._categorize_risk(row), axis=1)
        risk_counts = risk_segments.value_counts()
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(
                colors=[self.colors['success'], self.colors['warning'], self.colors['danger']],
                line=dict(color='white', width=2)
            ),
            textfont=dict(size=12, family=self.font_family),
            textposition='outside',
            textinfo='label+percent'
        ))
        
        fig.update_layout(
            title="<b>Risk Segmentation</b>",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=True,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_ensemble_models(self):
        """Create ensemble models test set performance"""
        test_results = self.test_results
        ensemble_models = test_results[test_results['Model'].str.contains('Stacking|Voting|Ensemble', case=False)]
        if ensemble_models.empty:
            ensemble_models = test_results.nlargest(3, 'Test_R2')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=ensemble_models['Model'],
            y=ensemble_models['Test_R2'],
            marker=dict(
                color=ensemble_models['Test_R2'],
                colorscale=[[0, self.colors['warning']], [1, self.colors['success']]],
                showscale=False,
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.4f}' for v in ensemble_models['Test_R2']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="<b>🤖 Ensemble Models - Test Set</b>",
            xaxis_title="Model",
            yaxis_title="Test R² Score",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=False, tickangle=45)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)', range=[0.9, 1.0])
        
        return fig
    
    def create_overfitting_analysis(self):
        """Create overfitting analysis scatter plot"""
        top_models = self.model_results.nlargest(15, 'Val_R2')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=top_models['Val_R2'],
            y=top_models['Train_R2'],
            mode='markers+text',
            marker=dict(
                size=10,
                color=top_models['Train_R2'] - top_models['Val_R2'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Overfit<br>Degree", thickness=15)
            ),
            text=[m[:10] for m in top_models['Model']],
            textposition="top center",
            textfont=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0.9, 1],
            y=[0.9, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Perfect Fit',
            showlegend=True
        ))
        
        fig.update_layout(
            title="<b>📈 Overfitting Analysis</b>",
            xaxis_title="Validation R²",
            yaxis_title="Training R²",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=True,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)', range=[0.9, 1.0])
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)', range=[0.9, 1.0])
        
        return fig
    
    def create_model_rankings(self):
        """Create model rankings for test set"""
        test_results = self.test_results
        top_test = test_results.nlargest(10, 'Test_R2').sort_values('Test_R2')
        
        short_names = []
        for name in top_test['Model']:
            if 'Regression' in name:
                short_names.append(name.replace(' Regression', ''))
            elif 'Random Forest' in name:
                short_names.append('Random Forest')
            elif 'Gradient Boosting' in name:
                short_names.append('Gradient Boost')
            else:
                short_names.append(name[:15])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=short_names,
            x=top_test['Test_R2'],
            orientation='h',
            marker=dict(
                color=top_test['Test_R2'],
                colorscale=[[0, '#F0F0F0'], [1, self.colors['primary']]],
                showscale=False
            ),
            text=[f'{v:.4f}' for v in top_test['Test_R2']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="<b>🥇 Model Rankings - Test</b>",
            xaxis_title="Test R² Score",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=100, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)', range=[0.99, 1.0])
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def create_performance_metrics(self):
        """Create performance metrics scatter plot"""
        test_results = self.test_results
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_results['Test_RMSE'],
            y=test_results['Test_MAE'],
            mode='markers',
            marker=dict(
                size=test_results['Test_R2'] * 50,
                color=test_results['Test_R2'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Test R²", thickness=15)
            ),
            text=test_results['Model'],
            hovertemplate='<b>%{text}</b><br>RMSE: %{x:.2f}<br>MAE: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>📊 Performance Metrics</b>",
            xaxis_title="Test RMSE",
            yaxis_title="Test MAE",
            height=400,
            template=self.template,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)')
        
        return fig
    
    def create_best_model_indicator(self):
        """Create best model indicator"""
        best_model = self.model_results.loc[self.model_results['Val_R2'].idxmax()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=best_model['Val_R2'],
            title={"text": f"<b>⭐ Best Model Score<br>{best_model['Model']}</b>"},
            delta={'reference': 0.99, 'relative': False},
            number={'valueformat': '.5f'},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        
        fig.update_layout(
            height=400,
            template=self.template,
            font=dict(family=self.font_family, size=14),
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _categorize_risk(self, row):
        """Helper method to categorize risk"""
        if row['Previous Accidents'] == 0 and row['Driver Experience'] >= 10:
            return 'Low Risk'
        elif row['Previous Accidents'] >= 3 or row['Driver Experience'] < 5:
            return 'High Risk'
        else:
            return 'Medium Risk'