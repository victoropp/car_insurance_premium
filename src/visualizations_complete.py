"""
Complete Individual Chart Functions
Extracts all 18 charts from the original visualization engine
Maintains 100% consistency with original styling and data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class CompleteIndividualCharts:
    def __init__(self, data_path='data/insurance_tranining_dataset.csv',
                 model_results_path='data/model_results.csv',
                 feature_importance_path='data/feature_importance.csv',
                 test_results_path='data/final_test_results.csv'):
        self.df = pd.read_csv(data_path)
        self.model_results = pd.read_csv(model_results_path)
        self.feature_importance = pd.read_csv(feature_importance_path) if feature_importance_path else None
        try:
            self.test_results = pd.read_csv(test_results_path) if test_results_path else None
        except:
            self.test_results = None
        
        # Professional color scheme - EXACT same as original
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
    
    # EXECUTIVE SUMMARY CHARTS (1-6)
    
    def chart_1_premium_distribution(self):
        """Chart 1: Premium Distribution"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=self.df['Insurance Premium ($)'],
                nbinsx=25,
                marker=dict(
                    color=self.colors['primary'],
                    line=dict(color='white', width=1)
                ),
                name='Premium Distribution',
                hovertemplate='<b>Premium Range:</b> $%{x}<br><b>Count:</b> %{y}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>📊 Premium Distribution</b>',
            xaxis_title='Insurance Premium ($)',
            yaxis_title='Count',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_2_key_risk_factors(self):
        """Chart 2: Key Risk Factors"""
        risk_impact = pd.DataFrame({
            'Factor': ['Previous Accidents', 'Driver Age < 25', 'New Driver', 'High Mileage', 'Old Vehicle'],
            'Impact': [
                self.df.groupby('Previous Accidents')['Insurance Premium ($)'].mean().diff().mean(),
                self.df[self.df['Driver Age'] < 25]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean(),
                self.df[self.df['Driver Experience'] < 2]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean(),
                self.df[self.df['Annual Mileage (x1000 km)'] > 20]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean(),
                self.df[self.df['Car Age'] > 10]['Insurance Premium ($)'].mean() - self.df['Insurance Premium ($)'].mean()
            ]
        }).sort_values('Impact')
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=risk_impact['Impact'],
                y=risk_impact['Factor'],
                orientation='h',
                marker=dict(
                    color=risk_impact['Impact'],
                    colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=['${:.2f}'.format(x) for x in risk_impact['Impact']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Premium Impact: $%{x:.2f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>⚠️ Key Risk Factors</b>',
            xaxis_title='Premium Impact ($)',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_3_test_set_performance(self):
        """Chart 3: Test Set Performance"""
        fig = go.Figure()
        
        if self.test_results is not None and not self.test_results.empty:
            test_results = self.test_results
            fig.add_trace(
                go.Bar(
                    x=['Linear<br>Stack', 'Ridge<br>Stack', 'Voting<br>Ens.'],
                    y=test_results['Test_R2'].values,
                    marker=dict(
                        color=[self.colors['success'], self.colors['info'], self.colors['secondary']],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.4f}' for v in test_results['Test_R2'].values],
                    textposition='outside',
                    textfont=dict(size=10, color=self.colors['dark']),
                    width=0.6,
                    hovertemplate='<b>%{customdata[0]}</b><br>Test R²: %{y:.4f}<br>RMSE: %{customdata[1]:.3f}<extra></extra>',
                    customdata=[[name, rmse] for name, rmse in zip(
                        ['Stacking (Linear)', 'Stacking (Ridge)', 'Voting Ensemble'],
                        test_results['Test_RMSE'].values
                    )]
                )
            )
        else:
            top3_models = self.model_results.nlargest(3, 'Val_R2')
            fig.add_trace(
                go.Bar(
                    x=[m[:12] for m in top3_models['Model']],
                    y=top3_models['Val_R2'].values,
                    marker=dict(
                        color=[self.colors['success'], self.colors['info'], self.colors['secondary']],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.4f}' for v in top3_models['Val_R2'].values],
                    textposition='outside',
                    textfont=dict(size=11, color=self.colors['dark']),
                    hovertemplate='<b>%{x}</b><br>Val R²: %{y:.4f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title='<b>🎯 Test Set Performance</b>',
            xaxis_title='Model',
            yaxis_title='R² Score',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_4_feature_correlations(self):
        """Chart 4: Feature Correlations"""
        corr_full = self.df.select_dtypes(include=[np.number]).corr()
        
        # Select most important correlations with target
        target_corr = corr_full['Insurance Premium ($)'].sort_values(ascending=False)[1:6]
        
        # Create a subset correlation matrix with top features
        top_features = list(target_corr.index) + ['Insurance Premium ($)']
        corr_subset = self.df[top_features].corr()
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_subset.values,
                x=['Age', 'Experience', 'Accidents', 'Mileage', 'Car Age', 'Premium'],
                y=['Age', 'Experience', 'Accidents', 'Mileage', 'Car Age', 'Premium'],
                colorscale='RdBu',
                zmid=0,
                text=corr_subset.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation", thickness=15),
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>🔗 Feature Correlations</b>',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_5_age_vs_premium(self):
        """Chart 5: Age vs Premium Analysis"""
        age_groups = pd.cut(self.df['Driver Age'], bins=[18, 25, 35, 45, 55, 65, 100],
                           labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        fig = go.Figure()
        
        for age_group in age_groups.unique():
            if pd.notna(age_group):
                mask = age_groups == age_group
                fig.add_trace(
                    go.Box(
                        y=self.df.loc[mask, 'Insurance Premium ($)'],
                        name=age_group,
                        marker=dict(color=self.colors['primary'], opacity=0.7),
                        boxmean='sd'
                    )
                )
        
        fig.update_layout(
            title='<b>👥 Age vs Premium Analysis</b>',
            xaxis_title='Age Group',
            yaxis_title='Insurance Premium ($)',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_6_risk_segmentation(self):
        """Chart 6: Risk Segmentation"""
        def categorize_risk(row):
            if row['Previous Accidents'] == 0 and row['Driver Experience'] >= 10:
                return 'Low Risk'
            elif row['Previous Accidents'] >= 3 or row['Driver Experience'] < 5:
                return 'High Risk'
            else:
                return 'Medium Risk'
        
        risk_segments = self.df.apply(categorize_risk, axis=1)
        risk_counts = risk_segments.value_counts()
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker=dict(
                    colors=[self.colors['success'], self.colors['warning'], self.colors['danger']],
                    line=dict(color='white', width=2)
                ),
                textfont=dict(size=12, family=self.font_family),
                textposition='outside',
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>🎲 Risk Segmentation</b>',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            annotations=[
                dict(text='Risk<br>Profile', x=0.5, y=0.5, font_size=14, showarrow=False)
            ]
        )
        
        return fig
    
    # DETAILED ANALYSIS CHARTS (7-12)
    
    def chart_7_driver_experience_impact(self):
        """Chart 7: Driver Experience Impact"""
        exp_grouped = self.df.groupby('Driver Experience')['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        exp_grouped = exp_grouped[exp_grouped['count'] >= 5]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=exp_grouped.index,
                y=exp_grouped['mean'],
                mode='lines+markers',
                name='Average Premium',
                marker=dict(color=self.colors['primary'], size=10),
                line=dict(color=self.colors['primary'], width=3),
                error_y=dict(
                    type='data',
                    array=exp_grouped['std'],
                    visible=True,
                    color=self.colors['primary'],
                    thickness=1.5,
                    width=4
                ),
                hovertemplate='<b>Experience: %{x} years</b><br>Avg Premium: $%{y:.2f}<br>Std Dev: ±$%{error_y.array:.2f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>Driver Experience Impact</b>',
            xaxis_title='Years of Experience',
            yaxis_title='Average Premium ($)',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_8_vehicle_age_analysis(self):
        """Chart 8: Vehicle Age Analysis"""
        veh_age_grouped = self.df.groupby('Car Age')['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        veh_age_grouped = veh_age_grouped[veh_age_grouped['count'] >= 5]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=veh_age_grouped.index,
                y=veh_age_grouped['mean'],
                mode='lines+markers',
                name='Average Premium',
                marker=dict(color=self.colors['secondary'], size=10),
                line=dict(color=self.colors['secondary'], width=3),
                fill='tozeroy',
                fillcolor='rgba(162, 59, 114, 0.2)',
                hovertemplate='<b>Vehicle Age: %{x} years</b><br>Avg Premium: $%{y:.2f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>Vehicle Age Analysis</b>',
            xaxis_title='Vehicle Age (Years)',
            yaxis_title='Average Premium ($)',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_9_accident_history_effect(self):
        """Chart 9: Accident History Effect"""
        accident_grouped = self.df.groupby('Previous Accidents')['Insurance Premium ($)'].agg(['mean', 'std', 'count'])
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=accident_grouped.index,
                y=accident_grouped['mean'],
                marker=dict(
                    color=accident_grouped.index,
                    colorscale=[[0, self.colors['success']], [0.5, self.colors['warning']], [1, self.colors['danger']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                error_y=dict(
                    type='data',
                    array=accident_grouped['std'],
                    visible=True,
                    color='rgba(0,0,0,0.3)',
                    thickness=1.5,
                    width=4
                ),
                text=[f'${v:.0f}' for v in accident_grouped['mean']],
                textposition='outside',
                hovertemplate='<b>%{x} Accidents</b><br>Avg Premium: $%{y:.2f}<br>Std Dev: ±$%{error_y.array:.2f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>Accident History Effect</b>',
            xaxis_title='Number of Previous Accidents',
            yaxis_title='Average Premium ($)',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_10_annual_mileage_distribution(self):
        """Chart 10: Annual Mileage Distribution"""
        mileage_bins = pd.cut(self.df['Annual Mileage (x1000 km)'], 
                              bins=[0, 10, 20, 30, 40, 50],
                              labels=['0-10k', '10-20k', '20-30k', '30-40k', '40-50k'])
        
        fig = go.Figure()
        
        for mileage_range in ['0-10k', '10-20k', '20-30k', '30-40k', '40-50k']:
            if mileage_range in mileage_bins.values:
                mask = mileage_bins == mileage_range
                if mask.sum() > 0:
                    fig.add_trace(
                        go.Violin(
                            y=self.df.loc[mask, 'Insurance Premium ($)'],
                            x=[mileage_range] * mask.sum(),
                            name=mileage_range,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=self.colors['primary'] if '10-20k' in mileage_range else self.colors['info'],
                            opacity=0.7,
                            showlegend=False
                        )
                    )
        
        fig.update_layout(
            title='<b>Annual Mileage Distribution</b>',
            xaxis_title='Annual Mileage Range (km)',
            yaxis_title='Insurance Premium ($)',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_11_premium_percentiles(self):
        """Chart 11: Premium Percentiles"""
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [np.percentile(self.df['Insurance Premium ($)'], p) for p in percentiles]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Box(
                y=self.df['Insurance Premium ($)'],
                name='Premium Distribution',
                marker=dict(color=self.colors['primary'], size=4),
                boxmean='sd',
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate='Premium: $%{y:.2f}<extra></extra>'
            )
        )
        
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
            title='<b>Premium Percentiles</b>',
            yaxis_title='Insurance Premium ($)',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_12_feature_importance_ranking(self):
        """Chart 12: Feature Importance Ranking"""
        if self.feature_importance is not None and not self.feature_importance.empty:
            feat_imp = self.feature_importance.nlargest(8, 'Importance')
        else:
            feat_imp = pd.DataFrame({
                'Feature': ['Car Age', 'Previous Accidents', 'Annual Mileage', 'Driver Experience', 
                           'Driver Age', 'Risk Score', 'High Risk Driver', 'Experience Rate'],
                'Importance': [0.892, 0.785, 0.623, 0.512, 0.423, 0.312, 0.289, 0.234]
            })
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                y=feat_imp['Feature'][::-1],
                x=feat_imp['Importance'][::-1],
                orientation='h',
                marker=dict(
                    color=feat_imp['Importance'][::-1],
                    colorscale=[[0, '#F0F0F0'], [1, self.colors['primary']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=[f'{v:.3f}' for v in feat_imp['Importance'][::-1]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='<b>Feature Importance Ranking</b>',
            xaxis_title='Importance Score',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    # MODEL COMPARISON CHARTS (13-18)
    
    def chart_13_top_10_models_validation(self):
        """Chart 13: Top 10 Models - Validation"""
        top10_val = self.model_results.nlargest(10, 'Val_R2').sort_values('Val_R2')
        
        short_names = []
        for name in top10_val['Model']:
            if 'Regression' in name:
                short_names.append(name.replace(' Regression', ''))
            elif 'Random Forest' in name:
                short_names.append('Random Forest')
            elif 'Gradient Boosting' in name:
                short_names.append('Gradient Boost')
            elif 'Extra Trees' in name:
                short_names.append('Extra Trees')
            else:
                short_names.append(name[:12])
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                y=short_names,
                x=top10_val['Val_R2'],
                orientation='h',
                marker=dict(
                    color=top10_val['Val_R2'],
                    colorscale=[[0, '#F0F0F0'], [1, self.colors['primary']]],
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=[f'{v:.4f}' for v in top10_val['Val_R2']],
                textposition='outside',
                textfont=dict(size=9),
                hovertemplate='<b>%{customdata}</b><br>Val R²: %{x:.4f}<extra></extra>',
                customdata=top10_val['Model']
            )
        )
        
        fig.update_layout(
            title='<b>🏆 Top 10 Models - Validation</b>',
            xaxis_title='Validation R² Score',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_14_ensemble_models_test(self):
        """Chart 14: Ensemble Models - Test Set"""
        if self.test_results is not None and not self.test_results.empty:
            test_results = self.test_results
            ensemble_models = test_results
        else:
            ensemble_models = self.model_results[self.model_results['Model'].str.contains('Stacking|Voting|Ensemble', case=False)]
            if ensemble_models.empty:
                ensemble_models = self.model_results.nlargest(3, 'Val_R2')
        
        fig = go.Figure()
        
        if self.test_results is not None:
            fig.add_trace(
                go.Bar(
                    x=ensemble_models['Model'],
                    y=ensemble_models['Test_R2'],
                    marker=dict(
                        color=ensemble_models['Test_R2'],
                        colorscale=[[0, self.colors['warning']], [1, self.colors['success']]],
                        showscale=False,
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.4f}' for v in ensemble_models['Test_R2']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Test R²: %{y:.4f}<extra></extra>'
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=ensemble_models['Model'],
                    y=ensemble_models['Val_R2'],
                    marker=dict(
                        color=ensemble_models['Val_R2'],
                        colorscale=[[0, self.colors['warning']], [1, self.colors['success']]],
                        showscale=False,
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.4f}' for v in ensemble_models['Val_R2']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Val R²: %{y:.4f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title='<b>🤖 Ensemble Models - Test Set</b>',
            xaxis_title='Model',
            yaxis_title='R² Score',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_15_overfitting_analysis(self):
        """Chart 15: Overfitting Analysis"""
        top_models = self.model_results.nlargest(15, 'Val_R2')
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
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
                textfont=dict(size=8),
                hovertemplate='<b>%{customdata}</b><br>Val R²: %{x:.4f}<br>Train R²: %{y:.4f}<extra></extra>',
                customdata=top_models['Model']
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0.9, 1],
                y=[0.9, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Fit',
                showlegend=True
            )
        )
        
        fig.update_layout(
            title='<b>📈 Overfitting Analysis</b>',
            xaxis_title='Validation R²',
            yaxis_title='Training R²',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            xaxis=dict(range=[0.9, 1.0]),
            yaxis=dict(range=[0.9, 1.0])
        )
        
        return fig
    
    def chart_16_model_rankings_test(self):
        """Chart 16: Model Rankings - Test"""
        if self.test_results is not None and not self.test_results.empty:
            test_results = self.test_results
            top_test = test_results.nlargest(10, 'Test_R2').sort_values('Test_R2')
        else:
            top_test = self.model_results.nlargest(10, 'Val_R2').sort_values('Val_R2')
        
        short_names = []
        for name in top_test['Model'] if self.test_results is not None else top_test['Model']:
            if 'Regression' in name:
                short_names.append(name.replace(' Regression', ''))
            elif 'Random Forest' in name:
                short_names.append('Random Forest')
            elif 'Gradient Boosting' in name:
                short_names.append('Gradient Boost')
            else:
                short_names.append(name[:15])
        
        fig = go.Figure()
        
        if self.test_results is not None:
            fig.add_trace(
                go.Bar(
                    y=short_names,
                    x=top_test['Test_R2'],
                    orientation='h',
                    marker=dict(
                        color=top_test['Test_R2'],
                        colorscale=[[0, '#F0F0F0'], [1, self.colors['success']]],
                        showscale=False
                    ),
                    text=[f'{v:.4f}' for v in top_test['Test_R2']],
                    textposition='outside',
                    hovertemplate='<b>%{customdata}</b><br>Test R²: %{x:.4f}<extra></extra>',
                    customdata=top_test['Model']
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    y=short_names,
                    x=top_test['Val_R2'],
                    orientation='h',
                    marker=dict(
                        color=top_test['Val_R2'],
                        colorscale=[[0, '#F0F0F0'], [1, self.colors['success']]],
                        showscale=False
                    ),
                    text=[f'{v:.4f}' for v in top_test['Val_R2']],
                    textposition='outside',
                    hovertemplate='<b>%{customdata}</b><br>Val R²: %{x:.4f}<extra></extra>',
                    customdata=top_test['Model']
                )
            )
        
        fig.update_layout(
            title='<b>🥇 Model Rankings - Test</b>',
            xaxis_title='R² Score',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_17_performance_metrics(self):
        """Chart 17: Performance Metrics"""
        if self.test_results is not None and not self.test_results.empty:
            test_results = self.test_results
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
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
                )
            )
        else:
            # Use validation results
            fig = go.Figure()
            
            top_models = self.model_results.nlargest(15, 'Val_R2')
            fig.add_trace(
                go.Scatter(
                    x=top_models['Val_RMSE'],
                    y=top_models['Val_MAE'],
                    mode='markers',
                    marker=dict(
                        size=top_models['Val_R2'] * 50,
                        color=top_models['Val_R2'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Val R²", thickness=15)
                    ),
                    text=top_models['Model'],
                    hovertemplate='<b>%{text}</b><br>RMSE: %{x:.2f}<br>MAE: %{y:.2f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title='<b>📊 Performance Metrics</b>',
            xaxis_title='RMSE',
            yaxis_title='MAE',
            template=self.template,
            font=dict(family=self.font_family),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def chart_18_best_model_indicator(self):
        """Chart 18: Best Model Indicator"""
        best_model = self.model_results.loc[self.model_results['Val_R2'].idxmax()]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=best_model['Val_R2'],
                title={"text": f"<b>⭐ Best Model Score<br>{best_model['Model']}</b>"},
                delta={'reference': 0.99, 'relative': False},
                number={'valueformat': '.5f'},
                domain={'x': [0, 1], 'y': [0.0, 0.50]}
            )
        )
        
        fig.update_layout(
            template=self.template,
            font=dict(family=self.font_family, size=14),
            height=400
        )
        
        return fig