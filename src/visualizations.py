import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class InsuranceVisualizationEngine:
    def __init__(self, data_path='insurance_tranining_dataset.csv', 
                 model_results_path='model_results.csv',
                 feature_importance_path='feature_importance.csv'):
        self.df = pd.read_csv(data_path)
        self.model_results = pd.read_csv(model_results_path)
        self.feature_importance = pd.read_csv(feature_importance_path) if feature_importance_path else None
        self.color_palette = px.colors.qualitative.Set3
        self.template = 'plotly_white'
        
    def create_data_overview(self):
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Dataset Distribution', 'Correlation Matrix', 'Premium Distribution',
                          'Feature Statistics', 'Data Quality', 'Feature Relationships'),
            specs=[[{'type': 'table'}, {'type': 'heatmap'}, {'type': 'histogram'}],
                   [{'type': 'bar'}, {'type': 'indicator'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        summary_data = []
        for col in self.df.columns:
            summary_data.append([
                col,
                f"{self.df[col].dtype}",
                f"{self.df[col].mean():.2f}" if self.df[col].dtype in ['int64', 'float64'] else 'N/A',
                f"{self.df[col].std():.2f}" if self.df[col].dtype in ['int64', 'float64'] else 'N/A'
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Feature', 'Type', 'Mean', 'Std Dev'],
                           fill_color='lightgray',
                           align='left'),
                cells=dict(values=list(zip(*summary_data)),
                          fill_color='white',
                          align='left')
            ),
            row=1, col=1
        )
        
        corr_matrix = self.df.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 8},
                showscale=True
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=self.df['Insurance Premium ($)'],
                nbinsx=30,
                marker_color='lightblue',
                name='Premium Distribution'
            ),
            row=1, col=3
        )
        
        feature_stats = self.df.describe().loc['mean'].sort_values()
        fig.add_trace(
            go.Bar(
                x=feature_stats.index,
                y=feature_stats.values,
                marker_color='coral',
                name='Mean Values'
            ),
            row=2, col=1
        )
        
        completeness = (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=completeness,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Completeness %"},
                delta={'reference': 100},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Driver Age'],
                y=self.df['Insurance Premium ($)'],
                mode='markers',
                marker=dict(
                    size=self.df['Annual Mileage (x1000 km)'],
                    color=self.df['Previous Accidents'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Accidents")
                ),
                text=[f"Age: {age}<br>Premium: ${premium:.2f}<br>Mileage: {mil}k km<br>Accidents: {acc}" 
                      for age, premium, mil, acc in zip(
                          self.df['Driver Age'], 
                          self.df['Insurance Premium ($)'],
                          self.df['Annual Mileage (x1000 km)'],
                          self.df['Previous Accidents'])],
                hovertemplate='%{text}<extra></extra>',
                name='Driver Profile'
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Comprehensive Data Overview Dashboard",
            title_font_size=20,
            template=self.template
        )
        
        return fig
    
    def create_feature_analysis(self):
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[f'{col} Distribution' for col in self.df.columns[:-1]] + 
                          ['Premium vs Features', 'Feature Importance', 'Risk Segments'],
            specs=[[{'type': 'violin'}, {'type': 'violin'}, {'type': 'violin'}],
                   [{'type': 'violin'}, {'type': 'violin'}, {'type': 'violin'}],
                   [{'type': 'scatter3d'}, {'type': 'bar'}, {'type': 'sunburst'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        for idx, col in enumerate(self.df.columns[:-1]):
            row, col_pos = positions[idx]
            fig.add_trace(
                go.Violin(
                    y=self.df[col],
                    name=col,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.color_palette[idx % len(self.color_palette)],
                    opacity=0.7
                ),
                row=row, col=col_pos
            )
        
        fig.add_trace(
            go.Scatter3d(
                x=self.df['Driver Age'],
                y=self.df['Annual Mileage (x1000 km)'],
                z=self.df['Insurance Premium ($)'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.df['Previous Accidents'],
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Accidents", x=0.02)
                ),
                text=[f"Age: {age}<br>Mileage: {mil}k<br>Premium: ${prem:.2f}<br>Accidents: {acc}"
                      for age, mil, prem, acc in zip(
                          self.df['Driver Age'],
                          self.df['Annual Mileage (x1000 km)'],
                          self.df['Insurance Premium ($)'],
                          self.df['Previous Accidents'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=3, col=1
        )
        
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            importance_df = self.feature_importance.sort_values('Importance', ascending=True).tail(10)
            fig.add_trace(
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='lightgreen',
                    text=importance_df['Importance'].round(3),
                    textposition='outside'
                ),
                row=3, col=2
            )
        
        risk_segments = pd.cut(self.df['Insurance Premium ($)'], 
                              bins=[0, 490, 500, 510, float('inf')],
                              labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
        segment_data = pd.DataFrame({
            'Segment': risk_segments,
            'Driver Age': self.df['Driver Age'],
            'Count': 1
        }).groupby(['Segment', 'Driver Age']).count().reset_index()
        
        fig.add_trace(
            go.Sunburst(
                labels=['All'] + segment_data['Segment'].tolist() + segment_data['Driver Age'].astype(str).tolist(),
                parents=[''] + ['All'] * len(segment_data['Segment'].unique()) + segment_data['Segment'].tolist(),
                values=[len(self.df)] + segment_data.groupby('Segment')['Count'].sum().tolist() + segment_data['Count'].tolist(),
                branchvalues="total",
                marker=dict(colorscale='RdYlGn', cmid=0.5),
                textinfo="label+percent parent"
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            height=1000,
            showlegend=False,
            title_text="Advanced Feature Analysis Dashboard",
            title_font_size=20,
            template=self.template
        )
        
        return fig
    
    def create_model_performance_dashboard(self):
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Model Comparison - RMSE', 'Model Comparison - R²', 'Training vs Validation',
                          'Cross-Validation Performance', 'Overfitting Analysis', 'Model Rankings',
                          'Error Distribution', 'Performance Radar', 'Model Complexity vs Performance'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'box'}, {'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'scatterpolar'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        models = self.model_results['Model'].tolist()
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=self.model_results['Val_RMSE'],
                name='Validation RMSE',
                marker_color='lightcoral',
                text=self.model_results['Val_RMSE'].round(3),
                textposition='outside'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=self.model_results['Val_R2'],
                name='Validation R²',
                marker_color='lightblue',
                text=self.model_results['Val_R2'].round(4),
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.model_results['Train_RMSE'],
                y=self.model_results['Val_RMSE'],
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(
                    size=10,
                    color=self.model_results['Overfit_Score'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Overfit Score")
                ),
                name='Train vs Val'
            ),
            row=1, col=3
        )
        
        cv_data = []
        for model, cv_rmse in zip(models, self.model_results['CV_RMSE']):
            cv_data.extend([cv_rmse] * 5)
        
        fig.add_trace(
            go.Box(
                x=[m for m in models for _ in range(5)],
                y=cv_data,
                marker_color='lightgreen',
                name='CV Performance'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.model_results['Overfit_Score'],
                y=self.model_results['Val_R2'],
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(
                    size=self.model_results['Val_RMSE'] * 20,
                    color=self.model_results['Val_MAE'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Val MAE")
                ),
                name='Overfit vs Performance'
            ),
            row=2, col=2
        )
        
        model_ranking = self.model_results.sort_values('Val_R2', ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=model_ranking['Val_R2'],
                y=model_ranking['Model'],
                orientation='h',
                marker_color='gold',
                text=model_ranking['Val_R2'].round(4),
                textposition='outside',
                name='Top Models'
            ),
            row=2, col=3
        )
        
        errors = self.model_results['Val_RMSE'] - self.model_results['Train_RMSE']
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=20,
                marker_color='lightpink',
                name='Error Distribution'
            ),
            row=3, col=1
        )
        
        top_5_models = self.model_results.nsmallest(5, 'Val_RMSE')
        categories = ['RMSE', 'MAE', 'R²', 'MAPE', 'CV Score']
        
        fig.add_trace(
            go.Scatterpolar(
                r=[1] * 5,
                theta=categories,
                fill='toself',
                name='Perfect Score',
                line=dict(color='gray', dash='dash')
            ),
            row=3, col=2
        )
        
        for idx, row in top_5_models.iterrows():
            r_values = [
                1 - (row['Val_RMSE'] / self.model_results['Val_RMSE'].max()),
                1 - (row['Val_MAE'] / self.model_results['Val_MAE'].max()),
                row['Val_R2'],
                1 - (row['Val_MAPE'] / self.model_results['Val_MAPE'].max()),
                1 - (row['CV_RMSE'] / self.model_results['CV_RMSE'].max())
            ]
            fig.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=categories,
                    fill='toself',
                    name=row['Model'][:15],
                    opacity=0.6
                ),
                row=3, col=2
            )
        
        complexity_score = np.arange(len(models))
        fig.add_trace(
            go.Scatter(
                x=complexity_score,
                y=self.model_results['Val_R2'],
                mode='markers+lines',
                text=models,
                marker=dict(
                    size=10,
                    color=self.model_results['Val_RMSE'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Val RMSE")
                ),
                line=dict(color='gray', dash='dash'),
                name='Complexity vs Performance'
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Model Performance Analysis Dashboard",
            title_font_size=20,
            template=self.template
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        
        return fig
    
    def create_prediction_analysis(self):
        try:
            predictions_df = pd.read_csv('predictions_holdout_test.csv')
        except:
            predictions_df = pd.DataFrame({
                'Actual': np.random.normal(500, 10, 100),
                'Predicted': np.random.normal(500, 10, 100)
            })
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Actual vs Predicted', 'Residual Plot', 'Error Distribution',
                          'Q-Q Plot', 'Prediction Intervals', 'Error Metrics'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'indicator'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        if 'Actual' in predictions_df.columns and 'Predicted' in predictions_df.columns:
            actual = predictions_df['Actual']
            predicted = predictions_df['Predicted']
            residuals = actual - predicted
            
            fig.add_trace(
                go.Scatter(
                    x=actual,
                    y=predicted,
                    mode='markers',
                    marker=dict(
                        color=np.abs(residuals),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Abs Error")
                    ),
                    name='Predictions'
                ),
                row=1, col=1
            )
            
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=predicted,
                    y=residuals,
                    mode='markers',
                    marker=dict(
                        color=np.abs(residuals),
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Abs Residual")
                    ),
                    name='Residuals'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[predicted.min(), predicted.max()],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Zero Line'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    marker_color='lightblue',
                    name='Residual Distribution'
                ),
                row=1, col=3
            )
            
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    marker=dict(color='blue'),
                    name='Q-Q Points'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                    y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Normal Line'
                ),
                row=2, col=1
            )
            
            sorted_idx = np.argsort(predicted)
            pred_sorted = predicted.iloc[sorted_idx]
            actual_sorted = actual.iloc[sorted_idx]
            
            std_error = np.std(residuals)
            upper_bound = pred_sorted + 1.96 * std_error
            lower_bound = pred_sorted - 1.96 * std_error
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(pred_sorted)),
                    y=pred_sorted,
                    mode='lines',
                    line=dict(color='blue'),
                    name='Predicted'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(actual_sorted)),
                    y=actual_sorted,
                    mode='markers',
                    marker=dict(color='red', size=4),
                    name='Actual'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(upper_bound)),
                    y=upper_bound,
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='95% CI Upper'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(lower_bound)),
                    y=lower_bound,
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty',
                    name='95% CI Lower'
                ),
                row=2, col=2
            )
            
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            r2 = 1 - (np.sum(residuals**2) / np.sum((actual - actual.mean())**2))
            
            fig.add_trace(
                go.Indicator(
                    mode="number+gauge+delta",
                    value=r2,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"R² Score<br>MAE: {mae:.2f}<br>RMSE: {rmse:.2f}"},
                    delta={'reference': 1.0},
                    gauge={'axis': {'range': [0, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 0.5], 'color': "lightgray"},
                               {'range': [0.5, 0.8], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 0.9}}
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Prediction Analysis Dashboard",
            title_font_size=20,
            template=self.template
        )
        
        return fig
    
    def create_risk_profiling(self):
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Risk Distribution by Age', 'Risk Heatmap', 'Risk Segments',
                          'Driver Profiles', 'Premium Clusters', 'Risk Factors'),
            specs=[[{'type': 'box'}, {'type': 'heatmap'}, {'type': 'pie'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter'}, {'type': 'bar'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        age_groups = pd.cut(self.df['Driver Age'], bins=[0, 25, 40, 60, 100], 
                           labels=['Young', 'Middle', 'Senior', 'Elder'])
        
        for group in age_groups.unique():
            mask = age_groups == group
            fig.add_trace(
                go.Box(
                    y=self.df.loc[mask, 'Insurance Premium ($)'],
                    name=str(group),
                    boxmean=True
                ),
                row=1, col=1
            )
        
        pivot_data = self.df.pivot_table(
            values='Insurance Premium ($)',
            index=pd.cut(self.df['Driver Age'], bins=10),
            columns=pd.cut(self.df['Previous Accidents'], bins=5),
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=[str(col) for col in pivot_data.columns],
                y=[str(idx) for idx in pivot_data.index],
                colorscale='RdYlGn_r',
                text=np.round(pivot_data.values, 1),
                texttemplate='%{text}',
                textfont={"size": 8}
            ),
            row=1, col=2
        )
        
        risk_categories = pd.cut(self.df['Insurance Premium ($)'],
                                bins=[0, 490, 500, 510, float('inf')],
                                labels=['Low', 'Medium', 'High', 'Very High'])
        risk_counts = risk_categories.value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.3,
                marker=dict(colors=px.colors.sequential.RdBu)
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=self.df['Driver Age'],
                y=self.df['Driver Experience'],
                z=self.df['Insurance Premium ($)'],
                mode='markers',
                marker=dict(
                    size=self.df['Previous Accidents'] * 3 + 2,
                    color=self.df['Annual Mileage (x1000 km)'],
                    colorscale='Turbo',
                    showscale=True,
                    colorbar=dict(title="Mileage", x=0.02)
                ),
                text=[f"Age: {age}<br>Exp: {exp}<br>Premium: ${prem:.2f}<br>Accidents: {acc}"
                      for age, exp, prem, acc in zip(
                          self.df['Driver Age'],
                          self.df['Driver Experience'],
                          self.df['Insurance Premium ($)'],
                          self.df['Previous Accidents'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        from sklearn.cluster import KMeans
        features_for_clustering = self.df[['Driver Age', 'Previous Accidents', 
                                          'Annual Mileage (x1000 km)', 'Insurance Premium ($)']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_for_clustering)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Driver Age'],
                y=self.df['Insurance Premium ($)'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=clusters,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cluster")
                ),
                text=[f"Cluster: {c}<br>Age: {age}<br>Premium: ${prem:.2f}"
                      for c, age, prem in zip(clusters, self.df['Driver Age'], 
                                             self.df['Insurance Premium ($)'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=2, col=2
        )
        
        risk_factors = pd.DataFrame({
            'Factor': ['High Accidents', 'Young Age', 'High Mileage', 'Low Experience', 'Old Car'],
            'Impact': [
                self.df[self.df['Previous Accidents'] > 2]['Insurance Premium ($)'].mean() - 
                self.df[self.df['Previous Accidents'] <= 2]['Insurance Premium ($)'].mean(),
                self.df[self.df['Driver Age'] < 25]['Insurance Premium ($)'].mean() - 
                self.df[self.df['Driver Age'] >= 25]['Insurance Premium ($)'].mean(),
                self.df[self.df['Annual Mileage (x1000 km)'] > 20]['Insurance Premium ($)'].mean() - 
                self.df[self.df['Annual Mileage (x1000 km)'] <= 20]['Insurance Premium ($)'].mean(),
                self.df[self.df['Driver Experience'] < 5]['Insurance Premium ($)'].mean() - 
                self.df[self.df['Driver Experience'] >= 5]['Insurance Premium ($)'].mean(),
                self.df[self.df['Car Age'] > 10]['Insurance Premium ($)'].mean() - 
                self.df[self.df['Car Age'] <= 10]['Insurance Premium ($)'].mean()
            ]
        }).sort_values('Impact', ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=risk_factors['Impact'],
                y=risk_factors['Factor'],
                orientation='h',
                marker=dict(
                    color=risk_factors['Impact'],
                    colorscale='RdYlGn_r',
                    showscale=False
                ),
                text=risk_factors['Impact'].round(2),
                textposition='outside'
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Risk Profiling Dashboard",
            title_font_size=20,
            template=self.template
        )
        
        return fig