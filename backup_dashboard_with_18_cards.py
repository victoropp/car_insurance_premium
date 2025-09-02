"""
Complete Dashboard with Individual Cards for All Charts
This version properly separates ALL charts into individual cards
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import os
from src.visualizations_updated import ProfessionalVisualizationEngine
from src.visualizations_individual import IndividualChartEngine
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app with professional theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
                    {"name": "description", "content": "Advanced ML-powered insurance premium analytics platform by Videbimus AI"},
                    {"name": "author", "content": "Victor Collins Oppon - Videbimus AI"}
                ])
app.title = "Videbimus AI - Insurance Premium Analytics"
server = app.server

# Initialize visualization engines
viz_engine = ProfessionalVisualizationEngine()
individual_charts = IndividualChartEngine()

# Load test results for metrics
test_results = pd.read_csv('data/final_test_results.csv')
best_test_r2 = test_results['Test_R2'].max() if not test_results.empty else 0.9978

# Feature engineering function (matching the training pipeline)
def create_features(df):
    """Apply the same feature engineering as in training"""
    df_feat = df.copy()
    
    # Interaction features
    df_feat['Age_Experience_Ratio'] = df_feat['Driver Age'] / (df_feat['Driver Experience'] + 1)
    df_feat['Accidents_Per_Year_Driving'] = df_feat['Previous Accidents'] / (df_feat['Driver Experience'] + 1)
    df_feat['Mileage_Per_Year_Driving'] = df_feat['Annual Mileage (x1000 km)'] / (df_feat['Driver Experience'] + 1)
    df_feat['Car_Age_Driver_Age_Ratio'] = df_feat['Car Age'] / df_feat['Driver Age']
    df_feat['Experience_Rate'] = df_feat['Driver Experience'] / df_feat['Driver Age']
    
    # Polynomial features
    df_feat['Driver_Age_Squared'] = df_feat['Driver Age'] ** 2
    df_feat['Experience_Squared'] = df_feat['Driver Experience'] ** 2
    df_feat['Accidents_Squared'] = df_feat['Previous Accidents'] ** 2
    
    # Risk indicators
    df_feat['High_Risk_Driver'] = ((df_feat['Driver Age'] < 25) | (df_feat['Driver Age'] > 65)).astype(int)
    df_feat['New_Driver'] = (df_feat['Driver Experience'] < 2).astype(int)
    df_feat['Old_Car'] = (df_feat['Car Age'] > 10).astype(int)
    df_feat['High_Mileage'] = (df_feat['Annual Mileage (x1000 km)'] > 20).astype(int)
    
    # Composite risk score
    df_feat['Risk_Score'] = (
        df_feat['High_Risk_Driver'] * 2 + 
        df_feat['New_Driver'] * 3 + 
        df_feat['Previous Accidents'] * 4 + 
        df_feat['Old_Car'] * 1 +
        df_feat['High_Mileage'] * 1
    )
    
    return df_feat

# Load models and scaler
def load_models():
    models = {}
    model_files = {
        'Stacking (Linear) - Best Performer': 'models/stacking_linear.pkl',
        'Stacking (Ridge)': 'models/stacking_ridge.pkl',
        'Voting Ensemble': 'models/voting_ensemble.pkl'
    }
    
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                models[name] = joblib.load(file)
            except:
                pass
    
    # Also load the scaler from training
    scaler = RobustScaler()
    df_train = pd.read_csv('data/insurance_tranining_dataset.csv')
    df_train_engineered = create_features(df_train)
    X_train = df_train_engineered.drop('Insurance Premium ($)', axis=1)
    scaler.fit(X_train)
    
    return models, scaler

available_models, scaler = load_models()

# Professional color scheme
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#73AB84',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6C91C2',
    'light': '#F8F9FA',
    'dark': '#2D3436',
    'background': '#F5F7FA',
    'card_bg': '#FFFFFF',
    'border': '#E1E8ED'
}

# Card styling - clean and consistent
card_style = {
    'backgroundColor': colors['card_bg'],
    'border': f"1px solid {colors['border']}",
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.08)',
    'marginBottom': '20px',
    'height': '450px',
    'overflow': 'hidden'
}

# Navigation bar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-car-side me-2", style={'fontSize': '24px'}),
                    html.Span("Videbimus AI", style={'fontSize': '20px', 'fontWeight': 'bold'})
                ], className="d-flex align-items-center")
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Executive", href="#executive", external_link=True)),
                    dbc.NavItem(dbc.NavLink("Analysis", href="#analysis", external_link=True)),
                    dbc.NavItem(dbc.NavLink("Models", href="#models", external_link=True)),
                    dbc.NavItem(dbc.NavLink("Calculator", href="#calculator", external_link=True)),
                ], navbar=True)
            ])
        ])
    ], fluid=True),
    color="dark",
    dark=True,
    sticky="top",
    className="mb-4"
)

# Create metric card helper
def create_metric_card(title, value, subtitle, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-2", style={'fontSize': '12px'}),
            html.H3(value, className="mb-2", style={'color': colors[color], 'fontWeight': 'bold'}),
            html.P(subtitle, className="text-muted mb-0", style={'fontSize': '11px'})
        ])
    ], style={'border': f'1px solid {colors["border"]}', 'borderRadius': '8px', 
              'boxShadow': '0 1px 3px rgba(0,0,0,0.05)'})

# Welcome Section
welcome_section = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Insurance Premium Analytics Platform", 
                   className="text-center mb-3", 
                   style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("Advanced Machine Learning for Precision Premium Calculation", 
                  className="text-center text-muted mb-4", 
                  style={'fontSize': '18px'})
        ])
    ])
], className="mb-4")

# Metrics Row
df = pd.read_csv('data/insurance_tranining_dataset.csv')
metrics_row = dbc.Row([
    dbc.Col(create_metric_card(
        "Dataset Size", 
        f"{len(df):,}", 
        "Training Records",
        "primary"
    ), width=3),
    dbc.Col(create_metric_card(
        "Average Premium", 
        f"${df['Insurance Premium ($)'].mean():.0f}", 
        "Across All Records",
        "info"
    ), width=3),
    dbc.Col(create_metric_card(
        "Best Model R²", 
        f"{best_test_r2:.4f}", 
        "Test Set Score",
        "success"
    ), width=3),
    dbc.Col(create_metric_card(
        "Premium Range", 
        f"${df['Insurance Premium ($)'].min():.0f} - ${df['Insurance Premium ($)'].max():.0f}", 
        "Min - Max Values",
        "warning"
    ), width=3),
], className="mb-4")

# Executive Summary Section with Individual Cards
executive_section = dbc.Container([
    html.Div(id="executive"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-chart-pie me-2"),
                "Executive Summary"
            ], className="mb-3", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("High-level overview of insurance premium patterns and key metrics.", 
                  className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    
    # Row 1: Premium Distribution, Risk Factors, Model Performance
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_premium_distribution(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_risk_factors(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_model_performance(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
    
    # Row 2: Correlations, Age Analysis, Experience Impact
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_correlation_heatmap(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_age_vs_premium(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_experience_vs_premium(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
], className="mb-5")

# Detailed Analysis Section
analysis_section = dbc.Container([
    html.Div(id="analysis"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-microscope me-2"),
                "Detailed Analysis"
            ], className="mb-3 mt-5", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("In-depth exploration of factors affecting insurance premiums.", 
                  className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    
    # Row 1: Vehicle Age, Mileage Distribution, Accidents Impact
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_vehicle_age_analysis(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_mileage_distribution(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_accidents_impact(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
    
    # Row 2: Premium Percentiles, Feature Importance, Risk Segmentation
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_premium_percentiles(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_feature_importance(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_risk_segmentation(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
], className="mb-5")

# Model Performance Section
models_section = dbc.Container([
    html.Div(id="models"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-robot me-2"),
                "Model Performance"
            ], className="mb-3 mt-5", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("Comprehensive model evaluation and comparison metrics.", 
                  className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    
    # Row 1: Top Models, Ensemble Models, Overfitting Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_top_models(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_ensemble_models(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_overfitting_analysis(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
    
    # Row 2: Model Rankings, Performance Metrics, Best Model Indicator
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_model_rankings(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_performance_metrics(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=individual_charts.create_best_model_indicator(),
                        config={'displayModeBar': False},
                        style={'height': '380px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
], className="mb-5")

# Premium Calculator Section (keeping original)
calculator_section = dbc.Container([
    html.Div(id="calculator"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-calculator me-2"),
                "Premium Calculator"
            ], className="mb-3 mt-5", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("Calculate insurance premiums with our AI-powered model.", 
                  className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Input Controls Column
                dbc.Col([
                    html.H5("Enter Driver & Vehicle Information", className="mb-4", 
                           style={'color': colors['dark'], 'fontWeight': 'bold'}),
                    
                    # Driver Information
                    html.Div([
                        html.H6("Driver Information", className="mb-3", 
                               style={'color': colors['secondary'], 'fontWeight': 'bold'}),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Driver Age", html_for="driver-age"),
                                dbc.Input(id="driver-age", type="number", min=18, max=100, value=35,
                                         placeholder="Enter age (18-100)"),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Label("Years of Experience", html_for="driver-experience"),
                                dbc.Input(id="driver-experience", type="number", min=0, max=82, value=10,
                                         placeholder="Enter experience (0-82)"),
                            ], md=6),
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Previous Accidents", html_for="previous-accidents"),
                                dbc.Input(id="previous-accidents", type="number", min=0, max=10, value=0,
                                         placeholder="Number of accidents (0-10)"),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Label("Annual Mileage (x1000 km)", html_for="annual-mileage"),
                                dbc.Input(id="annual-mileage", type="number", min=1, max=100, value=15,
                                         placeholder="Enter mileage (1-100)"),
                            ], md=6),
                        ], className="mb-3"),
                    ]),
                    
                    html.Hr(),
                    
                    # Vehicle Information
                    html.Div([
                        html.H6("Vehicle Information", className="mb-3",
                               style={'color': colors['secondary'], 'fontWeight': 'bold'}),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Car Manufacturing Year", html_for="car-year"),
                                dbc.Input(id="car-year", type="number", min=1990, max=2024, value=2019,
                                         placeholder="Enter year (1990-2024)"),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Label("Vehicle Age", html_for="vehicle-age-display"),
                                html.Div(id="vehicle-age-display", 
                                        className="form-control bg-light",
                                        style={'height': '38px', 'lineHeight': '38px'}),
                            ], md=6),
                        ], className="mb-3"),
                    ]),
                    
                    html.Hr(),
                    
                    # Model Selection
                    html.Div([
                        dbc.Label("Select Model", html_for="model-selector"),
                        dcc.Dropdown(
                            id="model-selector",
                            options=[{"label": name, "value": name} for name in available_models.keys()],
                            value=list(available_models.keys())[0] if available_models else None,
                            className="mb-3"
                        ),
                    ]),
                    
                    # Calculate Button
                    dbc.Button(
                        "Calculate Premium",
                        id="calculate-button",
                        color="primary",
                        size="lg",
                        className="w-100 mt-3",
                        n_clicks=0
                    ),
                ], md=5),
                
                # Results Column
                dbc.Col([
                    html.H5("Prediction Results", className="mb-4",
                           style={'color': colors['dark'], 'fontWeight': 'bold'}),
                    
                    # Results Display
                    html.Div(id="prediction-result", className="mb-4"),
                    
                    # AI Explanation
                    html.Div(id="ai-explanation", className="mt-4"),
                    
                ], md=7),
            ])
        ])
    ], style={'border': f'1px solid {colors["border"]}', 'borderRadius': '8px', 
              'boxShadow': '0 2px 8px rgba(0,0,0,.1)'})
], className="mb-5")

# Footer
footer = dbc.Container([
    html.Hr(className="my-4"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Developed by", className="text-center mb-3", 
                       style={'color': colors['secondary'], 'fontSize': '14px'}),
                html.H5("Victor Collins Oppon", className="text-center mb-1", 
                       style={'color': colors['primary'], 'fontWeight': 'bold'}),
                html.P("Data Scientist & AI Consultant", className="text-center mb-2", 
                      style={'color': colors['secondary'], 'fontSize': '14px', 'fontWeight': '500'}),
                html.P([
                    html.I(className="fas fa-building me-2"),
                    html.A("Videbimus AI", 
                          href="https://www.videbimusai.com",
                          target="_blank",
                          style={'color': 'inherit', 'textDecoration': 'none'}),
                    html.Span(" | ", className="mx-2"),
                    html.I(className="fas fa-envelope me-2"),
                    html.A("consulting@videbimusai.com",
                          href="mailto:consulting@videbimusai.com",
                          style={'color': 'inherit', 'textDecoration': 'none'}),
                    html.Span(" | ", className="mx-2"),
                    html.I(className="fas fa-globe me-2"),
                    html.A("https://www.videbimusai.com",
                          href="https://www.videbimusai.com",
                          target="_blank",
                          style={'color': 'inherit', 'textDecoration': 'none'}),
                ], className="text-center text-muted", style={'fontSize': '13px'}),
            ])
        ])
    ])
], className="mt-5 mb-4")

# App Layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        welcome_section,
        metrics_row,
        html.Hr(className="my-4", style={'borderColor': colors['border']}),
        executive_section,
        analysis_section,
        models_section,
        calculator_section,
        footer,
    ], fluid=True, style={'backgroundColor': colors['background'], 'padding': '20px'})
], style={'backgroundColor': colors['background'], 'minHeight': '100vh'})

# Callbacks (same as original)
@app.callback(
    Output('vehicle-age-display', 'children'),
    Input('car-year', 'value')
)
def update_vehicle_age(car_year):
    if car_year:
        current_year = 2024
        vehicle_age = current_year - car_year
        return f"{vehicle_age} years"
    return "Enter manufacturing year"

@app.callback(
    [Output('prediction-result', 'children'),
     Output('ai-explanation', 'children')],
    [Input('calculate-button', 'n_clicks')],
    [State('driver-age', 'value'),
     State('driver-experience', 'value'),
     State('previous-accidents', 'value'),
     State('annual-mileage', 'value'),
     State('car-year', 'value'),
     State('model-selector', 'value')]
)
def calculate_premium(n_clicks, driver_age, driver_experience, previous_accidents, 
                      annual_mileage, car_year, selected_model):
    if n_clicks == 0 or not all([driver_age, driver_experience is not None, 
                                 previous_accidents is not None, annual_mileage, 
                                 car_year, selected_model]):
        return html.Div(), html.Div()
    
    # Create input dataframe
    current_year = 2024
    car_age = current_year - car_year
    
    input_data = pd.DataFrame({
        'Driver Age': [driver_age],
        'Driver Experience': [driver_experience],
        'Previous Accidents': [previous_accidents],
        'Annual Mileage (x1000 km)': [annual_mileage],
        'Car Age': [car_age],
        'Car Manufacturing Year': [car_year]
    })
    
    # Apply feature engineering
    input_engineered = create_features(input_data)
    
    # Scale features
    input_scaled = scaler.transform(input_engineered)
    
    # Make prediction
    model = available_models[selected_model]
    prediction = model.predict(input_scaled)[0]
    
    # Create result display
    result_display = dbc.Card([
        dbc.CardBody([
            html.H3(f"${prediction:.2f}", 
                   className="text-center mb-3",
                   style={'color': colors['primary'], 'fontSize': '48px', 'fontWeight': 'bold'}),
            html.P("Estimated Annual Premium", 
                  className="text-center text-muted",
                  style={'fontSize': '16px'}),
            
            html.Hr(),
            
            # Risk Assessment
            html.Div([
                html.H6("Risk Assessment", className="mb-3", style={'color': colors['dark']}),
                
                # Risk factors
                html.Div([
                    dbc.Progress(
                        value=min(100, (previous_accidents / 3) * 100),
                        label=f"Accident Risk: {previous_accidents} accidents",
                        color="danger" if previous_accidents > 1 else "success",
                        className="mb-2",
                        style={"height": "25px"}
                    ),
                    dbc.Progress(
                        value=min(100, (driver_age / 100) * 100),
                        label=f"Age Factor: {driver_age} years",
                        color="warning" if driver_age < 25 or driver_age > 65 else "success",
                        className="mb-2",
                        style={"height": "25px"}
                    ),
                    dbc.Progress(
                        value=min(100, (annual_mileage / 50) * 100),
                        label=f"Mileage Factor: {annual_mileage}k km/year",
                        color="warning" if annual_mileage > 30 else "success",
                        className="mb-2",
                        style={"height": "25px"}
                    ),
                ])
            ])
        ])
    ], style={'border': f'2px solid {colors["primary"]}', 'borderRadius': '8px'})
    
    # AI Explanation
    risk_level = "High" if prediction > 3000 else "Medium" if prediction > 2000 else "Low"
    risk_color = colors['danger'] if risk_level == "High" else colors['warning'] if risk_level == "Medium" else colors['success']
    
    explanation = dbc.Alert([
        html.H5("🤖 AI Analysis", className="alert-heading"),
        html.Hr(),
        html.P([
            html.Strong("Risk Level: "),
            html.Span(risk_level, style={'color': risk_color, 'fontWeight': 'bold'})
        ]),
        html.P(f"Based on the provided information, our {selected_model} model has analyzed multiple risk factors."),
        html.Ul([
            html.Li(f"Driver with {driver_experience} years of experience" + 
                   (" (New driver - higher risk)" if driver_experience < 2 else "")),
            html.Li(f"Vehicle age: {car_age} years" + 
                   (" (Older vehicle - moderate risk)" if car_age > 10 else "")),
            html.Li(f"Previous accidents: {previous_accidents}" + 
                   (" (Clean record - lower risk)" if previous_accidents == 0 else " (Accident history - higher risk)")),
            html.Li(f"Annual mileage: {annual_mileage},000 km" + 
                   (" (High mileage - increased risk)" if annual_mileage > 30 else ""))
        ]),
        html.P(f"The model achieved {best_test_r2:.2%} accuracy on test data.", className="mb-0")
    ], color="info", dismissable=True, fade=True)
    
    return result_display, explanation

if __name__ == '__main__':
    app.run(debug=False, port=8050)