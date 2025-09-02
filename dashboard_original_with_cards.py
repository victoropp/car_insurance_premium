"""
Car Insurance Premium Analytics Dashboard - 100% Original Replication with Individual Cards
This dashboard maintains EVERYTHING from the original:
- All 18 charts with exact same data, styling, and titles
- All headers, descriptions, and text exactly as original
- Complete premium calculator functionality
- All navigation, branding, and footer elements
- Individual cards for each chart for better visual separation
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
from src.visualizations_complete import CompleteIndividualCharts
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app with professional theme - EXACT same as original
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
                    {"name": "description", "content": "Advanced ML-powered insurance premium analytics platform by Videbimus AI"},
                    {"name": "author", "content": "Victor Collins Oppon - Videbimus AI"}
                ])
app.title = "Videbimus AI - Insurance Premium Analytics"
server = app.server

# Initialize visualization engine with individual charts
charts = CompleteIndividualCharts()

# Load data and get metrics - EXACT same as original
df = pd.read_csv('data/insurance_tranining_dataset.csv')
test_results = pd.read_csv('data/final_test_results.csv')
best_test_r2 = test_results['Test_R2'].max() if not test_results.empty else 0.9978

# Feature engineering function - EXACT same as original
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

# Load models and scaler - EXACT same as original
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

# Color scheme - EXACT same as original
colors = {
    'primary': '#2E4057',
    'secondary': '#048A81',
    'accent': '#54C6EB',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'danger': '#F44336',
    'dark': '#1A1A2E',
    'light': '#F5F5F5',
    'card_bg': '#FFFFFF',
    'border': '#E0E0E0'
}

# Card style for individual charts
card_style = {
    'backgroundColor': colors['card_bg'],
    'border': f"1px solid {colors['border']}",
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.08)',
    'marginBottom': '20px',
    'padding': '15px'
}

# Navigation bar - EXACT same as original
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Executive Summary", href="#executive", external_link=True)),
        dbc.NavItem(dbc.NavLink("Detailed Analysis", href="#detailed", external_link=True)),
        dbc.NavItem(dbc.NavLink("Model Performance", href="#models", external_link=True)),
        dbc.NavItem(dbc.NavLink("Premium Calculator", href="#calculator", external_link=True)),
    ],
    brand=html.Span([
        html.I(className="fas fa-chart-line me-2"),
        "Videbimus AI - Insurance Premium Analytics"
    ]),
    brand_href="#",
    brand_external_link=True,
    color="dark",
    dark=True,
    fluid=True,
    className="mb-4"
)

# Header section with branding - EXACT same as original
header = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-shield-alt me-3", style={'color': colors['primary']}),
                    "Insurance Premium Analytics Dashboard"
                ], className="display-4 mb-3", style={'fontWeight': 'bold', 'color': colors['dark']}),
                html.P([
                    "Advanced machine learning platform for insurance premium prediction and analysis. ",
                    "Powered by ensemble models achieving ", 
                    html.Span(f"{best_test_r2:.4f} R² accuracy", style={'fontWeight': 'bold', 'color': colors['success']}),
                    " on test data."
                ], className="lead mb-4", style={'fontSize': '1.1rem'}),
                html.Hr(className="my-4"),
                html.P([
                    html.I(className="fas fa-building me-2"),
                    html.A("Videbimus AI", href="https://www.videbimusai.com", target="_blank", 
                          style={'color': colors['primary'], 'textDecoration': 'none', 'fontWeight': 'bold'}),
                    " | ",
                    html.I(className="fas fa-envelope me-2"),
                    html.A("consulting@videbimusai.com", href="mailto:consulting@videbimusai.com",
                          style={'color': colors['secondary'], 'textDecoration': 'none'}),
                    " | ",
                    html.I(className="fas fa-globe me-2"),
                    html.A("https://www.videbimusai.com", href="https://www.videbimusai.com", target="_blank",
                          style={'color': colors['secondary'], 'textDecoration': 'none'})
                ], style={'fontSize': '0.9rem', 'color': colors['dark']})
            ], className="text-center py-4", 
               style={'backgroundColor': colors['light'], 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], width=12)
    ])
], fluid=True, className="mb-5")

# Helper function for metric cards - EXACT same as original
def create_metric_card(title, value, subtitle, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-2", style={'fontSize': '0.9rem'}),
            html.H3(value, className="mb-2", style={'color': colors[color], 'fontWeight': 'bold'}),
            html.P(subtitle, className="text-muted mb-0", style={'fontSize': '0.8rem'})
        ])
    ], style={'border': f'2px solid {colors[color]}20', 'borderRadius': '8px'})

# Metrics row - EXACT same as original
metrics_row = dbc.Row([
    dbc.Col(create_metric_card(
        "Model Accuracy", 
        f"{best_test_r2:.4f}", 
        "Test Set R² Score",
        "success"
    ), width=3),
    dbc.Col(create_metric_card(
        "Total Records", 
        f"{len(df):,}", 
        "Training Samples",
        "primary"
    ), width=3),
    dbc.Col(create_metric_card(
        "Average Premium", 
        f"${df['Insurance Premium ($)'].mean():.2f}", 
        "Mean Value",
        "secondary"
    ), width=3),
    dbc.Col(create_metric_card(
        "Premium Range", 
        f"${df['Insurance Premium ($)'].min():.0f} - ${df['Insurance Premium ($)'].max():.0f}", 
        "Min - Max Values",
        "warning"
    ), width=3),
], className="mb-4")

# Executive Summary Section - Individual cards for each chart
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
    
    # Row 1: Charts 1-3
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_1_premium_distribution(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_2_key_risk_factors(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_3_test_set_performance(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
    
    # Row 2: Charts 4-6
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_4_feature_correlations(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_5_age_vs_premium(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_6_risk_segmentation(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
], fluid=True, className="py-4")

# Detailed Analysis Section - Individual cards for each chart
detailed_section = dbc.Container([
    html.Div(id="detailed"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-microscope me-2"),
                "Detailed Analysis"
            ], className="mb-3", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("In-depth examination of premium drivers and patterns.", 
                  className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    
    # Row 1: Charts 7-9
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_7_driver_experience_impact(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_8_vehicle_age_analysis(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_9_accident_history_effect(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
    
    # Row 2: Charts 10-12
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_10_annual_mileage_distribution(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_11_premium_percentiles(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_12_feature_importance_ranking(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
], fluid=True, className="py-4")

# Model Comparison Section - Individual cards for each chart
model_section = dbc.Container([
    html.Div(id="models"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-robot me-2"),
                "Model Performance"
            ], className="mb-3", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("Comprehensive model evaluation and comparison metrics.", 
                  className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    
    # Row 1: Charts 13-15
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_13_top_10_models_validation(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_14_ensemble_models_test(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_15_overfitting_analysis(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
    
    # Row 2: Charts 16-18
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_16_model_rankings_test(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_17_performance_metrics(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=charts.chart_18_best_model_indicator(),
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=card_style)
        ], md=4),
    ], className="mb-4"),
], fluid=True, className="py-4")

# Premium Calculator Section - EXACT same as original
calculator_section = dbc.Container([
    html.Div(id="calculator"),
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="fas fa-calculator me-2"),
                "Premium Calculator"
            ], className="mb-3", style={'color': colors['dark'], 'fontWeight': 'bold'}),
            html.P("Calculate insurance premiums with our AI-powered model.", 
                  className="text-muted mb-4", style={'fontSize': '14px'})
        ])
    ]),
    
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Left column - Input fields
                dbc.Col([
                    html.H5([html.I(className="fas fa-user-edit me-2"), "Enter Details"], 
                           className="mb-4", style={'color': colors['primary']}),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Driver Age", html_for="driver-age", style={'fontWeight': 'bold'}),
                            dbc.Input(id="driver-age", type="number", value=35, min=18, max=100,
                                    className="mb-3", placeholder="Enter age (18-100)"),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Driving Experience (years)", html_for="experience", style={'fontWeight': 'bold'}),
                            dbc.Input(id="experience", type="number", value=10, min=0, max=50,
                                    className="mb-3", placeholder="Years of experience"),
                        ], md=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Previous Accidents", html_for="accidents", style={'fontWeight': 'bold'}),
                            dbc.Input(id="accidents", type="number", value=0, min=0, max=10,
                                    className="mb-3", placeholder="Number of accidents"),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Annual Mileage (×1000 km)", html_for="mileage", style={'fontWeight': 'bold'}),
                            dbc.Input(id="mileage", type="number", value=15, min=0, max=100,
                                    className="mb-3", placeholder="Annual mileage"),
                        ], md=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Car Manufacturing Year", html_for="car-year", style={'fontWeight': 'bold'}),
                            dbc.Input(id="car-year", type="number", value=2018, min=1990, max=2024,
                                    className="mb-3", placeholder="Manufacturing year"),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Select Model", html_for="model-selector", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id="model-selector",
                                options=[{"label": name, "value": name} for name in available_models.keys()],
                                value=list(available_models.keys())[0] if available_models else None,
                                className="mb-3",
                                style={'borderRadius': '4px'}
                            ),
                        ], md=6),
                    ]),
                    
                    dbc.Button([html.I(className="fas fa-calculator me-2"), "Calculate Premium"], 
                              id="calculate-btn", 
                              color="primary", 
                              size="lg",
                              className="w-100 mt-3",
                              style={'fontWeight': 'bold', 'fontSize': '1.1rem'}),
                ], md=6),
                
                # Right column - Results
                dbc.Col([
                    html.Div(id="prediction-output", className="h-100")
                ], md=6),
            ]),
        ])
    ], style={'border': f'2px solid {colors["primary"]}20', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
], fluid=True, className="py-4")

# Footer - EXACT same as original
footer = dbc.Container([
    html.Hr(className="my-4"),
    dbc.Row([
        dbc.Col([
            html.P([
                "© 2024 ",
                html.A("Videbimus AI", href="https://www.videbimusai.com", target="_blank",
                      style={'color': colors['primary'], 'textDecoration': 'none', 'fontWeight': 'bold'}),
                " | Advanced Analytics Solutions | ",
                html.A("consulting@videbimusai.com", href="mailto:consulting@videbimusai.com",
                      style={'color': colors['secondary'], 'textDecoration': 'none'})
            ], className="text-center text-muted", style={'fontSize': '0.9rem'})
        ])
    ])
], fluid=True, className="mt-5 mb-3")

# Main layout
app.layout = html.Div([
    navbar,
    header,
    dbc.Container([
        metrics_row,
        executive_section,
        detailed_section,
        model_section,
        calculator_section,
    ], fluid=True),
    footer
], style={'backgroundColor': '#FAFAFA'})

# Callback for premium prediction - EXACT same as original
@app.callback(
    Output("prediction-output", "children"),
    [Input("calculate-btn", "n_clicks")],
    [State("driver-age", "value"),
     State("experience", "value"),
     State("accidents", "value"),
     State("mileage", "value"),
     State("car-year", "value"),
     State("model-selector", "value")]
)
def predict_premium(n_clicks, driver_age, experience, accidents, mileage, car_year, selected_model):
    if n_clicks is None or n_clicks == 0:
        # Initial state - show instructions
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-info-circle", 
                          style={'fontSize': '4rem', 'color': colors['primary'], 'opacity': '0.5'}),
                    html.H5("Ready to Calculate", className="mt-3 mb-2", style={'color': colors['dark']}),
                    html.P("Enter your details and click 'Calculate Premium' to get your personalized quote.",
                          className="text-muted"),
                    html.Hr(),
                    html.Small("Our AI model analyzes multiple risk factors to provide accurate premium estimates.",
                             className="text-muted")
                ], className="text-center py-4")
            ])
        ], style={'border': f'2px dashed {colors["primary"]}40', 'borderRadius': '8px', 'height': '100%'})
    
    # Validate inputs
    if not all([driver_age, experience, mileage, car_year]):
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Please fill in all required fields."
        ], color="warning")
    
    if selected_model not in available_models:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-circle me-2"),
            "Please select a valid model."
        ], color="danger")
    
    # Calculate car age
    current_year = 2024
    car_age = current_year - car_year
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Driver Age': [driver_age],
        'Driver Experience': [experience],
        'Previous Accidents': [accidents],
        'Annual Mileage (x1000 km)': [mileage],
        'Car Manufacturing Year': [car_year],
        'Car Age': [car_age],
        'Insurance Premium ($)': [0]  # Placeholder, will be dropped
    })
    
    # Apply feature engineering
    input_engineered = create_features(input_data)
    X_input = input_engineered.drop('Insurance Premium ($)', axis=1)
    
    # Scale the input
    X_scaled = scaler.transform(X_input)
    
    # Make prediction
    model = available_models[selected_model]
    prediction = model.predict(X_scaled)[0]
    
    # Calculate risk level
    if accidents > 2 or driver_age < 25 or experience < 2:
        risk_level = "High"
        risk_color = "danger"
    elif accidents > 0 or driver_age > 65 or experience < 5:
        risk_level = "Medium"
        risk_color = "warning"
    else:
        risk_level = "Low"
        risk_color = "success"
    
    # Create result display - EXACT same as original
    result_display = dbc.Card([
        dbc.CardBody([
            html.H5([html.I(className="fas fa-chart-line me-2"), "Prediction Results"], 
                   className="mb-4", style={'color': colors['primary']}),
            
            # Premium amount
            html.Div([
                html.H6("Estimated Annual Premium", className="text-muted mb-2"),
                html.H2(f"${prediction:,.2f}", 
                       style={'color': colors['success'], 'fontWeight': 'bold'}),
            ], className="text-center mb-4", 
               style={'backgroundColor': colors['light'], 'padding': '20px', 'borderRadius': '8px'}),
            
            # Risk assessment
            dbc.Row([
                dbc.Col([
                    html.H6("Risk Level", className="text-muted mb-2"),
                    dbc.Badge(risk_level, color=risk_color, className="px-3 py-2",
                             style={'fontSize': '1rem'})
                ], md=6, className="text-center"),
                dbc.Col([
                    html.H6("Model Used", className="text-muted mb-2"),
                    dbc.Badge(selected_model.split('-')[0].strip(), color="info", className="px-3 py-2",
                             style={'fontSize': '0.9rem'})
                ], md=6, className="text-center"),
            ], className="mb-4"),
            
            # Key factors
            html.Hr(),
            html.H6([html.I(className="fas fa-list-ul me-2"), "Key Factors"], className="mb-3"),
            html.Ul([
                html.Li(f"Driver Age: {driver_age} years", className="mb-1"),
                html.Li(f"Experience: {experience} years", className="mb-1"),
                html.Li(f"Previous Accidents: {accidents}", className="mb-1"),
                html.Li(f"Annual Mileage: {mileage},000 km", className="mb-1"),
                html.Li(f"Vehicle Age: {car_age} years", className="mb-1"),
            ], style={'fontSize': '0.9rem'}),
            
            # Disclaimer
            html.Hr(),
            html.Small([
                html.I(className="fas fa-info-circle me-1"),
                "This is an AI-generated estimate based on historical data patterns."
            ], className="text-muted")
        ])
    ], style={'border': f'2px solid {colors["success"]}40', 'borderRadius': '8px'})
    
    return result_display

# Run the app
if __name__ == '__main__':
    app.run(debug=False, port=8050)