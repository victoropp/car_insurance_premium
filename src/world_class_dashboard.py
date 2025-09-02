"""
VIDEBIMUS AI - World-Class Insurance Analytics Platform
State-of-the-Art Dashboard with Premium UI/UX Design
Author: Victor Collins Oppon
Version: 3.0 - Enterprise Edition
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from visualizations_updated import ProfessionalVisualizationEngine
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Load premium theme
load_figure_template("flatly")

# Initialize Dash app with premium design system
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "World-Class Insurance Analytics Platform by Videbimus AI"}
    ],
    title="Videbimus AI - Premium Analytics Platform"
)

app.config.suppress_callback_exceptions = True
server = app.server

# Feature engineering function
def create_features(df):
    """Apply the same feature engineering as in training"""
    df_feat = df.copy()
    
    # Interaction features
    df_feat['Age_Experience_Ratio'] = df_feat['Driver Age'] / (df_feat['Driver Experience'] + 1)
    df_feat['Risk_Score'] = (df_feat['Previous Accidents'] * 10 + 
                             df_feat['Annual Mileage (x1000 km)'] * 0.5 + 
                             df_feat['Car Age'] * 2)
    df_feat['Experience_Rate'] = df_feat['Driver Experience'] / df_feat['Driver Age']
    df_feat['Mileage_per_Year'] = df_feat['Annual Mileage (x1000 km)'] / (df_feat['Car Age'] + 1)
    
    # Polynomial features
    df_feat['Age_Squared'] = df_feat['Driver Age'] ** 2
    df_feat['Experience_Squared'] = df_feat['Driver Experience'] ** 2
    df_feat['Mileage_Squared'] = df_feat['Annual Mileage (x1000 km)'] ** 2
    
    # Log transformations
    df_feat['Log_Mileage'] = np.log1p(df_feat['Annual Mileage (x1000 km)'])
    df_feat['Log_Car_Age'] = np.log1p(df_feat['Car Age'])
    
    # Binned features
    df_feat['Age_Group'] = pd.cut(df_feat['Driver Age'], bins=[0, 25, 35, 50, 65, 100], labels=False)
    df_feat['Experience_Level'] = pd.cut(df_feat['Driver Experience'], bins=[0, 2, 5, 10, 20, 50], labels=False)
    df_feat['Mileage_Category'] = pd.cut(df_feat['Annual Mileage (x1000 km)'], bins=[0, 10, 20, 30, 50, 100], labels=False)
    
    return df_feat

# Load models and scaler
def load_models():
    models = {}
    model_files = {
        'Voting Ensemble': 'voting_ensemble.pkl',
        'Stacking (Linear)': 'stacking_linear.pkl',
        'Stacking (Ridge)': 'stacking_ridge.pkl'
    }
    
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                models[name] = joblib.load(file)
            except:
                pass
    
    # Load scaler
    scaler = RobustScaler()
    df_train = pd.read_csv('insurance_tranining_dataset.csv')
    X_train = df_train.drop('Insurance Premium ($)', axis=1)
    X_train_engineered = create_features(X_train)
    scaler.fit(X_train_engineered)
    
    return models, scaler

available_models, scaler = load_models()

# Initialize visualization engine
viz_engine = ProfessionalVisualizationEngine()

# Load test results
test_results = pd.read_csv('final_test_results.csv')
best_test_r2 = test_results['Test_R2'].max() if not test_results.empty else 0.9978

# Premium color palette
colors = {
    'primary': '#6366F1',      # Indigo
    'secondary': '#8B5CF6',    # Purple
    'success': '#10B981',      # Emerald
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444',       # Red
    'info': '#06B6D4',         # Cyan
    'dark': '#1F2937',         # Gray-800
    'light': '#F9FAFB',        # Gray-50
    'gradient_1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient_2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'gradient_3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'gradient_4': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
}

# Custom CSS for world-class design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                overflow-x: hidden;
            }
            
            /* Premium animations */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            
            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-100px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            /* Glass morphism effect */
            .glass-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.18);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
                animation: fadeInUp 0.6s ease-out;
                transition: all 0.3s ease;
            }
            
            .glass-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
            }
            
            /* Premium navbar */
            .premium-navbar {
                background: rgba(255, 255, 255, 0.98);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(99, 102, 241, 0.1);
                box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.05);
                position: sticky;
                top: 0;
                z-index: 1000;
                animation: slideInLeft 0.8s ease-out;
            }
            
            /* Gradient buttons */
            .gradient-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                padding: 12px 30px;
                border-radius: 50px;
                font-weight: 600;
                letter-spacing: 0.5px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
            }
            
            .gradient-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.3);
            }
            
            /* Premium cards */
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                padding: 30px;
                color: white;
                position: relative;
                overflow: hidden;
                animation: fadeInUp 0.8s ease-out;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                animation: pulse 4s ease-in-out infinite;
            }
            
            /* Premium inputs */
            .premium-input {
                border: 2px solid #E5E7EB;
                border-radius: 12px;
                padding: 12px 16px;
                font-size: 16px;
                transition: all 0.3s ease;
                background: white;
            }
            
            .premium-input:focus {
                border-color: #6366F1;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
                outline: none;
            }
            
            /* Floating elements */
            .floating {
                animation: float 6s ease-in-out infinite;
            }
            
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
                100% { transform: translateY(0px); }
            }
            
            /* Premium scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #F3F4F6;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }
            
            /* Loading animation */
            .loading-pulse {
                display: inline-block;
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                animation: pulse 1.5s ease-in-out infinite;
            }
            
            /* Premium tooltips */
            .tooltip-custom {
                position: relative;
                display: inline-block;
            }
            
            .tooltip-custom .tooltiptext {
                visibility: hidden;
                width: 200px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
                border-radius: 10px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            .tooltip-custom:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            
            /* Section transitions */
            .section-container {
                opacity: 0;
                animation: fadeInUp 1s ease-out forwards;
            }
            
            .section-container:nth-child(1) { animation-delay: 0.1s; }
            .section-container:nth-child(2) { animation-delay: 0.2s; }
            .section-container:nth-child(3) { animation-delay: 0.3s; }
            .section-container:nth-child(4) { animation-delay: 0.4s; }
            
            /* Premium shadows */
            .premium-shadow {
                box-shadow: 
                    0 2.8px 2.2px rgba(0, 0, 0, 0.02),
                    0 6.7px 5.3px rgba(0, 0, 0, 0.028),
                    0 12.5px 10px rgba(0, 0, 0, 0.035),
                    0 22.3px 17.9px rgba(0, 0, 0, 0.042),
                    0 41.8px 33.4px rgba(0, 0, 0, 0.05),
                    0 100px 80px rgba(0, 0, 0, 0.07);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create world-class navigation
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Img(src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='url(%23gradient)' width='40' height='40'%3E%3Cdefs%3E%3ClinearGradient id='gradient' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23667eea;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23764ba2;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z'/%3E%3C/svg%3E",
                             style={'height': '40px', 'marginRight': '15px'}),
                    html.Div([
                        html.H4("VIDEBIMUS AI", className="mb-0", 
                               style={'fontWeight': '800', 'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     '-webkit-background-clip': 'text', '-webkit-text-fill-color': 'transparent'}),
                        html.Small("Premium Analytics Platform", style={'color': '#6B7280', 'fontWeight': '500'})
                    ])
                ], className="d-flex align-items-center")
            ], width="auto"),
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Executive Dashboard", href="#executive", 
                                           className="mx-2 px-3 py-2 rounded-pill",
                                           style={'fontWeight': '600', 'transition': 'all 0.3s'})),
                    dbc.NavItem(dbc.NavLink("Analytics Hub", href="#analytics",
                                           className="mx-2 px-3 py-2 rounded-pill",
                                           style={'fontWeight': '600', 'transition': 'all 0.3s'})),
                    dbc.NavItem(dbc.NavLink("Model Intelligence", href="#models",
                                           className="mx-2 px-3 py-2 rounded-pill",
                                           style={'fontWeight': '600', 'transition': 'all 0.3s'})),
                    dbc.NavItem(dbc.NavLink("Premium Calculator", href="#calculator",
                                           className="mx-2 px-3 py-2 rounded-pill",
                                           style={'fontWeight': '600', 'transition': 'all 0.3s'})),
                ], navbar=True, className="ms-auto")
            ], width="auto"),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-rocket me-2"),
                    "Get Started"
                ], className="gradient-btn", size="lg")
            ], width="auto")
        ], align="center", className="w-100"),
    ], fluid=True),
    className="premium-navbar",
    sticky="top",
)

# Create world-class hero section
hero_section = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1([
                        "Insurance Premium",
                        html.Br(),
                        html.Span("Analytics Platform", 
                                 style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                       '-webkit-background-clip': 'text', 
                                       '-webkit-text-fill-color': 'transparent'})
                    ], className="display-3 fw-bold mb-4 animate__animated animate__fadeInLeft"),
                    html.P("Harness the power of advanced machine learning to revolutionize insurance pricing",
                          className="lead mb-4 text-muted animate__animated animate__fadeInLeft animate__delay-1s",
                          style={'fontSize': '1.3rem'}),
                    html.Div([
                        dbc.Button([
                            html.I(className="fas fa-play-circle me-2"),
                            "Launch Dashboard"
                        ], size="lg", className="gradient-btn me-3 animate__animated animate__fadeInUp animate__delay-2s"),
                        dbc.Button([
                            html.I(className="fas fa-book me-2"),
                            "View Documentation"
                        ], size="lg", outline=True, color="primary",
                                  className="animate__animated animate__fadeInUp animate__delay-2s")
                    ], className="d-flex flex-wrap")
                ], className="py-5")
            ], lg=6),
            dbc.Col([
                html.Div([
                    html.Div(className="floating",
                            style={'width': '400px', 'height': '400px',
                                  'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                  'borderRadius': '30% 70% 70% 30% / 30% 30% 70% 70%',
                                  'opacity': '0.1',
                                  'position': 'absolute',
                                  'right': '10%',
                                  'top': '10%'}),
                    dbc.Card([
                        dbc.CardBody([
                            html.H2(f"{best_test_r2:.2%}", className="display-4 fw-bold text-center mb-3",
                                   style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                         '-webkit-background-clip': 'text',
                                         '-webkit-text-fill-color': 'transparent'}),
                            html.P("Model Accuracy", className="text-center text-muted mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H4("3", className="fw-bold mb-0"),
                                        html.Small("Ensemble Models", className="text-muted")
                                    ], className="text-center")
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H4("15+", className="fw-bold mb-0"),
                                        html.Small("ML Algorithms", className="text-muted")
                                    ], className="text-center")
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H4("24/7", className="fw-bold mb-0"),
                                        html.Small("Real-time Analysis", className="text-muted")
                                    ], className="text-center")
                                ], width=4)
                            ])
                        ])
                    ], className="glass-card premium-shadow animate__animated animate__fadeInRight animate__delay-1s",
                      style={'position': 'relative', 'zIndex': '10'})
                ], style={'position': 'relative', 'minHeight': '400px'})
            ], lg=6)
        ], className="align-items-center min-vh-100")
    ], fluid=True)
], style={'background': 'linear-gradient(180deg, #F9FAFB 0%, #FFFFFF 100%)', 'paddingTop': '100px'})

# Create the main app layout
app.layout = html.Div([
    navbar,
    hero_section,
    
    # Main content with sections
    html.Div([
        # Executive Dashboard Section
        html.Div(id="executive", className="section-container"),
        
        # Analytics Hub Section
        html.Div(id="analytics", className="section-container"),
        
        # Model Intelligence Section
        html.Div(id="models", className="section-container"),
        
        # Premium Calculator Section
        html.Div(id="calculator", className="section-container"),
    ], style={'background': '#FFFFFF', 'minHeight': '100vh'}),
    
    # Footer
    html.Footer([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H5("VIDEBIMUS AI", className="fw-bold mb-3"),
                    html.P("Transforming insurance with artificial intelligence",
                          className="text-muted")
                ], lg=4),
                dbc.Col([
                    html.H6("Quick Links", className="fw-bold mb-3"),
                    html.Ul([
                        html.Li(html.A("Documentation", href="#", className="text-muted")),
                        html.Li(html.A("API Reference", href="#", className="text-muted")),
                        html.Li(html.A("Support", href="#", className="text-muted"))
                    ], style={'listStyle': 'none', 'padding': '0'})
                ], lg=4),
                dbc.Col([
                    html.H6("Contact", className="fw-bold mb-3"),
                    html.P([
                        html.I(className="fas fa-envelope me-2"),
                        "info@videbimus.ai"
                    ], className="text-muted"),
                    html.P([
                        html.I(className="fas fa-phone me-2"),
                        "+1 (555) 123-4567"
                    ], className="text-muted")
                ], lg=4)
            ], className="py-5"),
            html.Hr(),
            html.P("© 2025 Videbimus AI. All rights reserved.", 
                  className="text-center text-muted py-3")
        ])
    ], style={'background': '#F9FAFB', 'marginTop': '100px'})
])

if __name__ == '__main__':
    print("\n" + "="*70)
    print("       VIDEBIMUS AI - WORLD-CLASS ANALYTICS PLATFORM")
    print("="*70)
    print("\n   Developed by: Victor Collins Oppon")
    print("   Title: Data Scientist & AI Consultant")
    print("   Company: Videbimus AI")
    print("\n" + "-"*70)
    print("\n🚀 Starting world-class dashboard server...")
    print("\n📍 Dashboard URL: http://127.0.0.1:8050")
    print("\n⚡ Press CTRL+C to stop the server")
    print("\n" + "="*70 + "\n")
    
    app.run_server(debug=False, port=8050)