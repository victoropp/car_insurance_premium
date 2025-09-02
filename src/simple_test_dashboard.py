import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        html.H1("🚀 Dashboard Test", className="text-center mb-4"),
        dbc.Card([
            dbc.CardBody([
                html.H4("Dashboard is Running Successfully!", className="text-success"),
                html.P("This confirms the dashboard can start and serve content."),
                html.Hr(),
                html.P("✅ Server: Active"),
                html.P("✅ Port 8050: Accessible"),
                html.P("✅ Dash Framework: Working")
            ])
        ])
    ], className="mt-5")
])

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Simple Test Dashboard")
    print("URL: http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run_server(debug=False, port=8050)