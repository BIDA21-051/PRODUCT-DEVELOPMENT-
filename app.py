import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import datetime

# Load and sample data
df = pd.read_csv('C:\\Users\\bida21-051\\AI_Dashboard\\raw_synthetic_iis_logs.csv')
df_sample = df.sample(n=5000, random_state=42).copy()

# Preprocess
df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'])
df_sample['week'] = df_sample['timestamp'].dt.isocalendar().week

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    html.H1('AI Solutions Performance Dashboard', className='mb-4 text-center'),

    dcc.Tabs([
        dcc.Tab(label='Global Overview', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Market Penetration", className="card-title"),
                            dcc.Graph(
                                id='world-map',
                                figure=px.choropleth(
                                    df_sample,
                                    locations='country',
                                    locationmode='country names',
                                    color='conversion_status',
                                    hover_data=['campaign_tag', 'session_duration'],
                                    color_continuous_scale=px.colors.sequential.Plasma
                                ).update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                            )
                        ])
                    ], className='mb-4')
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Key Metrics", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(f"{len(df_sample['country'].dropna().unique())}", className="metric-value"),
                                    html.Small("Active Countries", className="metric-label")
                                ], className="metric-box"),
                                dbc.Col([
                                    html.Div(f"{df_sample['conversion_status'].mean():.1%}", className="metric-value"),
                                    html.Small("Global Conversion", className="metric-label")
                                ], className="metric-box"),
                                dbc.Col([
                                    html.Div(f"${df_sample['session_duration'].quantile(0.9):.0f}", className="metric-value"),
                                    html.Small("Top 10% Session", className="metric-label")
                                ], className="metric-box")
                            ]),
                            html.Hr(),
                            dcc.Graph(
                                figure=px.line(
                                    df_sample.groupby('week')['conversion_status'].mean().reset_index(),
                                    x='week',
                                    y='conversion_status',
                                    title='Weekly Conversion Trend'
                                ).update_layout(
                                    template='plotly_dark',
                                    margin=dict(t=40, b=20)
                                )
                            )
                        ])
                    ])
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Campaign ROI Analysis", className="card-title"),
                            dcc.Graph(
                                figure=px.treemap(
                                    df_sample.dropna(subset=['campaign_tag', 'traffic_source']),
                                    path=['campaign_tag', 'traffic_source'],
                                    values='conversion_status',
                                    color='session_duration',
                                    color_continuous_scale='Viridis'
                                ).update_layout(plot_bgcolor='rgba(0,0,0,0)')
                            )
                        ])
                    ])
                ])
            ])
        ]),
        dcc.Tab(label='Sales Analytics', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Conversion Funnel by Source", className="card-title"),
                            dcc.Graph(
                                figure=px.funnel_area(
                                    df_sample,
                                    names='traffic_source',
                                    values='conversion_status',
                                    color='traffic_source',
                                    template='plotly_dark'
                                )
                            )
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Regional Performance Matrix", className="card-title"),
                            dcc.Dropdown(
                                id='region-select',
                                options=[{'label': r, 'value': r} for r in df_sample['region'].dropna().unique()],
                                multi=True,
                                placeholder='Select regions'
                            ),
                            dcc.Graph(id='region-heatmap')
                        ])
                    ])
                ], width=8)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Campaign Effectiveness", className="card-title"),
                            dcc.Graph(
                                figure=px.scatter(
                                    df_sample,
                                    x='session_duration',
                                    y='conversion_status',
                                    color='campaign_tag',
                                    trendline="lowess"
                                ).update_layout(
                                    xaxis_title='Engagement Time',
                                    yaxis_title='Conversion Rate'
                                )
                            )
                        ])
                    ])
                ])
            ])
        ]),
        dcc.Tab(label='Product Insights', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Product Performance", className="card-title"),
                            dcc.Graph(
                                figure=px.sunburst(
                                    df_sample.dropna(subset=['product', 'country']),
                                    path=['product', 'country'],
                                    values='conversion_status',
                                    color='engagement_score',
                                    color_continuous_scale='RdYlGn'
                                )
                            )
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Live Anomaly Detection", className="card-title"),
                            html.Div(id='anomaly-alerts', className='alert-container'),
                            dcc.Interval(id='refresh-interval', interval=10000, n_intervals=0)
                        ])
                    ])
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Engagement Metrics", className="card-title"),
                            dcc.Graph(
                                figure=go.Figure(
                                    data=go.Indicator(
                                        mode="gauge+number",
                                        value=df_sample['session_duration'].mean(),
                                        title={'text': "Average Session Duration"},
                                        gauge={
                                            'axis': {
                                                'range': [None, df_sample['session_duration'].max()]
                                            }
                                        }
                                    )
                                ).update_layout(height=300)
                            )
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Conversion Flow", className="card-title"),
                            dcc.Graph(
                                figure=px.parallel_categories(
                                    df_sample,
                                    dimensions=['traffic_source', 'product', 'conversion_status'],
                                    color='conversion_status',
                                    color_continuous_scale=px.colors.sequential.Inferno
                                )
                            )
                        ])
                    ])
                ], width=8)
            ])
        ])
    ])
], fluid=True, style={'backgroundColor': '#0a0a0a'})

# Callback: Region Heatmap
@app.callback(
    Output('region-heatmap', 'figure'),
    Input('region-select', 'value')
)
def update_heatmap(regions):
    filtered = df_sample if not regions else df_sample[df_sample['region'].isin(regions)]
    return px.density_heatmap(
        filtered,
        x='timestamp',
        y='product',
        z='conversion_status',
        histfunc="avg",
        template='plotly_dark'
    )

# Callback: Anomaly Alerts
@app.callback(
    Output('anomaly-alerts', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_alerts(n):
    alerts = [
        {"type": "demo_request", "status": "critical", "message": "30% drop in demo requests - APAC region"},
        {"type": "conversion", "status": "warning", "message": "15% decrease in AI Assistant conversions"}
    ]
    return [
        dbc.Alert(
            alert['message'],
            color="danger" if alert['status'] == 'critical' else 'warning',
            className="alert-item"
        ) for alert in alerts
    ]

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
