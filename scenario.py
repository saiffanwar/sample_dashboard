import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from utils import *  # Ensure these are defined
from PIL import Image
import dash_ag_grid as dag

# Load dataset
df = pd.read_csv('50_startups.csv')

expenses_data = pd.read_csv('company_financials.csv')


logo = Image.open('assets/logo.png')

# Create a Dash application
app = Dash(__name__, external_stylesheets=[])  # <--- Set theme here
server = app.server
app.title = "VIOMAR"

app.layout = dbc.Container([html.Div([
    html.Div([
        html.Img(src=app.get_asset_url('logo.png'), style={'height': '100px'}),
    ], style={
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'padding-top': '20px'
    }),
    html.H1("Cost Profit Analysis", style={'textAlign': 'center'}),

    html.Div(
        style={
            'display': 'flex',
            'gap': '20px',
            'margin-bottom': '20px',
            'flex-wrap': 'wrap'
        },
        children=[
            html.Div([
                html.Label(""),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value='R&D Spend'
                )
            ], style={'width': '250px'}),


        ]
    ),

    html.Div([
        html.Label("Data to include in Profit Projection Model:")
    ]),
    html.Div([
    html.Div([
        # One row per checklist item + input
        html.Div([
            dcc.Checklist(
                id='check-rd',
                options=[{'label': 'R&D Spend', 'value': 'R&D Spend'}],
                value=[],
                style={'margin-right': '10px'}
            ),
            dcc.Input(id='future-rd', type='number', value=150000,
                      placeholder='Future R&D',
                      style={'width': '120px'})
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

        html.Div([
            dcc.Checklist(
                id='check-admin',
                options=[{'label': 'Administration', 'value': 'Administration'}],
                value=[],
                style={'margin-right': '10px'}
            ),
            dcc.Input(id='future-admin', type='number', value=150000,
                      placeholder='Future Admin',
                      style={'width': '120px'})
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

        html.Div([
            dcc.Checklist(
                id='check-ms',
                options=[{'label': 'Marketing Spend', 'value': 'Marketing Spend'}],
                value=[],
                style={'margin-right': '10px'}
            ),
            dcc.Input(id='future-ms', type='number', value=150000,
                      placeholder='Future Marketing',
                      style={'width': '120px'})
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),
    ],
    style={'display': 'flex', 'flex-direction': 'column'}),

    html.Button('Project', id='project-button', n_clicks=0, style={'margin-top': '10px'})
    ], className='dashboard_card', style={'padding': '40px', 'border-radius': '10px', 'background-color': '#f9f9f9'}),

    dcc.Graph(id='profit-graph', figure=plot_relationship(df, 'R&D Spend', 'Profit')),

    html.H1("Income & Expenses", style={'textAlign': 'center', 'padding-top': '30px'}),
     html.Div([
         html.Label("Expenses Data:"),
                dcc.Dropdown(
                    id='budget-dropdown',
                    options=['Net Income', 'Spending Breakdown', 'Income Breakdown'],
                    value='Net Income',
                )
            ], style={'width': '250px'}),

    dcc.Graph(id='budget-graph', figure=plot_budget(expenses_data, 'Net Income')[0]),

    html.Div([
    html.Label("Upload your own data:"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '20px 20px 20px 20px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
])
], style={
#    'width': '90%',
    'margin': '100px',
    'padding': '20px',
    'backgroundColor': '#f9f9f9',
    'borderRadius': '10px'
})
], fluid=True)


# Callback to update graph
@app.callback(
    Output('profit-graph', 'figure'),
    Input('project-button', 'n_clicks'),
#    Input('my-checklist', 'value'),
    Input('check-rd', 'value'),
    Input('check-admin', 'value'),
    Input('check-ms', 'value'),
    Input('column-dropdown', 'value'),
    State('future-rd', 'value'),
    State('future-ms', 'value'),
    State('future-admin', 'value'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def update_profit_graph(n_clicks, check_rd, check_admin, check_ms, selected_column, future_rd, future_ms, future_admin):
    # Perform projection using the future_x value
    selected_features = check_rd + check_admin + check_ms
    if callback_context.triggered_id == 'project-button':
        # Get the value from the button
        lr_model, projected_value = project(df, 'Profit', features=selected_features, future_x = [future_rd, future_ms, future_admin])
    else:
        lr_model=None
        projected_value=None
    fig = plot_relationship(df, selected_column, 'Profit', lr_model, projected_value, future_x = [future_rd, future_ms, future_admin])
    return fig

@app.callback(
    Output('budget-graph', 'figure'),
    Input('budget-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=True,
)

def update_budget_graph(plot_type, contents, filename, date):
#    if callback_context.triggered_id == 'budget-dropdown':
        # Get the value from the button
    fig, max_exp, max_inc = plot_budget(expenses_data, plot_type)
    if plot_type == 'Net Income':
        if callback_context.triggered_id == 'upload-data':
            fig = add_new_data(filename[0], fig, max_exp, max_inc)
    return fig

#@app.callback(
#    Output('output-data-upload', 'children'),
#    Output('budget-graph', 'figure'),
#    prevent_initial_call=True,
#    allow_duplicate=True
#)
#def update_output(contents, fig, filename, date):
##    gpt_pred(extract_text_from_pdf(filename))  # Assuming this function is defined in utils.py
#    fig = add_new_data('structured_data.csv', fig)
#    return fig

if __name__ == '__main__':
    app.run(debug=True)
