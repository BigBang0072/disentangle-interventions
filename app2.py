import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_table

from non_overlap_intv_solver import redistribute_probability_mass
from app_utils import load_network

###########################################################################
#####################       Global Variables       ########################
###########################################################################
all_networks={
                "asia"  :load_network("asia"),
                "alarm" :load_network("alarm"),
                "flipkart":load_network("flipkart_7jul19")
            }

#Making the CPD good boy
total_distribute_mass=0.05
redistribute_probability_mass(all_networks["asia"],total_distribute_mass)
redistribute_probability_mass(all_networks["alarm"],total_distribute_mass)
# redistribute_probability_mass(all_networks["flipkart"],total_distribute_mass)

#Dummmy Sample size and Date Range configs
date_range_size={
    "last 7 days":1000,
    "last 14 days":10000,
    "last 1 month":100000,
}

external_stylesheets = [#'https://codepen.io/chriddyp/pen/bWLwgP.css',
                        dbc.themes.BOOTSTRAP,
                        ]
app=dash.Dash("Root Cause Analysis",
            external_stylesheets=external_stylesheets,)
app.config.suppress_callback_exceptions=True

###########################################################################
#####################     Creating the Layout      ########################
###########################################################################

app.layout=dbc.Container([
    dbc.Card([
        html.H1("Root Cause Analysis",style={"textAlign":"center"}),
        #Taking the input from user
        dbc.Form([
            #Selecting the graph type
            dbc.FormGroup([
                dbc.Label("Select Graph for Anomaly Analysis:",
                            html_for="graph_type",width=5),
                dbc.Col(
                    dcc.Dropdown(
                        options=[
                            {"label":"Asia","value":"asia"},
                            {"label":"Alarm","value":"alarm"},
                            {"label":"Flipkart","value":"flipkart"},
                        ],
                        id="graph_type",
                        value="asia",
                    ),
                    width=5,
                )
            ],row=True),
            #Selecting the Date range for analysis
            dbc.FormGroup([
                dbc.Label("Date Range to analyse Anomaly:",
                            html_for="sample_size",width=5),
                dbc.Col(
                    dcc.Dropdown(
                        options=[
                            {"label":range,"value":size}
                                for range,size in date_range_size.items()
                        ],
                        id="sample_size",
                    ),
                    width=5,
                ),
            ],row=True),
            #Button to Start the analysis
            dbc.Button("Find Root Cause",color="primary",id="go_button")
        ]),
    ]),

    #Now we will create tabs for different purposes
    dbc.Card([
        dbc.Tabs(
            [
                dbc.Tab(label="Root Causes Visualization",tab_id="root_viz"),
                dbc.Tab(label="Conterfactual Effect",tab_id="counter_efffect"),
            ],
            id="tabs",
            active_tab="root_viz",
        ),
        html.Div(id="tab_content")
    ]),
],fluid=True)


###########################################################################
############################    CALLBACKS      ############################
###########################################################################
#Now we will have to generate the tab content
@app.callback(Output("tab_content","children"),
            [Input("tabs","active_tab")])
def render_tab_content(active_tab):
    '''
    '''
    if active_tab=="root_viz":
        return render_root_viz_tab()
    elif active_tab=="counter_effect":
        return html.H3("Work in Progres")
    else:
        raise NotImplementedError


###########################################################################
############################    HELPER FUNCTIONS      #####################
###########################################################################
def render_root_viz_tab():
    '''
    '''
    content=html.Div([
        dbc.Row([
        #Creating the root Cause Table
            dbc.Col(id="root_cause_tbl",children=[
                html.H2("Root Cause Table",style={"textAlign":"center"}),
                dash_table.DataTable(
                    id="root_cause",
                    columns=[{"id":"Decision Group","name":"Decision Group"}],
                    row_selectable="single",
                    style_header={
                        "backgroundColor":"rgb(230, 230, 230)",
                        "fontWeight":"bold",
                        "border":"2px solid black",
                        "textAlign":"center"
                    },
                    style_data={"border":"1px solid black",
                                "height":"auto"},
                )
            ],width=3,style={"border":"1px black solid"},),

            #Creating the Prediction Graph
            dbc.Col([
                html.H2("Root Cause Visualization",
                        style={"textAlign":"center"}),
                dcc.Graph(id="node_viz_graph")
            ],width=4,style={"border":"1px black solid"},),

            dbc.Col([
                html.H2("Strength of Contribution",
                        style={"textAlign":"center"}),
                dcc.Graph(id="strength_viz_graph")
            ],width=4,style={"border":"1px black solid"},)
        ],justify="around")
    ])

    return content

if __name__=="__main__":
    app.run_server(debug=True,port=8050)
