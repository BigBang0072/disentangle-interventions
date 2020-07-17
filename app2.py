import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_table

from non_overlap_intv_solver import *
from app_utils import load_network,disentangle

###########################################################################
#####################       Global Variables       ########################
###########################################################################
all_networks={
                "asia"  :load_network("asia"),
                "alarm" :load_network("alarm"),
                "flipkart":load_network("flipkart_7jul19")
            }
all_do_configs={
        "asia"  :get_random_internvention_config(all_networks["asia"]),
        "alarm" :get_random_internvention_config(all_networks["alarm"]),
        "flipkart":None
}

#Global variable to hold the predicted configs for plotting
predicted_configs={}
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
    "last 1 year":"infinite",
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
                        value=1000,
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
            [Input("tabs","active_tab"),
             Input("go_button","n_clicks")],
            [State("graph_type","value"),
             State("sample_size","value")])
def render_tab_content(active_tab,n_clicks,graph_type,sample_size):
    '''
    '''
    if active_tab=="root_viz":
        return render_root_viz_tab(graph_type,sample_size)
    elif active_tab=="counter_effect":
        return html.H3("Work in Progres")
    else:
        raise NotImplementedError

@app.callback(Output("strength_viz_graph","figure"),
                [Input("root_cause_tbl","derived_virtual_row_ids"),
                 Input("root_cause_tbl","selected_rows")])
def pull_from_strength_piechart(order_row_ids,sidx):
    '''
    '''
    # assert len(derived_virtual_row_ids)<=1,"One selection boy!!"
    if order_row_ids==None:
        return go.Figure()
    print("sidx:{}\t order:{}".format(sidx,order_row_ids))
    #Now lets create the pie-chard data
    labels=[]
    values=[]
    pull=[]
    global predicted_configs
    for tidx,(cid,config) in enumerate(predicted_configs.items()):
        labels.append(cid)
        #Getting the mixing coefficient value
        pi=config[-1]
        values.append(pi)

        #Creating the pull vector
        if sidx!=None and sidx[-1]==tidx:
            pull.append(0.2)
        else:
            pull.append(0.0)

    #Now we will plot the piechart
    fig=go.Figure(data=[
        go.Pie(labels=labels,values=values,pull=pull)
    ])

    return fig

###########################################################################
############################    HELPER FUNCTIONS      #####################
###########################################################################
def render_root_viz_tab(graph_type,sample_size):
    '''
    '''
    #Finding the root cause for the given graph and sample_size
    network=all_networks[graph_type]
    do_config=all_do_configs[graph_type]
    #Now lets disentangle the mixture
    global predicted_configs
    root_cause_tbl,predicted_configs=disentangle(graph_type,
                                        network,do_config,sample_size)

    #Creating the Layout for the tab
    content=html.Div([
        dbc.Row([
        #Creating the root Cause Table
            dbc.Col(id="root_cause_div",children=[
                html.H2("Root Cause Table",style={"textAlign":"center"}),
                root_cause_tbl,
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
