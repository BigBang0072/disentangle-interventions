import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import plotly.graph_objects as go

from app_utils import *
from non_overlap_intv_solver import redistribute_probability_mass


###########################################################################
#####################       Global Variables       ########################
###########################################################################
all_networks={
                "asia"  :load_network("asia"),
                "alarm" :load_network("alarm")
            }

#Making the CPD good boy
total_distribute_mass=0.05
redistribute_probability_mass(all_networks["asia"],total_distribute_mass)
redistribute_probability_mass(all_networks["alarm"],total_distribute_mass)

sample_size_choice=[10,100,1000,10000,100000,"infinite"]
table_columns=["Actual Nodes","Actual Category",
                "Predicted Nodes","Predicted Category"]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app=dash.Dash("Root Cause Analysis",
            external_stylesheets=external_stylesheets,)
app.config.suppress_callback_exceptions=True

###########################################################################
#####################     Creating the Layout      ########################
###########################################################################
#Creating a basic layout
app.layout=html.Div(children=[
    #Initializing the First tab
    dcc.Tabs(id="tabs",value="synth",children=[
        dcc.Tab(label="Synthetic Experiment",value="synth"),
        dcc.Tab(label="Real Dataset",value="real"),
    ]),
    html.Div(id="tab_output"),
])
#Callback function for the tabs
@app.callback(Output("tab_output","children"),
                [Input("tabs","value")])
def render_tabs(tab):
    if tab=="synth":
        return render_synthetic_tab()
    elif tab=="real":
        return create_graph_plot(alarm_network.base_graph,
                                alarm_network.topo_level)
    else:
        raise NotImplementedError

#Function to render the synthetic experiment tab
def render_synthetic_tab():
    '''
    This fucntion will render the synthetic experiment tab.
    '''
    content=html.Div([
        #Creating the div for configuration block and graph
        html.Div([
            #Creating the configuration block
            html.Div([
                html.H2("Experiment Configuration",style={"textAlign":"center"}),
                #Selecting the type of Graph
                html.Label("Select Synthetic Graph:"),
                dcc.Dropdown(
                    options=[
                        {"label":"Asia","value":"asia"},
                        {"label":"Alarm","value":"alarm"},
                        {"label":"Flipkart","value":"flipkart"},
                    ],
                    id="graph_type",
                    value="asia",
                ),

                #Synthetic Prediction Sample Size
                html.Label("Sample Size for Root Cause Prediction:"),
                dcc.Slider(
                    min=0,
                    max=len(sample_size_choice)-1,
                    marks={idx:str(size)
                            for idx,size in enumerate(sample_size_choice)},
                    value=len(sample_size_choice)-1,
                    id="root_sample_size",
                ),

                #CheckBox for samples size to be used for evaluation
                html.Label("Sample Size for Synthetic Evaluation:"),
                dcc.Checklist(
                    options=[
                        {"label":10, "value":10},
                        {"label":100,"value":100},
                        {"label":1000, "value":1000},
                        {"label":10000,"value":10000},
                        {"label":100000, "value":100000},
                        {"label":"infinite", "value":"infinite"},
                    ],
                    value=[10,100,1000,10000,"infinite"],
                    id="eval_sample_size",
                ),

                #Adding the Starting button for processing
                html.Button("Evaluate on Synthetic Mixture",id="eval_button"),

            ],style={'display':"inline-block","width":"49%",
                    "border":"1px black solid"}),

            #Creating the Causal Graph
            html.Div([
                html.H2("Causal Graph",style={"textAlign":"center"}),
                # dcc.Graph(id="causal_graph")
                html.Div(id="causal_graph"),
            ],style={'display':"inline-block","width":"49%",
                    "border":"1px black solid"})
        ]),



        #Creating the div for Component prediction and Evaluation Metrics
        html.Div([
            #Creating the configuration block
            html.Div([
                html.H2("Root Cause Prediction",style={"textAlign":"center"}),
                html.Div(id="matched_configs_tbl",children=[
                    html.Table([html.Tr("S.No.")]+[
                        html.Tr([html.Th(col) for col in table_columns])
                    ]),
                ])
            ],style={'display':"inline-block","width":"49%",
                    "border":"1px black solid"}),
            #Creating the Causal Graph
            html.Div([
                html.H2("Prediction Error Metrics",style={"textAlign":"center"}),
                dcc.Graph(id="metric_graph")
            ],style={'display':"inline-block","width":"49%",
                    "border":"1px black solid"})
        ])
    ])

    return content

#Call back to plot the graph
@app.callback(Output("causal_graph","children"),
            [Input("graph_type","value")])
def update_causal_graph(graph_type):
    network=all_networks[graph_type]

    # return create_graph_plot(network.base_graph,
    #                         network.topo_level)
    return create_graph_cytoscape(network.base_graph,
                                network.topo_level)


#Now we will create the evaluation function on different sample size
@app.callback([Output("metric_graph","figure"),
               Output("matched_configs_tbl","children")],
                [Input("eval_button","n_clicks")],
                [State("graph_type","value",),
                State("root_sample_size","value"),
                 State("eval_sample_size","value")]
            )
def evaluate_on_samples(n_clicks,graph_type,root_size_idx,eval_sizes):
    network=all_networks[graph_type]
    #Adding the root sample size
    root_size=sample_size_choice[root_size_idx]
    if root_size not in eval_sizes:
        eval_sizes.append(root_size)
    #First of all we have to generate a random do config
    print("Getting the Do-Config")
    do_config=get_random_internvention_config(network)

    #Now we will evaluate the graph on different sample sizes
    avg_jaccard_sim_list=[]
    avg_mse_list=[]
    tbl_element=None
    for size in eval_sizes:
        avg_jaccard_sim,avg_mse,tbl=disentangle_and_evaluate(
                                    network,do_config,
                                    size,table_columns)

        #Keeping all the metrics
        avg_jaccard_sim_list.append(avg_jaccard_sim)
        avg_mse_list.append(avg_mse)

        if size==root_size:
            tbl_element=tbl

    #Now we are ready to plot the metrics
    # pdb.set_trace()
    fig=plot_evaluation_metrics(avg_jaccard_sim_list,avg_mse_list,eval_sizes)

    return fig,tbl_element


if __name__=="__main__":
    app.run_server(debug=True)
