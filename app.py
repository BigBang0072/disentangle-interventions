import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
import plotly.graph_objects as go

from app_utils import load_network,create_graph_plot


###########################################################################
#####################       Global Variables       ########################
###########################################################################
asia_network=load_network("asia")
alarm_network=load_network("alarm")
app=dash.Dash(__name__)

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
    html.Div(children=[
            dcc.Graph(id="network_graph")
    ],),
])
#Callback function for the tabs
@app.callback(Output("network_graph","figure"),
                [Input("tabs","value")])
def render_tabs(tab):
    if tab=="synth":
        return create_graph_plot(asia_network.base_graph,
                                asia_network.topo_level)
    elif tab=="real":
        return create_graph_plot(alarm_network.base_graph,
                                alarm_network.topo_level)
    else:
        raise NotImplementedError

#

if __name__=="__main__":
    app.run_server(debug=True)
