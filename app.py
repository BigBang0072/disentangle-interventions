import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
from plotly.graph_objects as go

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
    html.Div(id="tabs-content")
])
#Callback function for the tabs
@app.callback(Output("tabs-content","children"),
                [Input("tabs","value")])
def render_tabs(tab):
    if tab=="synth":
        return html.Div([html.H3("Synth Placeholder")])
    elif tab=="real":
        return html.Div([html.H3("Real Placeholder")])
    else:
        raise NotImplementedError

#

if __name__=="__main__":
    

    app.run_server(debug=True)
