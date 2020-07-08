import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly.graph_objects as go

def load_graph(graph_name):
    graph_name="graph_name"
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    base_network=BnNetwork(modelpath)

    return base_network

def create_graph_plot(graph_name):
    '''
    This fucntion will create plotly graph to be made
    '''
