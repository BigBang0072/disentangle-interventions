import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import networkx as nx
import numpy as np

from data_handle import BnNetwork
#Import code downloaded from other's github repo for directed edges in plotly
from addEdge import addEdge


#Iniitaizing the global graph attributes
node_size=20
node_color="Blue"
line_width=2
line_color="rgba(220,217,200,0.8)"

def load_network(network_name):
    '''
    This will initialize all the graph once as a global variable.
    '''
    graph_name=network_name
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    base_network=BnNetwork(modelpath)

    return base_network

def generate_topo_level_layout(topo_level,scale=5):
    '''
    This function will generate a custom generate loyout pattern for the nodes
    based on our idea of level-orderded topological ordering.
    '''
    topo_loc={}
    y_coord=len(topo_level)      #Iniitial height of the graph
    step_size=scale                #Amount to decrease each level
    max_width=max([len(level_nodes) for level_nodes in topo_level.values()])

    #Now going one by one at each level
    for level,level_nodes in topo_level.items():
        num_nodes=len(level_nodes)
        x_coords=np.linspace(0.0,scale*max_width,num_nodes+2)[1:-1]
        for tidx,node in enumerate(level_nodes):
            node_coord=np.array([x_coords[tidx],y_coord])
            topo_loc[node]=node_coord

        #Going down the level height
        y_coord-=step_size
    return topo_loc

def create_graph_plot(graph_obj,topo_level):
    '''
    This fucntion will create plotly graph to be using the base_graph
    arrtibutes in the base_network.
    '''
    #Getting the planar node attributes for the graph
    # planar_locs=nx.random_layout(graph_obj) #present in new version of networkx
    planar_locs=generate_topo_level_layout(topo_level)

    #Creating the edges of the graph
    edges_x=[]
    edges_y=[]
    for fro,to in graph_obj.edges():
        start_point=planar_locs[fro].tolist()
        end_point=planar_locs[to].tolist()
        edges_x,edges_y=addEdge(start_point,end_point,
                                edges_x,edges_y,
                                lengthFrac=0.85,
                                arrowPos="end",
                                arrowLength=0.04,
                                arrowAngle=30,
                                dotSize=node_size)
    #Making the edge trace
    edge_trace=go.Scatter(x=edges_x,y=edges_y,
                            line=dict(width=line_width,
                                        color=line_color),
                            hoverinfo="none",
                            mode="lines")


    #Now we will create the nodes trace
    node_x=[]
    node_y=[]
    node_name=[]
    for node in graph_obj.nodes():
        x,y=planar_locs[node].tolist()
        node_x.append(x)
        node_y.append(y)
        node_name.append(node)
    #Craeting the node trace
    node_trace=go.Scatter(x=node_x,y=node_y,
                            mode="markers",
                            hoverinfo="text",
                            marker=dict(showscale=False,
                                        color=node_color,
                                        size=node_size,
                                        ),
                            hovertext=node_name,
                            )
    fig=go.Figure(data=[edge_trace,node_trace],
                    layout=go.Layout(showlegend=False,
                                    hovermode="closest",
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False,
                                                zeroline=False,
                                                showticklabels=False),
                                    yaxis=dict(showgrid=False,
                                                zeroline=False,
                                                showticklabels=False),
                                    )
                    )
    fig.update_layout(yaxis=dict(scaleanchor="x",
                                scaleratio=1),
                        plot_bgcolor="rgb(255,255,255)")

    return fig
