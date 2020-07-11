import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_cytoscape as cyto
import dash_table

import networkx as nx
import numpy as np

from data_handle import BnNetwork
from non_overlap_intv_solver import *
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

##############################################################################
########################  GRAPH LAYOUT DESIGN ################################
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

def create_graph_cytoscape(network,topo_level,matched_config,selected_row_id,):
    '''
    '''
    graph_obj=network.base_graph
    #Getting the location with topo-ordered levels
    planar_locs=generate_topo_level_layout(topo_level)

    #Now we have to first get the nodes we want to paint
    node_colors={}
    if selected_row_id!=None:
        #Get the selected row
        assert len(selected_row_id)<=1,"Single Selection allowed"
        selected_row=matched_config[selected_row_id[0]]

        #Get the compoenent elements from the row
        numberify=lambda x:[int(ele) for ele in x[1:-1].split(",")]
        actual_nodes=numberify(selected_row[0])
        actual_cats=numberify(selected_row[1])
        predict_nodes=numberify(selected_row[2])
        predict_cats=numberify(selected_row[3])
        #We know only node,cat will appear in this and they are sorted
        actual_pairs=set(zip(actual_nodes,actual_cats))
        predict_pairs=set(zip(predict_nodes,predict_cats))

        for pair in actual_pairs.union(predict_pairs):
            node_name=network.topo_i2n[pair[0]]
            #If the node is present in both the prediction and actual
            if pair in actual_pairs and pair in predict_pairs:
                node_colors[node_name]="green"
            elif pair in actual_pairs:
                node_colors[node_name]="blue"
            else:
                node_colors[node_name]="red"
    #Now defining the function to get the color
    def get_class_assignment(node):
        if node in node_colors:
            return node_colors[node]
        else:
            return "generic"


    #Creating the node element list
    node_list=[]
    for node in graph_obj.nodes():
        #Retreiving the position of node
        x,y=planar_locs[node].tolist()

        #Creating the node element
        element={"data":{"id":node,"label":node},
                "position":{"x":x,"y":y},
                "classes":get_class_assignment(node)
                }

        node_list.append(element)

    #Creating the edges of the graph
    edge_list=[]
    for fro,to in graph_obj.edges():
        element={
            "data":{"source":fro,"target":to,"label":fro+"_"+to},
        }
        edge_list.append(element)

    #Now we are ready to create graph
    graph=cyto.Cytoscape(
            id="causal_graph_cyto",
            layout={"name":"breadthfirst"},
            style={'width': '100%', 'height': '400px'},
            elements=node_list+edge_list,
            stylesheet=[
                {
                    "selector":"node",
                    "style":{
                        "content":"data(label)",
                        "line_color":"black",
                    }
                },
                {
                    "selector":"edge",
                    "style":{
                        "curve-style":"bezier",
                        "target-arrow-shape":"triangle",

                    }
                },
                {
                    "selector":".red",
                    "style":{
                        "background-color":"red",
                        "line-color":"black"
                    }
                },
                {
                    "selector":".blue",
                    "style":{
                        "background-color":"blue",
                        "line-color":"black"
                    }
                },
                {
                    "selector":".green",
                    "style":{
                        "background-color":"green",
                        "line-color":"black"
                    }
                },
            ]
    )
    return graph


#############################################################################
###################### DISENTANGLING THE MIXTURE ############################
def disentangle_and_evaluate(base_network,do_config,sample_size,table_columns):
    '''
    This function will disentangle the internvetion and then evalaute it
    and also generate the table.
    '''
    #Gettig the samples
    print("Getting the Samples for Disentangling")
    infinite_mix_sample=False
    if sample_size=="infinite":
        mixture_samples=None
        infinite_mix_sample=True
    else:
        mixture_samples=base_network.generate_sample_from_mixture(
                                            do_config=do_config,
                                            sample_size=sample_size)
    # pdb.set_trace()

    #Now lets solve the problem
    #Initializing our Solver
    solver=NonOverlapIntvSolve(base_network=base_network,
                                do_config=do_config,
                                infinite_mix_sample=infinite_mix_sample,
                                mixture_samples=mixture_samples,
                                opt_eps=1e-10,
                                zero_eps=1e-5,
                                insert_eps=0.05)#This is in percentage error
    predicted_configs,x_bars=solver.solve()

    #Getting the evaluation metric
    avg_jaccard_sim,avg_mse,matched_configs=match_and_get_score(do_config,
                                                predicted_configs)
    print("\n\nAverage Jaccard Score:",avg_jaccard_sim)
    print("Average_mse:",avg_mse)

    #Now we have to create a table object
    # table_element=html.Table([
    #     html.Tr([html.Th("S.No.")]+[
    #         html.Th(col) for col in table_columns]),
    #
    #     html.Tbody([
    #         html.Tr([html.Td(idx)]+[html.Td(matched_configs[idx][col])
    #                                 for col in range(len(table_columns))
    #         ]) for idx in range(len(matched_configs))
    #     ])
    # ]),

    #Creating the table using dash table
    table_columns=table_columns
    table_element=dash_table.DataTable(
        id="matched_configs",
        columns=[{"id":col,"name":col} for col in table_columns],
        data=[
            {table_columns[col]:matched_configs[idx][col]
                        for col in range(len(table_columns))
            } for idx in range(len(matched_configs))
        ],
        page_action="none",
        fixed_rows={"headers":True},
        style_table={"height":"600px"},
        style_header={
            "backgroundColor":"rgb(230, 230, 230)",
            "fontWeight":"bold",
            "border":"2px solid black",
        },
        style_data={"border":"1px solid black"},
        row_selectable="single",
    )

    return avg_jaccard_sim,avg_mse,table_element,matched_configs

def plot_evaluation_metrics(jaccard_list,mse_list,sample_sizes):
    '''
    This fucntion will plot the evalaution metrics as a function of sample
    sizes.
    '''
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    x_axis_tickvals=list(range(len(sample_sizes)))

    #Sorting the lsit base on sample size
    inf_flag=1 if "infinite" in sample_sizes else 0
    for sidx,size in enumerate(sample_sizes):
        if size=="infinite":
            sample_sizes[sidx]=1000000000

    comb_list=list(zip(jaccard_list,mse_list,sample_sizes))
    comb_list.sort(key=lambda x:x[-1])
    jaccard_list,mse_list,sample_sizes=zip(*comb_list)
    sample_sizes=list(sample_sizes)
    if inf_flag==1:
        sample_sizes[-1]="infinite"
    # pdb.set_trace()

    #Adding the mse trace
    fig.add_trace(
        go.Scatter(y=mse_list,
                        name="MSE",
                        mode="lines+markers",
                        marker=dict(symbol="square",
                                    size=10),
                    ),
        secondary_y=False,
    )
    #Adding the Jaccard Trace
    fig.add_trace(
        go.Scatter(y=jaccard_list,
                        name="Jaccard Sim",
                        mode="lines+markers",
                        marker=dict(symbol="circle",
                                    size=10),
                ),
        secondary_y=True,
    )

    #Setting the x and y axis titles
    fig.update_xaxes(title_text="Sample Size",ticktext=sample_sizes,
                                                tickvals=x_axis_tickvals,
                                                ticklen=10)
    fig.update_yaxes(title_text="<b>MSE</b>",secondary_y=False)
    fig.update_yaxes(title_text="<b>Similarity</b>",secondary_y=True)

    return fig
