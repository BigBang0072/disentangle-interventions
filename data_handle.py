import numpy as np
from toposort import toposort_flatten
from collections import defaultdict

import pgmpy.readwrite import BIFReader

class BnNetwork():
    '''
    This class will have contol all the functionality related to base graph,
    intervention graph, generating samples or mixture from interventions.
    '''
    #Initializing the global variabels
    base_graph=None         #The actual Bayesian Network without interventions
    nodes=None              #The name of nodes in graph
    card_node=None          #The cardiantlity of the nodes
    edges=None              #The edge list in base graph
    adj_set=None            #Adjacency list of base graph
    topo_i2n=None           #The topological ordering (index to node dict)
    topo_n2i=None           #node to index in topological ordering


    def __init__(self,modelpath):
        #Reading the model from the file
        reader=BIFReader(modelpath)
        #Initializing the base graph
        self._initialize_base_graph(reader)

    def _initialize_base_graph(self,reader):
        '''
        This function will create the base distribution, have a hash of nodes
        name and a numbering.
        '''
        #Getting the base distributions
        base_graph=reader.get_model()
        base_graph.check_model()

        #Getting the names of nodes and its edges
        nodes=reader.get_variables()
        edges=reader.get_edges()
        #Getting the topological order and adjacency list
        adj_set=defaultdict(set)
        for from,to in edges:
            adj_set[from].add(to)
        topo_nodes=toposort_flatten(adj_set)
        topo_i2n={i:node for i,node in enumerate(topo_nodes)}
        topo_n2i={node:i for i,node in enumerate(topo_nodes)}

        #Adding the property to class
        self.base_graph=base_graph
        self.nodes=nodes
        self.card_node=model.get_cardinality()
        self.edges=edges
        self.topo_i2n=topo_i2n
        self.topo_n2i=topo_n2i
        self.adj_set=adj_set

    def generate_intervention_graph(self,node_idx,node_cat):
        '''
        This function will generate the intervention graph at the given
        node number and category according to topological order.

        node_cat is zero indexed
        '''
        #Getting the name of node
        node=self.topo_i2n[node_idx]
        assert node_cat<self.card_node[node],"category index out of bound!!"

        #Copying the model first of all
        do_graph=self.base_graph.copy()
        #Now saving the cpds of the children of current node
        node_cpd=do_graph.get_cpds(node).copy()
        child_old_cpds=[do_graph.get_cpds(child) for child in adj_set[node]]
        child_cpds=[do_graph.get_cpds(child).copy() for child in adj_set[node]]

        #Now we will perform the do operation
        do_graph.remove_node(node)
        #But this has removed all node and children connection. Readd
        for child in adj_set[node]:
            do_graph.add_edge(node,child)
        #Now we will add the cpds of childrens
        do_graph.remove_cpds(child_old_cpds)
        do_graph.add_cpds(child_cpds)

        #Now we have to change the cpd of current node
        node_parents=(node_cpd.variables).remove(node_cpd.variable)
        node_cpd.marginalize(node_parents)
        #Set the probability of intervented category to 1
        node_cpd.values=np.zeros(self.card_node[node])
        node_cpd.values[node_cat]=1.0
        #Add this cpd to graph
        do_graph.add_cpds(node_cpd)

        #Finally testing the model
        do_graph.check_model()
        return do_graph
