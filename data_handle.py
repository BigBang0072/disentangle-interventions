import pdb
import numpy as np
from toposort import toposort_flatten
from collections import defaultdict

from pgmpy.readwrite import BIFReader
from pgmpy.factors.discrete import TabularCPD

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
        print("Initializing the base_graph")
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
        inv_adj_set=defaultdict(set)
        for fr,to in edges:
            adj_set[fr].add(to)
            inv_adj_set[to].add(fr)
        #BEWARE: this function take inverse adj_list
        topo_nodes=toposort_flatten(inv_adj_set)
        topo_i2n={i:node for i,node in enumerate(topo_nodes)}
        topo_n2i={node:i for i,node in enumerate(topo_nodes)}

        #Adding the property to class
        self.base_graph=base_graph
        self.nodes=nodes
        self.card_node=base_graph.get_cardinality()
        self.edges=edges
        self.topo_i2n=topo_i2n
        self.topo_n2i=topo_n2i
        self.adj_set=adj_set

    def do(self,node_ids,node_cats):
        '''
        Perform size(node_ids)-order internveiton on the graph.
        node_ids    : list of node indexes where to perform internvetions
        node_cats   : the category which we have to intervene at those nodes.
        '''
        #Copying the model first of all (new intervened dist will be this)
        do_graph=self.base_graph.copy()

        #Now one by one we will perform all the necessary intervnetions
        for node_id,node_cat in zip(node_ids,node_cats):
            self._single_do(do_graph,node_id,node_cat)

        return do_graph

    def _single_do(self,do_graph,node_idx,node_cat):
        '''
        This function will generate the intervention graph at the given
        node number and category according to topological order. This is limited
        to perfoming a single do to the graph.

        node_cat is zero indexed
        '''
        print("Creating intervention Graph")
        #Getting the name of node
        node=self.topo_i2n[node_idx]
        assert node_cat<self.card_node[node],"category index out of bound!!"

        #Now saving the cpds of the children of current node
        child_old_cpds=[do_graph.get_cpds(child).copy() for child in self.adj_set[node]]
        # pdb.set_trace()

        #Now we will perform the do operation
        do_graph.remove_node(node)
        #But this has removed all node and children connection. Readd
        do_graph.add_node(node)
        for child in self.adj_set[node]:
            do_graph.add_edge(node,child)
        #Now we will add the cpds of childrens
        child_cur_cpds=[do_graph.get_cpds(child) for child in self.adj_set[node]]
        for cur_cpds,old_cpds in zip(child_cur_cpds,child_old_cpds):
            do_graph.remove_cpds(cur_cpds)
            do_graph.add_cpds(old_cpds)

        #Now we have to change the cpd of current node
        #Set the probability of intervented category to 1
        node_cpd=TabularCPD(node,
                            self.card_node[node],
                            np.zeros((1,self.card_node[node])))
        node_cpd.values[node_cat]=1.0
        #Add this cpd to graph
        do_graph.add_cpds(node_cpd)
        # pdb.set_trace()

        #Finally testing the model
        do_graph.check_model()

if __name__=="__main__":
    #Testing the base model and intervention
    modelpath="dataset/asia.bif"
    network=BnNetwork(modelpath)

    #Testing internvention
    # pdb.set_trace()
    do_graph=network.do([2,3,7],[1,0,1])
    pdb.set_trace()
