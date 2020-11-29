import igraph as ig
import networkx as nx
import numpy as np
import pdb
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete.CPD import TabularCPD
from scipy.stats import dirichlet

class GraphGenerator():
    '''
    This class will generate the random graph to be used as input for our
    mixture generation experiment. This class is motivated from
    DAG with NoTears paper way of generating random DAG.
    '''
    def __init__(self,args):
        '''
        args:
            scale_alpha     : >1 will reduce the sampling from edge of simplex
        '''
        self.scale_alpha = args["scale_alpha"]

    ######################### ADJ-MAT GEN ##########################
    def sample_graph(self,num_nodes,num_edges,graph_type):
        '''
        We will sample a DAG from from the Erdos-Reneyi Model with given
        number of node and edge.
        '''
        if graph_type=="ER":
            #Erdos-Renayi
            G_und = ig.Graph.Erdos_Renyi(n=num_nodes,m=num_edges)
            B_und = self._graph_to_adj_matrix(G_und)
            B = self._dagify_randomly(B_und)
        elif graph_type=="SF":
            #Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=num_nodes,
                                    m=int(round(num_edges/num_nodes)),
                                    directed=True)
            B = self._graph_to_adj_matrix(G)
        else:
            raise ValueError("unknown graph type")

        #Now we have a adj mat of DAG, just permute it
        B_perm = self._permute_adj_matrix(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag(),"AdjMat not DAG"

        return B_perm

    def _permute_adj_matrix(self,adj_mat):
        P = np.random.permutation(np.eye(adj_mat.shape[0]))
        return P.T @ adj_mat @ P

    def _graph_to_adj_matrix(self,G):
        return np.array(G.get_adjacency().data)

    def _dagify_randomly(self,adj_mat):
        return np.tril(self._permute_adj_matrix(adj_mat),k=-1)

    def get_graph_metrics(self,adj_mat):
        num_edges=np.sum(adj_mat)
        exp_indegree=np.mean(np.sum(adj_mat,axis=1))

        return (num_edges,exp_indegree)

    ####################### BAYESIAN NETWORK CREATION ##############
    def generate_bayesian_network(self,num_nodes,node_card,num_edges,graph_type):
        '''
        This function will take the wireframe of the graph i.e adj_mat
        and generate a full fledged bayesian network to be consumed by the
        data handler via a bif file of generated graph.
        '''
        #Getting the wireframe for the graph
        adj_mat = self.sample_graph(num_nodes,num_edges,graph_type)

        #Creating the bayesian network based on the adj matrix
        network = self._create_network_with_edges(adj_mat)
        self._add_cpds_to_network(network,node_card,adj_mat)
        network.check_model()
        pdb.set_trace()

        #Now its time to save the network as bif file and return path


    def _create_network_with_edges(self,adj_mat):
        #Creating the edge list
        edge_list=[]
        for fro in range(adj_mat.shape[1]):
            for to in range(adj_mat.shape[0]):
                if(adj_mat[to][fro]==1):
                    edge_list.append((fro,to))

        #Creating the base network
        network = BayesianModel(edge_list)
        return network

    def _add_cpds_to_network(self,network,node_card,adj_mat):
        '''
        Here we will create CPD for each of the node based on the criteria:
        1. Sample each Conditional distribution p(node| pa_config)
            1.1 Using Dirchilet Prior to sample in probability simplex
        2. Stack these one-liner together to get full CPD of a node

        The cardinality of each node is assumed to be same in this experiment.
        '''
        for node in network.nodes():
            parents = [pidx for pidx in range(adj_mat.shape[1])
                            if adj_mat[node][pidx]==1]
            #Getting the random CPD
            cpd_arr = self._generate_random_cpd(node_card,len(parents))
            # pdb.set_trace()
            node_cpd = TabularCPD(node,node_card,cpd_arr,
                                    evidence=parents,
                                    evidence_card=[node_card]*len(parents))

            network.add_cpds(node_cpd)

    def _generate_random_cpd(self,node_card,num_parents):
        '''
        '''
        num_pa_config = node_card**num_parents
        dirch_alphas = np.ones(node_card)*self.scale_alpha

        #Sampling the CPD for each of the configuration of prents
        cpd = dirichlet.rvs(size=num_pa_config, alpha=dirch_alphas).T

        return cpd


if __name__=="__main__":
    #Now lets test the DAG creation with expected number of edges etc.
    graph_metrics=[]
    num_graphs=1000

    #Creating the arguments for generator
    args={}
    args["scale_alpha"]=5

    #Starting the generation process
    graphGenerator = GraphGenerator(args)
    graphGenerator.generate_bayesian_network(num_nodes=10,
                                            node_card=3,
                                            num_edges=30,
                                            graph_type="SF",
                                            )
    for idx in range(num_graphs):
        #Generating the graph
        graph_type = "SF" #if idx%2==0 else "ER"
        adj_mat = graphGenerator.sample_graph(num_nodes=10,
                                    num_edges=30,
                                    graph_type=graph_type)
        #Getting the graph metrics
        graph_metrics.append(graphGenerator.get_graph_metrics(adj_mat))
        #pdb.set_trace()

    #Plotting the graph metrics
    import matplotlib.pyplot as plt
    edge_dist,indegree_dist = zip(*graph_metrics)
    plt.hist(edge_dist,bins=20,edgecolor='k')
    plt.show()
    plt.close()
    plt.hist(indegree_dist,bins=20,edgecolor='k')
    plt.show()
