import igraph as ig
import networkx as nx
import numpy as np
import pdb

class GraphGenerator():
    '''
    This class will generate the random graph to be used as input for our
    mixture generation experiment. This class is motivated from
    DAG with NoTears paper way of generating random DAG.
    '''
    def __init__(self,args=None):
        '''
        '''
        self.args = args

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


if __name__=="__main__":
    #Now lets test the DAG creation with expected number of edges etc.
    graph_metrics=[]
    num_graphs=1000

    #Starting the generation process
    graphGenerator = GraphGenerator()
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
