import pdb
import numpy as np
np.random.seed(211)
import pandas as pd
from toposort import toposort_flatten
from collections import defaultdict
import multiprocessing

from pgmpy.readwrite import BIFReader
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

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
    states_c2i=None         #dict of dict
    states_i2c=None         #dict of each nodes's category from idx to name

    #Variables for one-hot encoding of samples
    vector_length=None      #Length of one-hot vector
    one2loc=None            #Mapping from one-hot index to actual node and cat
    loc2one=None            #Mapping from node,cat to location in one hot
    data_schema=None        #Empty dataframe to hold the reconverted people

    #For intervention
    orphan_nodes=set()        #Set of nodes which no longer will have parent

    def __init__(self,modelpath):
        #Initializing the base graph
        print("Initializing the base_graph")
        self._initialize_base_graph(modelpath)

        #Initializing the one hot variables
        self._get_one_hot_mapping()

    def _read_model_bif(self,modelpath):
        #Reading the model from the file
        reader=BIFReader(modelpath)

        #Getting the base distributions
        base_graph=reader.get_model()
        base_graph.check_model()

        #Getting the names of nodes and its edges
        nodes=reader.get_variables()
        edges=reader.get_edges()
        #Getting the variables names /state for each nodes
        self.states=reader.get_states()
        self.states_c2i={key:{val:np.int32(idx) for idx,val
                                                in enumerate(kval)}
                            for key,kval in self.states.items()}
        self.states_i2c={key:{np.int32(idx):val for idx,val
                                                in enumerate(kval)}
                            for key,kval in self.states.items()}

        return base_graph,nodes,edges

    def _initialize_base_graph(self,modelpath):
        '''
        This function will create the base distribution, have a hash of nodes
        name and a numbering.
        '''
        if type(modelpath) is str:
            #If we are given the bif file to read from
            base_graph,nodes,edges = self._read_model_bif(modelpath)
        else:
            #If we are directly given an instance of the Bayesian Network
            base_graph = modelpath
            nodes = list(base_graph.nodes())
            edges = list(base_graph.edges())

        #Getting the topological order and adjacency list
        adj_set={node:set() for node in nodes}
        inv_adj_set={node:set() for node in nodes}
        in_degree={node:0 for node in nodes}
        for fr,to in edges:
            adj_set[fr].add(to)
            inv_adj_set[to].add(fr)
            in_degree[to]+=1
        #BEWARE: this function take inverse adj_list
        topo_nodes=toposort_flatten(inv_adj_set)
        topo_i2n={i:node for i,node in enumerate(topo_nodes)}
        topo_n2i={node:i for i,node in enumerate(topo_nodes)}

        #Now we will calcuate the topological level ordering
        topo_level={}
        curr_level=0
        while(len(in_degree)>0):
            # pdb.set_trace()
            #Getting the nodes with zero indegree
            zero_nodes=[node for node,deg_left in in_degree.items()
                                            if deg_left==0]
            topo_level[curr_level]=set(zero_nodes)
            curr_level+=1

            #Now we will reduce the indegree of connection form these nodes
            for node in zero_nodes:
                #Removing these nodes from in_degree list
                del in_degree[node]
                #Now reducing the degrre of it's to conenction
                for to_node in adj_set[node]:
                    in_degree[to_node]-=1
        #Now we are done with the topological levels
        self.topo_level=topo_level

        #Adding the property to class
        self.base_graph=base_graph
        self.nodes=nodes
        self.card_node=base_graph.get_cardinality()
        self.edges=edges
        self.topo_i2n=topo_i2n
        self.topo_n2i=topo_n2i
        self.adj_set=adj_set

    #Generating the intervention Graph
    def do(self,node_ids,node_cats):
        '''
        Perform size(node_ids)-order internveiton on the graph.
        node_ids    : list of node indexes where to perform internvetions
        node_cats   : the category which we have to intervene at those nodes.
        '''
        #Copying the model first of all (new intervened dist will be this)
        do_graph=self.base_graph.copy()

        #Now one by one we will perform all the necessary intervnetions
        orphan_nodes=set()
        for node_id,node_cat in zip(node_ids,node_cats):
            self._single_do(do_graph,node_id,node_cat,orphan_nodes)

        return do_graph

    def _single_do(self,do_graph,node_idx,node_cat,orphan_nodes):
        '''
        This function will generate the intervention graph at the given
        node number and category according to topological order. This is limited
        to perfoming a single do to the graph.

        node_cat is zero indexed
        '''
        # print("Creating intervention Graph")
        #Getting the name of node
        node=self.topo_i2n[node_idx]
        assert node_cat<self.card_node[node],"category index out of bound!!"

        #Now saving the cpds of the children of current node
        child_old_cpds=[do_graph.get_cpds(child).copy() for child in self.adj_set[node]]
        # pdb.set_trace()

        #Now we will perform the do operation
        do_graph.remove_node(node)
        #We have not removed the edge from parents to current node in adj_set
        #If we ensure not to edge later from any parent to it later then fine
        orphan_nodes.add(node)

        #But this has removed all node and children connection. Readd
        do_graph.add_node(node)
        for child in self.adj_set[node]:
            if child not in orphan_nodes:
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
                            np.zeros((self.card_node[node],1)))
        node_cpd.values[node_cat]=1.0
        #Add this cpd to graph
        do_graph.add_cpds(node_cpd)
        # pdb.set_trace()

        #Finally testing the model
        do_graph.check_model()

    #Sampling functions
    def generate_sample_from_mixture(self,do_config,sample_size,savepath=None):
        '''
        Generating the sample for the mixture distribution given by do_config.
        do_config   : list of [ [node_ids,node_cats,pi], ... ]

        node_ids could represent multiple interventions
        '''
        all_samples=[]
        #Now we will sample from the base distribution
        _,_,pis=zip(*do_config)
        phi=1-sum(pis)
        assert phi>=(-1e-10),"Illegal mixture Distribtuion"
        if int(sample_size*phi)>0:
            sampler=BayesianModelSampling(self.base_graph)
            samples=sampler.forward_sample(size=int(sample_size*phi),
                                            return_type="dataframe")
            all_samples.append(samples)
        # pdb.set_trace()

        #One by one we will generate the mixture graph and corresponding sample
        for node_ids,node_cats,pi in do_config:
            #Getting the intervention distributions
            do_graph=self.do(node_ids,node_cats)

            #Now we are ready to sample from our distribution
            sampler=BayesianModelSampling(do_graph)
            #Only sample the amount required
            num_sample=int(sample_size*pi)
            if num_sample>0:
                samples=sampler.forward_sample(size=num_sample,
                                        return_type="dataframe")
                all_samples.append(samples)

        # pdb.set_trace()
        #Now we will merge all the samples in one and shuffle it
        all_samples=pd.concat(all_samples)
        all_samples=all_samples.sample(frac=1.0).reset_index(drop=True)

        #We will save the schema for the later reconversions
        self.data_schema=pd.DataFrame(columns=all_samples.columns)

        #Saving the dataframe (for reproducabilty)
        if savepath!=None:
            filepath="{}mixture_{}_{}.csv".format(savepath,num_sample,
                                                str(do_config))
            all_samples.to_csv(filepath,index=False)

        return all_samples

    def encode_sample_one_hot(self,samples_df):
        '''
        This function will take tha sample dataframe and encode them in one
        hot way, merging all nodes into a single vector.
        '''
        #Now we will create the input one by one for each sample
        samples_one_hot=[]
        for sidx in range(samples_df.shape[0]):
            #Getting the sample and empyty vector
            sample=samples_df.iloc[sidx]
            vector=np.zeros((1,self.vector_length),dtype=np.float32)
            for nidx in range(len(self.topo_i2n)):
                node=self.topo_i2n[nidx]
                cat=sample[node]
                #Assigning the location to be hot
                vec_pos=self.loc2one[(node,cat)]
                vector[0,vec_pos]=1
            #Adding the one hot vector to list
            samples_one_hot.append(vector)
        #Now we will convert them to one array
        samples_one_hot=np.concatenate(samples_one_hot,axis=0)
        # pdb.set_trace()
        return samples_one_hot

    def _get_one_hot_mapping(self):
        '''
        Generate the location map from category number to one hot and
        vice-versa.
        '''
        vector_length=0
        one2loc={}
        loc2one={}
        for nidx in range(len(self.topo_i2n)):
            node=self.topo_i2n[nidx]
            card=self.card_node[node]
            #Now we will hash the nodes
            for cidx in range(card):
                loc2one[(node,cidx)]=vector_length
                one2loc[vector_length]=(node,cidx)
                vector_length+=1

        self.vector_length=vector_length
        self.one2loc=one2loc
        self.loc2one=loc2one

    #Probability of a sample (for loss function)
    def get_sample_probability(self,interv_locs,input_samples):
        '''
        Given a set of sample, this function will calculate the overall sample,
        probability and then reaturn it back to tensorflow for likliehood
        calculation nad backprop.

        We return the sample probability for each of the intervention component
        not bother about summing up the overall probability which will be done
        inside the decoder or full model.

        Output shape: [num_sample,num_intervention_loc(=sparsity)]
        '''
        #First of all generating all the required intervention graph
        interv_graphs=[]
        for loc in interv_locs:
            node_ids,cat_ids=loc
            #TODO: Right now we will deal with single inerventions
            #assert len(node_ids)==1,"Multiple simultaneous intervention"
            if node_ids[0]==len(self.topo_i2n) and len(node_ids)==1:
                interv_graphs.append(self.base_graph)
            else:
                interv_graphs.append(self.do(node_ids,cat_ids))

        #Now we are ready to get sample prob for each interventions
        input_samples=input_samples.numpy()
        #reconvert the input samples from one hot to actual
        # input_samples=self.decode_sample_one_hot(input_samples)

        #For each sample, generate the graph and calculate the probability
        def worker_kernel(child_conn,child_samples,
                            interv_graphs,network_parameters):
            #First of all decode the sample from one-hot representation
            child_samples=decode_sample_one_hot(child_samples,
                                                network_parameters)

            #Calculating the probability
            all_sample_prob=[]
            for idx in range(child_samples.shape[0]):
                sample=child_samples.iloc[idx]
                sample_prob=[]
                for graph in interv_graphs:
                    prob=get_graph_sample_probability(graph,
                                                    sample,network_parameters)
                    sample_prob.append(prob)
                all_sample_prob.append(sample_prob)
            #Now converting this to a numpy array
            all_sample_prob=np.array(all_sample_prob)
            #Sending all the samples back to the parent
            child_conn.send(all_sample_prob)
            child_conn.close()

        #Now we will create multiple workers to parallelize
        njobs=multiprocessing.cpu_count()/2
        num_per_job=int(np.ceil((input_samples.shape[0]*1.0)/njobs))
        process_pipe_list=[]
        process_list=[]
        for jidx in range(njobs):
            #Slicing our big input for our job
            print("Worker:{} manipulating from:{} to:{}".format(jidx,
                                                    jidx*num_per_job,
                                                    (jidx+1)*num_per_job))
            child_samples=input_samples[jidx*num_per_job:(jidx+1)*num_per_job]

            #Now we will first create a pipe to receive results
            parent_conn,child_conn=multiprocessing.Pipe()
            process_pipe_list.append(parent_conn)

            #Starting the child process
            network_parameters={}
            network_parameters["topo_i2n"]=self.topo_i2n
            network_parameters["card_node"]=self.card_node
            network_parameters["data_schema"]=self.data_schema.copy()
            network_parameters["vector_length"]=self.vector_length
            network_parameters["one2loc"]=self.one2loc

            p=multiprocessing.Process(target=worker_kernel,
                                        args=(child_conn,
                                            child_samples,
                                            interv_graphs,
                                            network_parameters))
            p.start()
            process_list.append(p)

        #Now we will receive the results for all the child (join)
        child_probs=[parent_conn.recv() for parent_conn in process_pipe_list]
        #Stopping all the process
        [p.join() for p in process_list]

        #Now we will return the final concatenated result
        all_sample_prob=np.concatenate(child_probs,axis=0)
        print("merged prob size:",all_sample_prob.shape)
        return all_sample_prob

#Helper function to be used by parallel worker to compute probability of
#individual sample
def get_graph_sample_probability(graph,sample,network_parameters,marginal=False):
    '''
    This function will calcuate the probability of a sample in a graph,
    which will be later used to calcuate the overall mixture probability.

    graph   : the graph on which we have to calculate the sample probability
    sample  : the sample array in form of dictionary or numpy recarray

    Since we cant vectorize this function, cuz every sample will generate,
    a separate distribution and in that distribution we have to calculate
    the probability. We will see later how to vectorize
    '''
    #Getting the network parametrs
    topo_i2n=network_parameters["topo_i2n"]
    card_node=network_parameters["card_node"]

    def _get_columns_index(nodes_idx,nodes_card):
        '''
        This will convert the index to row major number for column of cpd to
        access.
        '''
        assert len(nodes_idx)==len(nodes_card)
        multiplier=nodes_card.copy()
        multiplier[-1]=1  #we dont have to offset last index
        for tidx in range(len(nodes_card)-2,-1,-1):
            multiplier[tidx]=nodes_card[tidx+1]*multiplier[tidx+1]
        #Now we are ready with the offset multiplier
        ridx=0
        for tidx in range(len(nodes_idx)):
            assert nodes_idx[tidx]<nodes_card[tidx]
            ridx+=nodes_idx[tidx]*multiplier[tidx]
        return ridx

    #Now we will start in the topological order to get prob
    marginal_length=len(topo_i2n)
    if marginal==True:
        marginal_length=len(sample)

    #Initialing the porbabilty
    overall_prob=1.0
    for nidx in range(marginal_length):
        #Getting the information of node
        node=topo_i2n[nidx]
        node_cpd=graph.get_cpds(node)
        #Getting the row in which to look
        row_idx=node_val=sample[node]

        #Now we have to get the columns number
        pnodes=(node_cpd.variables.copy())
        pnodes.remove(node_cpd.variable)
        col_idx=None
        if len(pnodes)!=0:
            pnodes_card=[card_node[pn] for pn in pnodes]
            pnodes_vals=[sample[pn] for pn in pnodes]
            col_idx=_get_columns_index(pnodes_vals,pnodes_card)
            #Just to be safe we will reorder for now (Comment later for performance)
            node_cpd.reorder_parents(pnodes)
        else:
            col_idx=0

        #Now we will calculate the probabilityof the node given its parents
        prob_node_given_parents=node_cpd.get_values()[row_idx,col_idx]
        #Updating the overall probability
        overall_prob=overall_prob*prob_node_given_parents
    return overall_prob

def decode_sample_one_hot(samples_one_hot,network_parameters):
    '''
    This function will reconvert the samples to a dataframe as before,
    just like papaji didnt get to know what mishap has happened.
    '''
    #Getting the network parameters
    df=network_parameters["data_schema"]
    vector_length=network_parameters["vector_length"]
    one2loc=network_parameters["one2loc"]

    #Now we are ready to reconvert peeps
    all_row_entry=[]
    for sidx in range(samples_one_hot.shape[0]):
        sample=samples_one_hot[sidx,:]
        assert sample.shape[0]==vector_length

        #Now we will decode this example
        row_entry={}
        for tidx in range(vector_length):
            val=sample[tidx]
            if val==0:
                continue
            else:
                node,cat=one2loc[tidx]
                row_entry[node]=cat

        #Adding new row to the dataframe
        # pdb.set_trace()
        all_row_entry.append(pd.DataFrame(row_entry,index=[0]))

    #Concatenating all the rows into one big dataframe
    df=pd.concat(all_row_entry,ignore_index=True)
    return df

##########################################################################
##### Data handling for Flipkar dataset
def load_flipkart_mixture_sample(filepath,base_network):
    '''
    This will be used in the inference case when the the sample is taken
    in real world data. Then we have to rename the names of Category
    to the number in the sample dataset
    '''
    print("Reading the Real World Mixture Sample")
    df=pd.read_csv(filepath)
    #Subset only those variable which are in our bn network
    df=df[base_network.nodes]
    #COnverting everything into string
    def convert_to_str(element):
        #Hack to remove the error made by bnlearn by removing  (,) with _
        element=str(element).replace("(","_").replace(")","_")
        return element

    df=df.applymap(np.vectorize(convert_to_str))

    print("Converting the Category to Index")
    # pdb.set_trace()
    #Now we will define the maping function of the dataframe
    def map_cat2index(columns):
        # print("columns.name:",columns.name)
        return columns.map(base_network.states_c2i[columns.name])

    #Applying the mapping fucntion to every columns
    df_map=df.apply(map_cat2index,axis=0)
    assert df.shape==df_map.shape

    #Now we will remove the rows which which are null
    df_map=df_map[df_map.isnull().any(axis=1)==False]
    #Find out why it's getting converted to float (to accomodate NA mostly)
    df_map.astype(np.int32)
    print("Total Rows Left:",df.shape[0],df_map.shape[0])
    # pdb.set_trace()

    return df_map

if __name__=="__main__":
    #Testing the base model and intervention
    graph_name="asia"
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    network=BnNetwork(modelpath)
    # pdb.set_trace()

    #Creating the arguments for generator
    from graph_generator import GraphGenerator
    generator_args={}
    generator_args["scale_alpha"]=5
    #Starting the generation process
    graphGenerator = GraphGenerator(generator_args)
    modelpath = graphGenerator.generate_bayesian_network(num_nodes=10,
                                            node_card=3,
                                            num_edges=30,
                                            graph_type="SF")
    network=BnNetwork(modelpath)
    # pdb.set_trace()

    #Testing internvention
    do_graph=network.do([3,4,1,5],[1,1,0,0])
    # pdb.set_trace()

    #Testing the sampler for mixture
    sample_size=10
    savepath="dataset/{}/".format(graph_name)
    do_config=[
                [[2,3],[1,0],0.5],
                [[1,6],[0,1],0.3]
            ]
    samples=network.generate_sample_from_mixture(do_config,sample_size,
                                                    savepath)
    pdb.set_trace()

    #Testing the probability calculation function
    # prob=network.get_graph_sample_probability(network.base_graph,
    #                                             samples.iloc[0])
    # pdb.set_trace()

    #Testing the encoding and decoding function
    # sample_one_hot=network.encode_sample_one_hot(samples)
    # sample_prime=network.decode_sample_one_hot(sample_one_hot)
    # sample_prime=sample_prime[samples.columns]
    # assert samples.equals(sample_prime),"Encoded and Decoded data not same"
    # pdb.set_trace()
