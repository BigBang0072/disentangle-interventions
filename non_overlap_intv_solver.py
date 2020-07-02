import numpy as np
from scipy.optimize import minimize
import pdb

from data_handle import BnNetwork,get_graph_sample_probability


class DistributionHandler():
    '''
    This class will handle all the distribution calculations, like Calculating
    the proability of a point or marginal, giving the intervention probability,
    giving the mixture probability, estimate or calculate the infinite-limit
    estimate of the mixture probability i.e from actual mixture config.
    '''
    #Class Attributes
    base_network=None
    do_config=None

    true_comp_pis=None      #pi's corresponding to actual mixture config
    true_comp_graphs=None   #interv component corresponding to actual mixture

    def __init__(self,base_network,do_config):
        self.base_network=base_network
        self.do_config=do_config

        #Initialize the true mixture components (will simulate infinite data)
        if do_config!=None:
            self._get_true_mixture_graph()
        #

    def _get_true_mixture_graph(self,):
        '''
        This function will generate all the individual do_graphs to represent
        the true mixture graph, if we now it.
        '''
        #Keep track of all the actual component
        true_comp_pis=[]
        true_comp_graphs=[]

        #Now we will create the individual graph one by one
        pi_sum=0.0
        for node_ids,cat_ids,pi in self.do_config:
            pi_sum+=pi
            #Getting the intervented distribution
            do_graph=self.base_network.do(node_ids,cat_ids)

            #Keeping them safe for later use
            true_comp_graphs.append(do_graph)
            true_comp_pis.append(pi)

        #If the base distribution also contributes
        if 1-pi_sum>0:
            true_comp_pis.append(1-pi_sum)
            true_comp_graphs.append(self.base_network.base_graph)

        self.true_comp_pis=true_comp_pis
        self.true_comp_graphs=true_comp_graphs

    def _get_true_mixture_probability(self,sample):
        '''
        This function will calculate the true mixture probability of a sample,
        by running it on the actual components and giving the overall mixture
        probbaility. This could be used to simulate the infinite data setting
        in the mixture.
        '''
        #Config of the network
        network_parameters={}
        network_parameters["topo_i2n"]=self.base_network.topo_i2n
        network_parameters["card_node"]=self.base_network.card_node

        sample_prob=0.0
        #Now we will iterate over all the individual component
        for comp_pi,comp_graph in zip(self.true_comp_pis,self.true_comp_graphs):
            #Getting the sample probability in the interv graph
            comp_prob=get_graph_sample_probability(comp_graph,sample,
                                                    network_parameters,
                                                    marginal=True)
            #Adding to the probability in the mixture
            sample_prob+=comp_pi*comp_prob

        return sample_prob

    def get_mixture_probability(self,sample,infinte_sample=False):
        '''
        This function will give us the mixture probability of a point either
        in the infinite sample limit or by using the estimated CPDs from
        the samples
        '''
        if infinte_sample==True:
            return self._get_true_mixture_probability(sample)
        else:
            raise NotImplementedError

    def get_intervention_probability(self,sample,eval_do):
        '''
        This will evaluate the marginal probability of a point either on the
        base distribuion or on the do distribution as required.

        eval_do     : None for base dist else will generate a new do graph
                        and evalate the sampple on that.
        '''
        #Config of the network
        network_parameters={}
        network_parameters["topo_i2n"]=self.base_network.topo_i2n
        network_parameters["card_node"]=self.base_network.card_node

        #Evaluating the sample on the approporate graph
        if eval_do==None:
            #This means we have to evaluate on base graph
            base_graph=self.base_network.base_graph

            prob=get_graph_sample_probability(base_graph,sample,
                                                network_parameters,
                                                marginal=True)
        else:
            #First of all we have to get the graph for this eval_do config
            node_ids,cat_ids=eval_do
            do_graph=self.base_network.do(node_ids,cat_ids)

            prob=get_graph_sample_probability(do_graph,sample,
                                                network_parameters,
                                                marginal=True)

        return prob

class NonOverlapIntvSolve():
    '''
    This class will be responsible for solving the non-overlapping intervention
    problem.
    '''
    base_network=None       #The BnNetwork class instance of current porblem
    do_config=None          #The configuration of the true mixture

    def __init__(self,base_network,do_config,infinte_mix_sample,opt_eps):
        self.base_network=base_network
        self.do_config=do_config
        self.infinte_mix_sample=infinte_mix_sample
        self.opt_eps=opt_eps

        #Creating the distibution handler
        self.dist_handler=DistributionHandler(base_network,do_config)
        self.solve()

    def solve(self,):
        '''
        This function will get the estimate of all the mixing coefficient
        along with mixing coefficient.
        '''
        #initialize the x_bar to be used for eliminating the variables
        x_bars={
                "all":{},
                }
        #Initializing the dictionary to hold the component configuration
        comp_dict={}

        #Lets start iterating over all the variables
        for nidx in range(len(self.base_network.topo_i2n)):
            print("Starting Step-1 for Node:{}".format(nidx))
            #Step 1: Get the category which appear as new components
            self._get_new_components(comp_dict,x_bars,nidx)


    def _get_new_components(self,comp_dict,x_bars,nidx):
        '''
        This function will give us the category of this current node, nidx
        which will give us a new compoenent instead of getting assimilated
        inside already existing ones.
        '''
        #First of all we will have to generate the system of equations
        node_id=self.base_network.topo_i2n[nidx]
        num_cats=self.base_network.card_node[node_id]
        #Initializing the system matrix
        A=np.zeros((num_cats,num_cats),dtype=np.float32)
        b=np.zeros((num_cats,),dtype=np.float32)

        for cidx in range(num_cats):
            #Get the marginal-point to evaluate on
            x_eval=x_bars["all"].copy()
            x_eval[node_id]=cidx

            #Get the mixture probability on this point
            p_mix=self.dist_handler.get_mixture_probability(x_eval,
                                                    self.infinte_mix_sample)
            #Get the base dist prob on this eval point
            p_base=self.dist_handler.get_intervention_probability(x_eval,
                                                    eval_do=None)
            #Get the porb dist when the interv matches
            x_left=x_bars["all"].copy()
            p_left=self.dist_handler.get_intervention_probability(x_left,
                                                    eval_do=None)
            print("pmix:{}\t pbase:{}\t pleft:{}".format(p_mix,p_base,p_left))

            #Now we will fill up the rows of the system matix
            A[cidx,:]=(-1*p_base)
            A[cidx,cidx]+=p_left

            #Now we will fill the entry in the b-vector
            old_pis=[pi for nodes,cats,pi in comp_dict.values()]
            b[cidx]=p_mix-(1-sum(old_pis))*p_base

        #Now we are ready with the system of equation, we have to solve it
        #Setting up the optimization objective
        def optimization_func(x,):
            residual=np.sum((np.matmul(A,x)-b)**2)  #TODO
            print("residual:",residual)
            return residual

        #Setting up the constraints
        bounds=[(0.0,1.0)]*num_cats
        sum_constraints=[{"type":"ineq",
                        "fun":lambda x:1-np.sum(x)-sum(old_pis)}]
        zero_constraints=[{"type":"ineq",
                        "fun":lambda x: self.opt_eps-np.min(x)}]

        #Making the initial guess : #TODO
        pi_0=np.zeros((num_cats,),dtype=np.float32)+[0.2,0.99]
        opt_result=minimize(optimization_func,pi_0,
                            method="SLSQP",
                            bounds=bounds,
                            constraints=zero_constraints+sum_constraints)

        #Now we will have to prune the least values guy as impossible node
        print(opt_result)
        pdb.set_trace()


if __name__=="__main__":
    #Initializing the graph
    graph_name="asia"
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    base_network=BnNetwork(modelpath)

    #Creating artificial intervention
    do_config=[
                ((0,),(1,),0.8),
            ]

    #Initializing our Solver
    solver=NonOverlapIntvSolve(base_network=base_network,
                                do_config=do_config,
                                infinte_mix_sample=True,
                                opt_eps=1e-10)
