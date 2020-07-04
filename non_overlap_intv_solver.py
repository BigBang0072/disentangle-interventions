import numpy as np
from scipy.optimize import minimize
import pdb
from pprint import pprint

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

    def __init__(self,base_network,do_config,infinte_mix_sample,
                    opt_eps,zero_eps,insert_eps):
        self.base_network=base_network
        self.do_config=do_config
        self.infinte_mix_sample=infinte_mix_sample
        self.opt_eps=opt_eps            #Threshold for zeroing one cat in optim
        self.zero_eps=zero_eps          #Threshold for new comp creation
        self.insert_eps=insert_eps      #threshold in error while insert to new

        #Creating the distibution handler
        self.dist_handler=DistributionHandler(base_network,do_config)
        self.solve()

    def solve(self,):
        '''
        This function will get the estimate of all the mixing coefficient
        along with mixing coefficient.

        Important Running Variables:
        x_bars      : dict containing all the nodes and its left out cats
                        upto that point of interation.
                        {
                        node_name : node_category_blacklisted
                        }

        comp_dict   : list of all the component
                        {
                        comp_name: [[nodes],[cats],pi]
                        }
        '''
        #initialize the x_bar to be used for eliminating the variables
        x_bars={}
        #Initializing the dictionary to hold the component configuration
        comp_dict={}

        #Lets start iterating over all the variables
        for nidx in range(len(self.base_network.topo_i2n)):
            print("Starting Step-1 for Node:{}".format(nidx))
            #Step 1: Get the category which appear as new components
            zero_cat_list,new_comp_dict=self._insert_as_new_components(
                                                comp_dict,x_bars,nidx)
            # pdb.set_trace()
            #Only insert the remaining category except 1.
            print("Starting Step-2 for Node:{}".format(nidx))
            left_cidx=self._insert_in_old_component(
                        comp_dict,new_comp_dict,x_bars,nidx,zero_cat_list)

            #Now we have to update our component list
            comp_dict.update(new_comp_dict)
            pprint(comp_dict)
            pprint(x_bars)

        return comp_dict,x_bars


    def _insert_as_new_components(self,comp_dict,x_bars,nidx):
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
            x_eval=x_bars.copy()
            x_eval[node_id]=cidx

            #Get the mixture probability on this point
            p_mix=self.dist_handler.get_mixture_probability(x_eval,
                                                    self.infinte_mix_sample)
            #Get the base dist prob on this eval point
            p_base=self.dist_handler.get_intervention_probability(x_eval,
                                                    eval_do=None)
            #Get the porb dist when the interv matches
            x_left=x_bars.copy()
            p_left=self.dist_handler.get_intervention_probability(x_left,
                                                    eval_do=None)
            print("pmix:{}\t pbase:{}\t pleft:{}".format(
                                              p_mix,p_base,p_left))

            #Now we will fill up the rows of the system matix
            A[cidx,:]=(-1*p_base)
            A[cidx,cidx]+=p_left

            #Now we will fill the entry in the b-vector
            old_pis=[pi for nodes,cats,pi in comp_dict.values()]
            b[cidx]=p_mix-(1-sum(old_pis))*p_base

        #Now we are ready with the system of equation, we have to solve it
        #Setting up the optimization objective
        def optimization_func(x,):
            residual=np.sum((np.matmul(A,x)-b)**2)*1000  #TODO
            print("residual:",residual)
            return residual

        #Setting up the constraints
        bounds=[(0.0,1.0)]*num_cats
        sum_constraints=[{"type":"ineq",
                        "fun":lambda x:1-np.sum(x)-sum(old_pis)}]
        zero_constraints=[{"type":"ineq",
                        "fun":lambda x: self.opt_eps-np.min(x)}]

        #Making the initial guess : #TODO
        pi_0=np.zeros((num_cats,),dtype=np.float32)
        opt_result=minimize(optimization_func,pi_0,
                            method="SLSQP",
                            bounds=bounds,
                            constraints=zero_constraints+sum_constraints)

        #Now we will have to prune the least values guy as impossible node
        print(opt_result)
        # pdb.set_trace()

        #Now we will create the new component for the guys with sig. pi
        zero_cat_list=[]
        new_comp_dict={}
        for cidx in range(num_cats):
            cat_pi=opt_result.x[cidx]
            #Now if this pi is significant then we will create new comp
            if cat_pi>self.zero_eps:
                new_comp=[[nidx,],[cidx,],cat_pi]
                new_comp_dict[
                    "c-{}".format(len(comp_dict)+len(new_comp_dict))]=new_comp
            else:
                #Categories which will be the candidate for insertion in old
                zero_cat_list.append(cidx)

        # pdb.set_trace()
        return zero_cat_list,new_comp_dict

    def _insert_in_old_component(self,comp_dict,new_comp_dict,x_bars,nidx,zero_cat_list):
        '''
        This function will try to insert the left over categories of this node
        into already existing components. Here we will
        1. see the compatibility of the category in all the component
        2. Assign the cat to old component only when they pass:
            2.1 |p_mix-p_test| <= eps-3
            2.2 And minimal among available candidate for insertion
        '''
        node_id=self.base_network.topo_i2n[nidx]
        num_cats=self.base_network.card_node[node_id]

        #Getting the pi's alrady discovered
        old_pis=[pi for _,_,pi in comp_dict.values()]
        new_pis=[pi for _,_,pi in new_comp_dict.values()]

        def get_x_rest(nidx,cname,comp_dict,x_bars):
            x_rest={}
            #Getting the nodes present in the current component
            curr_nodes,curr_cats,_=comp_dict[cname]

            #Iterating over nodes less than current
            for tnidx in range(0,nidx):
                tnidx_name=self.base_network.topo_i2n[tnidx]

                #The the idx is  in current component then use as its is
                if tnidx in curr_nodes:
                    cidx=[curr_cats[idx] for idx in range(len(curr_nodes))
                                            if curr_nodes[idx]==tnidx]
                    assert len(cidx)==1,"Mistake in Component"
                    x_rest[tnidx_name]=cidx[0]
                else:
                    x_rest[tnidx_name]=x_bars[tnidx_name]
            return x_rest

        #iterate over the already existing components and see who could go in
        candidate_insert_list=[]
        for cname,(cnode_ids,ccat_ids,cpi) in comp_dict.items():
            #First of all we have to generate the x_rest list
            x_rest=get_x_rest(nidx,cname,comp_dict,x_bars)

            #we will test the fitness of all the zero cat in this comp
            for zcidx in zero_cat_list:
                #Getting the mixture distribution probability
                x_full=x_rest.copy()
                x_full[node_id]=zcidx
                pmix_full=self.dist_handler.get_mixture_probability(
                                            x_full,self.infinte_mix_sample)

                #Now we have to generate the test probability
                pbase_full=self.dist_handler.get_intervention_probability(
                                                    x_full,eval_do=None)
                pcomp_left=self.dist_handler.get_intervention_probability(
                                                x_rest,
                                                eval_do=[cnode_ids,ccat_ids])
                #Finally getting the p_test
                ptest_full=(1-sum(old_pis+new_pis))*pbase_full\
                            +cpi*pcomp_left

                #Now we will see if we are equal within some slackness
                test_error=abs(pmix_full-ptest_full)
                if test_error<self.insert_eps:
                    candidate_insert_list.append([cname,zcidx,test_error])

                # pdb.set_trace()

        #Now we have to choose from candidate to insert based on least error
        candidate_insert_list.sort(key=lambda x:x[-1])
        inserted_cats=[]
        for cname,cidx,test_error in candidate_insert_list:
            #If we are just left with one category (which must not be here)
            if len(inserted_cats)==(len(zero_cat_list)-1):
                break
            #Other wise if the category is not inserted we will do it
            if cidx not in inserted_cats and nidx not in comp_dict[cname][0]:
                print("nidx:{}\t cidx:{}\t inserted in:{} with error:{}"
                        .format(nidx,cidx,cname,test_error))
                comp_dict[cname][0].append(nidx)
                comp_dict[cname][1].append(cidx)

                #Saying this node is already inserted
                inserted_cats.append(cidx)

        #Now its time to blacklist one of the category of this node
        #TODO:could be done smartly by looking at large prob to remove error
        left_cidx=list(set(zero_cat_list)-set(inserted_cats))
        x_bars[node_id]=left_cidx[-1]
        print("Left cidx:{} for node:{}".format(left_cidx[-1],nidx))

        # pdb.set_trace()
        return left_cidx[-1]


if __name__=="__main__":
    #Initializing the graph
    graph_name="asia"
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    base_network=BnNetwork(modelpath)

    #Tweaking of the CPDs
    #Skewed Asia CPD
    asia_cpd=base_network.base_graph.get_cpds("asia")
    from pgmpy.factors.discrete import TabularCPD
    new_asia_cpd=TabularCPD("asia",
                        2,
                        np.array([[0.10],[0.90]]))
    base_network.base_graph.remove_cpds(asia_cpd)
    base_network.base_graph.add_cpds(new_asia_cpd)

    #Zero entry in the either CPD
    new_either_cpd=np.array([[[0.95,0.95],[0.95,0.05]],
                            [[0.05,0.05],[0.05,0.95]]])
    base_network.base_graph.get_cpds("either").values=new_either_cpd
    # pdb.set_trace()



    #Creating artificial intervention
    do_config=[
                ((0,1,7),(1,0,1),0.3),
                ((2,3),(0,1),0.2),
            ]

    #Initializing our Solver
    solver=NonOverlapIntvSolve(base_network=base_network,
                                do_config=do_config,
                                infinte_mix_sample=True,
                                opt_eps=1e-10,
                                zero_eps=1e-5,
                                insert_eps=1e-5)
