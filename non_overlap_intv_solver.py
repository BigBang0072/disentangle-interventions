import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import time

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
            print("\n\n###############################################")
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

        #Get the porb dist when the interv matches
        x_left=x_bars.copy()
        p_left=self.dist_handler.get_intervention_probability(x_left,
                                                eval_do=None)
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
        # order_b=np.floor(np.log10(np.abs(np.min(b))))
        # if order_b<0:
        #     order_b=order_b*(-1)
        # scale_factor=10**order_b
        scale_factor=10000

        def optimization_func(x,):
            residual=np.sum((np.matmul(A,x)-b)**2)*scale_factor  #TODO
            print("residual:",residual)
            return residual

        #Setting up the constraints
        bounds=[(0.0,1.0)]*num_cats
        sum_constraints=[{"type":"ineq",
                        "fun":lambda x:1-np.sum(x)-sum(old_pis)}]
        zero_constraints=[{"type":"ineq",
                        "fun":lambda x: self.opt_eps-np.min(x)}]

        #Making the initial guess : #TODO
        # pi_0=np.zeros((num_cats,),dtype=np.float32)
        pi_0=self._get_good_guess_of_pi(A,b,p_left)
        print("Initial Guess:",pi_0)

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

    def _get_good_guess_of_pi(self,A,b,p_left):
        '''
        This function will get a better guess by solving the system of linear
        equation by hand and and choosing the appropriate zeroing strategy
        such that it's closest in the queare norm.

        O(k^2) time will be spent over this.
        '''
        assert A[-1,0]!=0,"Ak : Last element should not be zero"
        num_cats=b.shape[0]

        #Get all the relavent p_bases
        all_pbase=A[:,0]*(-1)
        all_pbase[0,]+=p_left

        #Finding the guy from which we will divide (largest for stability)
        deno_guy=np.argmax(all_pbase)
        deno_pbase=np.max(all_pbase)
        deno_b=b[deno_guy]

        #Calcuating the transformed b_value
        b_trans=(b-(all_pbase/deno_pbase)*deno_b)/p_left


        def solve_system_analytically(zero_guy):
            #First of all we have to find x_deno
            x_deno=None
            if zero_guy==deno_guy:
                x_deno=0.0
            else:
                x_deno=(-1*b_trans[zero_guy]*deno_pbase)/all_pbase[zero_guy]

            #Now we will find all the other values
            x_cand=np.zeros((num_cats,),dtype=np.float32)
            for cidx in range(num_cats):
                if cidx==zero_guy:
                    x_cand[cidx]=0.0
                elif cidx==deno_guy:
                    x_cand[cidx]=x_deno
                else:
                    x_cand[cidx]=(all_pbase[cidx]/deno_pbase)*x_deno\
                                    +b_trans[cidx]

            return x_cand

        #Now one by one we will try out the zeroing all categories
        min_x_cand=None
        min_error=10000000000000000.0
        for cidx in range(num_cats):
            x_cand=solve_system_analytically(cidx)

            #Now we will select only the one which give minimum residual
            residual=np.sum(((np.matmul(A,x_cand)-b))**2)
            print("Residual:{}\nNew Guess:{}".format(residual,x_cand))

            #Now first check we have to perform is positivity of all x
            # num_pos=np.sum(x_cand>=0.0)
            # if(num_pos<num_cats):
            #     # continue
            #     #Instead of removing the guy we will take abs val.
            #     #cuz mostly the error is -1e-25, which could be made +ve
            #     x_cand=np.abs(x_cand)

            if residual<min_error:
                min_error=residual
                min_x_cand=x_cand

        return min_x_cand

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
                test_error=(abs(pmix_full-ptest_full)/pmix_full)*100
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

                #Saying this node,category is already inserted
                inserted_cats.append(cidx)

        #Now its time to blacklist one of the category of this node
        #TODO:could be done smartly by looking at large prob to remove error
        left_cidx=list(set(zero_cat_list)-set(inserted_cats))
        x_bars[node_id]=left_cidx[-1]
        print("Left cidx:{} for node:{}".format(left_cidx[-1],nidx))

        # pdb.set_trace()
        return left_cidx[-1]


def redistribute_probability_mass(network,eps):
    '''
    This function will slightly nudge the CPDs of all the nodes such that,
    none of point will have zero probability.
    '''
    #One by one we will go through all the node in the network
    for node in network.topo_n2i.keys():
        #Retreiving the old network
        old_cpd=network.base_graph.get_cpds(node).values
        #Now we will go through all the columns (here) and spread goodness
        assert old_cpd.shape[0]==network.card_node[node],"dim mismatch"

        if np.sum(old_cpd==0)==0:
            continue
        print("Node:",node)
        print("old_cpd:\n",network.base_graph.get_cpds(node))

        def redistribute_column(col,eps):
            '''
            This fucntion will work on one single column
            eps     : total mass to be distributed among all the zeros
            '''
            #Getting the number of places to migrate
            num_zeros=np.sum(col==0.0)*1.0
            num_non_zero=col.shape[0]-num_zeros
            if num_zeros==0:
                return col

            #Getting the mass to distribute
            mass_add_per_zero=eps/num_zeros
            add_vector=(col==0.0)*mass_add_per_zero

            #Getting the vector to subtract (in proportion to their mass)
            sub_vector=col*eps

            #Now applying the redistribution
            new_col=col+add_vector-sub_vector
            assert abs(1-np.sum(new_col))<1e-20
            return new_col

        #Now we will apply this redistribution to every columns
        new_cpd=np.apply_along_axis(func1d=redistribute_column,
                                    axis=0,
                                    arr=old_cpd,
                                    eps=eps)

        #Now we will update the old_cpd
        network.base_graph.get_cpds(node).values=new_cpd
        print("new_cpd:\n",network.base_graph.get_cpds(node))
    return network

def get_random_internvention_config(network):
    '''
    This function will generate a random intervention to test our algo
    '''
    do_config=[]

    #Now we will start from node to node to create component
    for nidx in range(len(network.topo_i2n)):
        node_id=network.topo_i2n[nidx]
        num_cats=network.card_node[node_id]
        #Reset the random state stream
        rs = RandomState(MT19937(SeedSequence(int(time.time()%100000000))))

        '''
        Now we will generate a random string from 0 to len(do_config)
        0: dont take
        1: put in component 1 if present
        ..
        ..
        len(do_config): create new component


        1/2 prob of taking old, 1/2 (new or in no component)
        3/4 of coming and 1/4 not coming if old is there
        '''
        location_old=np.random.permutation(
                            np.arange(1,len(do_config)+1)).tolist()
        location_zero=np.random.randint(0,2,size=num_cats).tolist()
        #Now creating our location
        location=[]
        old_cidx=0
        for cidx in range(num_cats):
            if np.random.randint(0,2)==0 and old_cidx<len(location_old):
                #Come in some old component
                location.append(location_old[old_cidx])
                old_cidx+=1
            elif location_zero[cidx]==0:
                #Dont come
                location.append(0)
            else:
                #Come as new component
                location.append(len(do_config)+1)

        #Ensuring alteast one category is gone
        if np.sum(location==0)==0:
            location[np.random.randint(0,num_cats)]=0
        _=np.random.randint(0,num_cats)

        initial_num_comp=len(do_config)+1
        for cidx,choice in enumerate(location):
            print("Choice:",nidx,cidx,choice)
            if choice==0:
                #leave the guy
                continue
            elif choice==initial_num_comp:
                #Time for a new config
                do_config.append([[nidx,],[cidx,]])
            else:
                #Put it in old component
                do_config[choice-1][0].append(nidx)
                do_config[choice-1][1].append(cidx)
    #Now we will have to generate a mixing coefficient for these compoent
    mix_coefficient=np.random.uniform(1,100,size=len(do_config)+1)
    #Normalizing the coefficient
    mix_coefficient=mix_coefficient/np.sum(mix_coefficient)
    #Leaving out the probability for base distribution
    mix_coefficient=mix_coefficient[0:-1].tolist()

    do_nidx,do_cidx=zip(*do_config)
    do_config=list(zip(do_nidx,do_cidx,mix_coefficient))
    print("\n\n############# DO-CONFIG ##############")
    pprint(do_config)
    print("######################################\n\n")
    return do_config


if __name__=="__main__":
    #Initializing the graph
    graph_name="alarm"
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    base_network=BnNetwork(modelpath)

    #Tweaking of the CPDs
    total_distribute_mass=0.05
    redistribute_probability_mass(base_network,total_distribute_mass)

    #Creating artificial intervention
    do_config=get_random_internvention_config(base_network)[0:2]
    do_config[0]=(do_config[0][0],do_config[0][1],0.4)
    do_config[1]=(do_config[1][0],do_config[1][1],0.3)
    # pdb.set_trace()

    #Initializing our Solver
    solver=NonOverlapIntvSolve(base_network=base_network,
                                do_config=do_config,
                                infinte_mix_sample=True,
                                opt_eps=1e-10,
                                zero_eps=1e-5,
                                insert_eps=0.05)#This is in percentage error