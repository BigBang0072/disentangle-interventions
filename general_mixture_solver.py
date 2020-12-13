import numpy as np
np.random.seed(1)
import pdb
from pprint import pprint
from collections import defaultdict
from toposort import toposort_flatten

from data_handle import BnNetwork
from non_overlap_intv_solver import DistributionHandler
from interventionGenerator import InterventionGenerator
from evaluator import EvaluatePrediction

class GeneralMixtureSolver():
    '''
    This class will be responsible for disentangling the general mixture of
    intervention with assumption that one category of each node is not present
    in the mixture at all.
    '''
    base_network=None           #Bayesian network corresponding to problem
    infinite_sample_limit=None  #boolean to denote to simulate infinite sample

    def __init__(self,base_network,do_config,
                infinite_sample_limit,mixture_samples,
                pi_threshold,split_threshold):
        self.base_network=base_network
        self.infinite_sample_limit=infinite_sample_limit

        #Ininitializing the minimum threshold for the mixing coefficient
        self.pi_threshold=pi_threshold
        #Threshold allows the old target to split "slightly" more than possible
        self.split_threshold=split_threshold

        #Initializing the global target count (for numbering targets)
        self.global_target_counter=0

        #Initializing the distirubiton handler
        self.dist_handler=DistributionHandler(base_network,
                                            do_config,
                                            mixture_samples)

    def solve(self,):
        '''
        This function in the main driver for solving the mystery, getting all
        the interventions present in the mixture along with their mixing
        coefficient.

        Important Running variables:
        v_bars      : dict containing the left out category of the node
                        upto the point where we have reached in the
                        topological order.
                        {node: node_category_blacklisted}
        target_dict : list of all the targets (aka component)
                        {
                        target_name : [ [nidxs], [cidxs], pi ]
                        }
                        This time the pi will keep evolving until all the nodes
                        in topological order is not covered
        '''
        #Initializing the v_bar used for topoligically isolating targets
        v_bars={}
        #Initializing the target dictionary with empty internvetion
        target_dict={
                "t0":[[],[],1.0]
        }
        #Updating the counter since we now have one target
        self.global_target_counter=1

        #Starting finding the merged targets topologically
        for vidx in range(len(self.base_network.topo_i2n)):
            print("=================================================")
            print("\tSplitting for node:{}".format(vidx))
            print("=================================================")
            blacklisted_cidx,split_target_dict=\
                        self._split_all_targets(v_bars,vidx,target_dict)

            #Updating the v_bar for this node
            print("Glabal blacklisted category node:{} cat:{}".format(vidx,
                                                        blacklisted_cidx))
            node_id=self.base_network.topo_i2n[vidx]
            v_bars[node_id]=blacklisted_cidx

            #Now aggregating the split targets into one
            print("Aggregating the split target with the old ones")
            self._aggregate_target_and_splits(vidx,target_dict,
                                                split_target_dict)
            print("Completed the splitting process for current node")
            print("Tragets Found:")
            pprint(target_dict)

        return target_dict

    def _split_all_targets(self,v_bars,vidx,target_dict):
        '''
        This function will split the existing merged targets into new target
        by adding the possible internvetions present belonding to current node
        in those merged targets.
        '''
        #First of all we have to get the topological ordering among the targets
        tsorted_target_names=self._get_targets_topologically(target_dict)
        print("topologically_sorted_targets:")
        pprint(tsorted_target_names)

        #Now we will go one by one in this order to split each of the target
        split_target_dict={}
        for tidx,tname in enumerate(tsorted_target_names):
            print("Splitting target:{}".format(tname))
            #Getting the all the targets which have been split before
            before_target_names = tsorted_target_names[0:tidx]
            print("target_dict:")
            pprint(target_dict)
            print("before_target_names:")
            pprint(before_target_names)
            print("split_target_dict:")
            pprint(split_target_dict)

            #Splitting the current target by inserting the current node
            split_pis=self._split_target(v_bars,vidx,tname,target_dict,
                                            before_target_names,
                                            split_target_dict)
            #Adding the split solution for this target after thresholding
            split_pis = self._renormalize_threshold_split_pis(tname,
                                                    split_pis,target_dict)
            split_target_dict[tname]=split_pis
            print("#################################################")
            # pdb.set_trace()

        #Finding the global blacklist category for this node
        self._print_split_target_dict(split_target_dict)
        blacklisted_cidx = self._get_global_blacklist_cidx(split_target_dict)

        return blacklisted_cidx,split_target_dict

    def _get_targets_topologically(self,target_dict):
        '''
        This function will topologically sort the target based on the critera:
        a<b if a \subset b.
        '''
        def get_smallar_target(target_dict,tname1,tname2):
            '''
            This function will return the the lower target based on the desired
            comparison.
            '''
            target1=set(zip(target_dict[tname1][0],target_dict[tname1][1]))
            target2=set(zip(target_dict[tname2][0],target_dict[tname2][1]))
            if target1.issubset(target2):
                return tname1
            elif target2.issubset(target1):
                return tname2
            #Otherwise none of them is smallar hence no edge will be placed
            return None

        #Generating the inverted adjacency list
        target_names=list(target_dict.keys())
        inv_adj_set={tname:set() for tname in target_names}
        for iidx in range(0,len(target_names)):
            for jidx in range(iidx+1,len(target_names)):
                #Comparing the two targets
                tname1=target_names[iidx]
                tname2=target_names[jidx]
                smallar_target=get_smallar_target(target_dict,tname1,tname2)

                if smallar_target==tname1:
                    #There is an arrow from tname1 to tname2
                    inv_adj_set[tname2].add(tname1)
                elif smallar_target==tname2:
                    #There is an arrow from tname2 to tname1
                    inv_adj_set[tname1].add(tname2)
        #Now we will get the topological order among the targets
        # pdb.set_trace()
        tsorted_target_names=toposort_flatten(inv_adj_set)

        return tsorted_target_names

    def _split_target(self,v_bars,vidx,tname,target_dict,before_target_names,split_target_dict):
        '''
        This function will split the current "target:tidx" by instering
        different categories of "vidx" as intervention.

        split_target_dict   : dict correcsponding to every target in target_dict
                                with coefficient of split target from it which
                                have already been found.
                                {
                                ti : [pi_1,pi_2,\ldots,pi_k]
                                }

        '''
        #Getting the current target
        curr_target = target_dict[tname]

        #Now we will generate the system of equation for the current target
        node_id=self.base_network.topo_i2n[vidx]
        num_cats=self.base_network.card_node[node_id]
        #Initializing the system matrix
        A=np.zeros((num_cats,num_cats),dtype=np.float64)
        b=np.zeros((num_cats,),dtype=np.float64)
        small_a=np.zeros((num_cats,),dtype=np.float64)

        #First of all we have to get the vt_bars U t' for current target
        vbtut=self._get_vbtut(v_bars,curr_target)
        print("vbtut:")
        pprint(vbtut)

        print("Getting the entries for the system of equation")
        for cidx in range(num_cats):
            #Calculate the b'' for this category
            bp = self._get_bp(vbtut,node_id,cidx,target_dict,
                                before_target_names,split_target_dict)

            #Now getting the other terms of matrix for the current target
            curr_point = vbtut.copy()
            curr_point[node_id]=cidx
            p_ct = self.dist_handler.get_intervention_probability(
                                    curr_point,
                                    eval_do=[curr_target[0],curr_target[1]])

            vbtut  = vbtut.copy()
            p_ct_vbtut = self.dist_handler.get_intervention_probability(
                                    vbtut,
                                    eval_do=[curr_target[0],curr_target[1]])

            a = p_ct / p_ct_vbtut
            bpp = bp /p_ct_vbtut

            b[cidx] =bpp
            small_a[cidx] = a
            A[cidx,:] = (-1*a)
            A[cidx,cidx] += 1
            print("node_id:{}\tcidx:{}\ta:{}\tb:{}\t".format(
                                                    node_id,cidx,a,bpp))

        #Now we are ready with our matrix
        split_pis=self._solve_system_analytically(small_a,A,b)
        return split_pis

    def _get_vbtut(self,v_bars,curr_target):
        '''
        This function will get the vt_bar which will isolate all the other
        targets which comes topologically after current target.

        vbtut    : a "point" represented by dictionary of node name and
                    corresponding category of that node
        '''
        #Copying the full blacklisted candidates for now
        vbtut=v_bars.copy()
        #Replacing the category for node which are in current target
        for nidx,cidx in zip(curr_target[0],curr_target[1]):
            node_name = self.base_network.topo_i2n[nidx]
            vbtut[node_name]=cidx

        return vbtut

    def _get_bp(self,vbtut,node_id,cidx,target_dict,before_target_names,split_target_dict):
        '''
        This function will calcuate the b' for the current cateogy
        '''
        #Adding the current node to the rest of point
        point= vbtut.copy()
        point[node_id] = cidx
        #Getting the point leaving the current node
        vbtut = vbtut.copy()
        #Initilaizing the b''
        bpp=0

        #Getting the mixture probability
        p_mix = self.dist_handler.get_mixture_probability(point,
                                                    self.infinite_sample_limit)
        bpp = bpp + p_mix

        #Getting the old targets probability
        for tname,(tnode_idx,tcat_idx,tpi) in target_dict.items():
            #Get the point probability for this target
            p_t = self.dist_handler.get_intervention_probability(
                                                point,
                                                eval_do=[tnode_idx,tcat_idx])

            bpp = bpp - tpi*p_t

        #Now getting the already splitted targets contribution
        assert len(before_target_names)==len(split_target_dict),"Size mismatch"
        for btname in before_target_names:
            #Getting the intervention location of this splitted target
            tnode_idx,tcat_idx,_=target_dict[btname]

            #Getting the probability of vbtut
            p_t_vbtut = self.dist_handler.get_intervention_probability(
                                                vbtut,
                                                eval_do=[tnode_idx,tcat_idx])
            #Getting the probability of full point for the target
            p_t = self.dist_handler.get_intervention_probability(
                                                point,
                                                eval_do=[tnode_idx,tcat_idx])

            for tsidx,tspi in enumerate(split_target_dict[btname]):
                #Now we will add the contribution of split target on bpp
                delta_val = int(cidx == tsidx)
                bpp = bpp - tspi*(p_t_vbtut*delta_val - p_t)

        #Now we are done with bpp
        return bpp

    def _solve_system_analytically(self,small_a,A,b):
        '''
        This function will calculate the mixing coefficient for the possible
        spliting of the target after adding the curent node as intervention
        in the current target.

        Stage 1:
            1. First we will assume that, we will have one unique solution
            2. In case we have mutiple solution (which we should not technically)
                2.1 We will warn in the log
                2.2 Select the solution which has minimum residual then
            3. After getting the pi_s, we will threshold the small pi
                and not let them split furthur based on some threshold
                3.1 This threshold will be decided based on the actual/prior
                    about the strength of intervention component in mixture
            (This thresholding will be done )

        Stage 2:
            1. Here we will handle the case, if due to some error the line
                is passsing from outside the solution simplex.
            2. We will have to project the crossing over of the assumption
                face=0, with the closest point in the solution simplex.

        Assumption:
            1. Following our positivity assumption, all out CPDs are also
                positive. So no need to be concerned about the choosing
                the k^th guy.
        '''

        #Trying one by one each of the category to be zero
        feasible_solutions={}
        feasible_score=[]
        minimum_residual=float("inf")
        selected_cidx=-1
        print("Solving system analytically:")
        for cidx in range(A.shape[0]):
            #Getting the solution if this cidx is zero
            split_pis = b - (small_a/small_a[cidx])*b[cidx]
            #But the current cidx is taken as zero
            split_pis[cidx]=0.0
            #TODO: Possibly we could do thresholding here

            #Adding this solution to list of feasible solution
            feasible_solutions[cidx]=split_pis

            #Now checking the validity of this solution (Stage 1: only one should go)
            if np.sum(split_pis>=0)==A.shape[0]:
                #Getting the residual of the solution
                residual = np.sum((np.matmul(A,split_pis)-b)**2)
                feasible_score.append((cidx,residual))
                print("cidx:{}\t residual:{}\t fpi:".format(
                                                    cidx,residual,split_pis))

                if residual<minimum_residual:
                    minimum_residual=residual
                    selected_cidx=cidx
        print("selected cidx:{}\t residual:{}\t pis:{}\t".format(
                                        selected_cidx,
                                        minimum_residual,
                                        feasible_solutions[selected_cidx]))

        #So we have selected the solution with the minimum residual
        return feasible_solutions[selected_cidx]

    def _renormalize_threshold_split_pis(self,tname,split_pis,target_dict):
        '''
        This function will threshold the split pis such that small pis
        less than the threshold value is made zero, in order to have sparser
        solution. This thresold value will generally come from the domain
        knowledge based on the strength of the intervention targets.

        eg. if all the intervention targets have coefficient>0.01
            with confidence interveal over it as 0.005 then
            a good threshold will be 0.005

        CHANGE-LOG:
        Stage 0: We will threhsold the splitpis after we have done splitting
                    all the individual targets at this level.

        Stage 1: We will threshold the split pis before going to the other
                    targets, since mistake here will contribute to other
                    target bpp calculation and hence their split_pis.
                    So better threshold right now instead of being done
                    with all the target

        Stage 2: Now, since we could be splitting more than whatever possible
                    we will have to renormalize the spit pis:

                    If we are overshooting:
                    [split_pi/sum(split_pi)]*pi_actual

                    But should we renormalize again after we have thresholded
                    for any understrength which could occur in the pi_left?

                    It's debatable that we are rescaling i.e keeping the sol
                    direction same, instead of projecting into the right
                    direction. We could see that in next change.


        '''
        print("Renormalizing the split pis if overshooting or understrength")
        split_pis = self._renormalize_split_pis(tname,split_pis,target_dict)


        print("Threshold the split pis: pi_threshold:{}".format(
                                                    self.pi_threshold))
        #Clipping the pi's to zero if they are below our threshold
        split_pis = split_pis*(split_pis>self.pi_threshold)
        print("tname:{}\t thresholded_spi:{}".format(tname,split_pis))

        return split_pis

    def _get_global_blacklist_cidx(self,split_target_dict):
        '''
        This function will serach for the category which is zero coefficient
        i.e not present in any of the thresholded split of any of the
        targets.
        Change LOG:
        1. Now even if we dont have a global blacklist, we will forcefully
            make a particular cidx as zero based on the minimum sum in the
            particular column of the split_pis.
        '''
        all_target_split_pis = np.stack(
                                list(split_target_dict.values()),axis=0)

        #Now lets serach for the cidx which is absent from all targets
        sum_split_pis = np.sum(all_target_split_pis,axis=0)

        blacklisted_cidx=None
        sum_voilation=float("inf")
        for cidx in range(sum_split_pis.shape[0]):
            #Checking if the current columns has minimum sum voilations
            if sum_split_pis[cidx]<sum_voilation:
                blacklisted_cidx =  cidx
                sum_voilation = sum_split_pis[cidx]
            print("cidx:{} \tsum_voilation:{}".format(cidx,sum_split_pis[cidx]))
        assert blacklisted_cidx!=None,"No global missing category"

        #Now we will make that particular column explicitely zero
        for tname in split_target_dict.keys():
            #Making that particular column as zero
            split_target_dict[tname][blacklisted_cidx]=0.0

        return blacklisted_cidx

    def _aggregate_target_and_splits(self,vidx,target_dict,split_target_dict):
        '''
        This function will create new targets based on the split and
        redistribute the pis from the old target to those newly splitted
        targets.
        '''
        #Iterating over all the targets
        new_target_dict={}
        spent_target_list=[]
        for tname,(tnode_idxs,tcat_idxs,tpi) in target_dict.items():
            #Getting the split coefficients
            split_pis = split_target_dict[tname]
            #Frist of all let's correct if there is some understrength
            split_pis = self._renormalize_split_pis(tname,split_pis,target_dict)

            for cidx,spi in enumerate(split_pis):
                #Leaving out the categories which have zero coefficient
                if spi==0.0:
                    continue
                #Adding the internvetion config
                split_tnode_idxs = tnode_idxs.copy()
                split_tcat_idxs = tcat_idxs.copy()
                #Adding the split variabels on the base target config
                split_tnode_idxs.append(vidx)
                split_tcat_idxs.append(cidx)
                split_tname = "t"+str(self.global_target_counter)

                new_target_dict[split_tname] = [split_tnode_idxs,
                                                split_tcat_idxs,
                                                spi]
                #Incrementing the global counter
                self.global_target_counter+=1

            #Now we will remove the coefficient from tpi which is gone in split
            tpi = tpi - np.sum(split_pis)
            assert (tpi>=self.split_threshold),"Splitting more than it could afford"
            #if we have given our complete mixing coefficent in splitting
            if(tpi<self.pi_threshold):
                spent_target_list.append(tname)
            target_dict[tname][-1]=tpi

        print("old_target_dict:")
        pprint(target_dict)
        print("new_target_dict:")
        pprint(new_target_dict)

        #Removing the spent target from the old dict
        for spent_target in spent_target_list:
            del target_dict[spent_target]

        #Now its time to merge theold and new target dict
        for tname,config in new_target_dict.items():
            assert tname not in target_dict, "Target name already present"
            target_dict[tname]=config
        print("Retribution and Merging the split with the targets complete!")

    def _print_split_target_dict(self,split_target_dict):
        '''
        Useful in visualizing the global blacklisted category for this node
        among all the splitted targets
        '''
        for tname,split_pis in split_target_dict.items():
            print("tname:{}\t thresholded_spi:{}".format(tname,split_pis))

    def _renormalize_split_pis(self,tname,split_pis,target_dict):
        '''
        Now, since we could be splitting more than whatever possible
        we will have to renormalize the spit pis:

        If we are overshooting:
        [split_pi/sum(split_pi)]*pi_actual

        It's debatable that we are rescaling i.e keeping the sol
        direction same, instead of projecting into the right
        direction. We could see that in next change.
        '''
        #Now we will renormalize the pis if we are oversplitting
        target_pi = target_dict[tname][-1]
        sum_split_pis  = np.sum(split_pis)
        #Case 1 : If we are oversplitting
        if(sum_split_pis>target_pi):
            #Scale Down (overshooting)
            #Automatically this will render the target_pi_left zero in aggregation step
            split_pis = (split_pis/sum_split_pis)*target_pi
            print("tname:{}\t downscaled_spi:{}".format(tname,split_pis))
        elif( (target_pi-sum_split_pis) < self.pi_threshold ):
            #Scale Up (understrength)
            #Then we will make this target_pi_left as zero by redistributing again
            split_pis = (split_pis/sum_split_pis)*target_pi
            #Other wise we are in happy flow and have to do nothing
            print("tname:{}\t upscaling_spi:{}".format(tname,split_pis))

        return split_pis

if __name__=="__main__":
    num_nodes=16
    node_card=8
    #Creating a random graph
    from graph_generator import GraphGenerator
    generator_args={}
    generator_args["scale_alpha"]=5
    graphGenerator = GraphGenerator(generator_args)
    modelpath = graphGenerator.generate_bayesian_network(num_nodes=num_nodes,
                                            node_card=node_card,
                                            num_edges=53,
                                            graph_type="SF")
    base_network=BnNetwork(modelpath)
    # pdb.set_trace()

    #Getting a random internvetion
    target_generator = InterventionGenerator(S=50,
                                            max_nodes=num_nodes,
                                            max_cat=node_card,
                                            num_node_temperature=float("inf"),
                                            pi_dist_type="inverse",
                                            pi_alpha_scale=5)
    do_config = target_generator.generate_all_targets()

    #Testing by having the sample from mixture distribution
    infinite_sample_limit=True
    mixture_samples=None
    if not infinite_sample_limit:
        mixture_sample_size=100000
        mixture_samples = base_network.generate_sample_from_mixture(
                                            do_config=do_config,
                                            sample_size=mixture_sample_size)

    #Testing the general mixture solver
    solver = GeneralMixtureSolver(
                            base_network=base_network,
                            do_config=do_config,
                            infinite_sample_limit=infinite_sample_limit,
                            mixture_samples=mixture_samples,
                            pi_threshold=1e-10,
                            split_threshold=(-1e-10),
            )
    pred_target_dict=solver.solve()

    #Now lets evaluate the solution
    evaluator = EvaluatePrediction(matching_weight=0.5)
    evaluator.get_evaluation_scores(pred_target_dict,do_config)
