import numpy as np
np.random.seed(1)
import pdb
from pprint import pprint
from collections import defaultdict
from toposort import toposort_flatten
import itertools as it
from scipy.stats import dirichlet
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool

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
                infinite_sample_limit,base_samples,mixture_samples,
                pi_threshold,split_threshold,
                positivity_epsilon,positive_sol_threshold):
        self.base_network=base_network
        self.infinite_sample_limit=infinite_sample_limit

        #Ininitializing the minimum threshold for the mixing coefficient
        self.pi_threshold=pi_threshold
        #Threshold allows the old target to split "slightly" more than possible
        self.split_threshold=split_threshold
        #Threshold to allow positive sol when solving Ax=b
        self.positive_sol_threshold=positive_sol_threshold

        #Initializing the global target count (for numbering targets)
        self.global_target_counter=0

        #Initializing the distirubiton handler
        self.dist_handler=DistributionHandler(base_network,
                                            do_config,
                                            base_samples,
                                            mixture_samples,
                                            infinite_sample_limit,
                                            positivity_epsilon,
                                            )

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
        minimum_residual_positive=float("inf")
        minimum_residual_project=float("inf")
        selected_cidx_positive=-1
        selected_cidx_project=-1
        print("Solving system analytically:")
        for cidx in range(A.shape[0]):
            #Getting the solution if this cidx is zero
            split_pis = b - (small_a/small_a[cidx])*b[cidx]
            #But the current cidx is taken as zero
            split_pis[cidx]=0.0
            #TODO: Possibly we could do thresholding here

            #Now checking the validity of this solution (Stage 1: only one should go)
            if np.sum(split_pis>self.positive_sol_threshold)==A.shape[0]:
                split_pis = split_pis*(split_pis>0)

                #Getting the residual of the solution
                residual = np.sum((np.matmul(A,split_pis)-b)**2)
                feasible_score.append((cidx,residual))
                print("cidx:{}\t residual:{}\t fpi:".format(
                                                    cidx,residual,split_pis))

                if residual<minimum_residual_positive:
                    #Adding this solution to list of feasible solution
                    feasible_solutions[cidx]=split_pis
                    #Updating the solution tracker
                    minimum_residual_positive=residual
                    selected_cidx_positive=cidx
            elif selected_cidx_positive==-1:
                #If no positive solution is yet found
                split_pis = split_pis*(split_pis>0)

                #Getting the residual of the solution
                residual = np.sum((np.matmul(A,split_pis)-b)**2)
                feasible_score.append((cidx,residual))
                print("cidx:{}\t residual:{}\t fpi:".format(
                                                    cidx,residual,split_pis))

                if residual<minimum_residual_project:
                    #Adding this solution to list of feasible solution
                    feasible_solutions[cidx]=split_pis
                    #Updating the solution tracker
                    minimum_residual_project=residual
                    selected_cidx_project=cidx

        #Now we are ready to select one of cidx
        selected_cidx=selected_cidx_positive
        minimum_residual= minimum_residual_positive
        if selected_cidx_positive==-1:
            selected_cidx=selected_cidx_project
            minimum_residual=minimum_residual_project


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

    def solve_by_em(self,max_target_order,epochs,log_epsilon,num_parallel_calls):
        '''
        This function will take a different route to get the mixing coefficient
        by estimating the mixing coefficients by Expectation Maximization.

        Caution:
        This function generate all the possible internvetions as a candidate
        and then update their posterior based on the evidence.
        So, for a larger sized graph there could be exponentially many
        possible intervention targets which could fill up the RAM.

        Arguments:
            max_target_order    : the degree/order of intervention possible
                                    in the mixture. We will only search all
                                    possible intenrvention upto this order.
                                    This could at max be of "num_nodes"
                                    degree. But with prior knowledge it
                                    there could be upper bound on order
                                    of the interventions.
        '''
        import random
        random.seed(142345321)
        np.random.seed(2142453)
        
        #First of all we generate all possible internvetion targets
        all_target_dict = self._generate_all_possible_targets(max_target_order)
        all_target_keys,all_target_pi = zip(*all_target_dict.items())
        all_target_pi = np.array(all_target_pi,dtype=np.float64)
        # pdb.set_trace()

        #Getting the do_config hash for ease of calculating mse
        do_config_dict = {
                            (tuple(tnode),tuple(tcat)):tpi 
                                for tnode,tcat,tpi in self.dist_handler.do_config
                        }
        
        #Having all the targets with equal prior
        # all_target_pi = (all_target_pi*0+1)/all_target_pi.shape[0]
        #Printing the initial target pi
        print("Initialized the Intervnetion Targets")
        for tidx,target in enumerate(all_target_keys):
            target_loc = (tuple(target[0]),tuple(target[1]))
            atpi = self.dist_handler.do_config[target_loc] if target_loc in self.dist_handler.do_config else 0.0
            print("tnode:{}\ttcats:{}\tatpi:{}\titpi:{}".format(
                                        target[0],
                                        target[1],
                                        atpi,
                                        all_target_pi[tidx],
                )
            )

        #Creating the evaluation function
        # evaluator = EvaluatePrediction(matching_weight=0.5)

        #Now running the EM algorithm given number of epochs
        pred_target_dict=None
        mse_overall_list=[]
        avg_logprob_list=[]
        for enum in range(1,epochs+1):
            #First of all we need to impose the exclusion assumption
            # print("Enforcing the Pauli's Exclusion Principle")
            # all_target_pi = self._impose_exclusion(all_target_keys,all_target_pi)

            #Running one step of the EM
            print("\n\n\nRunning one EM Step:")
            all_target_pi = self._run_em_step_once(
                                        all_target_keys,
                                        all_target_pi,
                                        num_parallel_calls
            )
            print("Target Pi:")
            pprint(all_target_pi)

            #Returning the predicted target dict
            pred_target_dict={}
            step_mse = 0.0
            for tidx,target in enumerate(all_target_keys):
                tnode = tuple(target[0])
                tcat  = tuple(target[1])
                tpi   = all_target_pi[tidx]
                pred_target_dict["t{}".format(tidx)]=[tnode,tcat,]
                
                #Calculating the mse here itself
                delta = None
                atpi = 0.0
                if (tnode,tcat) in do_config_dict:
                    delta = tpi - do_config_dict[(tnode,tcat)]
                    atpi = do_config_dict[(tnode,tcat)]
                else:
                    delta = tpi
                #Adding the contribution
                step_mse += delta**2
                
                #Logging the estimated pi value
                print("tnode:{}\ttcat:{}\tatpi:{:.3f}\tptpi:{:.3f}".format(
                                            tnode,
                                            tcat,
                                            atpi,
                                            tpi
                     )
                )
            
            step_mse = step_mse/len(all_target_keys)
            print("Step MSE: {}".format(step_mse))

            # #Getting the evaluation score
            # _,_,mse_all=evaluator.get_evaluation_scores(
            #                             pred_target_dict,
            #                             self.dist_handler.do_config,
            # )
            mse_overall_list.append(step_mse)

            #Next we will estimate the likelihood of the optimizer
            if enum%5==0:
                avg_logprob_list.append(
                            self._get_loglikelihood(all_target_keys,
                                                    all_target_pi,
                                                    log_epsilon,
                            )
                )
                print("Average Log probability of the learnt mixture:",avg_logprob_list[-1])

            #Plotting the evaluation metrics
            # if enum%5==0:
            #     plt.plot(range(len(mse_overall_list)),mse_overall_list,"o-")
            #     plt.show()
            #     plt.close()

        print("===============================================")
        pprint("STEP MSE TIMELINE")
        pprint(mse_overall_list)
        print("===============================================")
        pprint("AVG LOGPROB TIMELINE")
        pprint(avg_logprob_list)
        print("===============================================")
        pprint("ACTUAL DO CONFIG")
        pprint(do_config_dict)
        print("===============================================")
        pprint("PRED TARGET DICT")
        pprint(pred_target_dict)
        print("===============================================")
        pprint("PRED TARGET PI")
        pprint(all_target_pi)
        

        return pred_target_dict,mse_overall_list,avg_logprob_list
    
    def _impose_exclusion(self,all_target_keys,all_target_pi):
        '''
        Since in all our work we are expecting that intervention exclusion
        principle holds, so here also we will ensure that the least valued
        (in terms of sum pi in all target) category for every node is zeroed
        out. Then we will renormalize the whole distribution so that
        it is still valid.
        '''
        #Initializign the node,cat importance dict
        node_cat_value_dict = {
                    nidx:[
                            dict(cidx=cidx , cscore=0.0 , ptidx=[])
                                for cidx in range(
                                    self.base_network.card_node[self.base_network.topo_i2n[nidx]]
                                )
                        ]
                            for nidx in range(len(self.base_network.topo_i2n.keys()))
        }

        #Getting the node-cat importance
        for tidx,(tnode,tcat) in enumerate(all_target_keys):
            #Going through every node and category pair in this target
            for tnidx,tcidx in zip(tnode,tcat):
                #Adding the score
                node_cat_value_dict[tnidx][tcidx]["cscore"]+=all_target_pi[tidx]
                #Keeping track of the location if push come to shove and we need to nullify it
                node_cat_value_dict[tnidx][tcidx]["ptidx"].append(tidx)
        
        #Now we need to find the category to remove for each node
        for nidx,clist in node_cat_value_dict.items():
            #Getting the "minimum-guy"
            min_citem = min(clist,key=lambda x:x["cscore"]) 
            print("For Exclusion Principle: nidx:{}\tcidx:{}".format(nidx,min_citem["cidx"]))

            #Now mulifing all the target pis for this category of this node
            for tidx in min_citem["ptidx"]:
                all_target_pi[tidx]=0.0

        #Renormalizing the pi's
        print("Renormalizing after exclusion principle enforcement")
        all_target_pi = all_target_pi / np.sum(all_target_pi)

        return all_target_pi

    def _get_loglikelihood(self,all_target_keys,all_target_pi,log_epsilon):
        '''
        This function will return the log likelihood of data given the current
        mixture parameters.
        '''
        overall_logprob=0.0
        num_examples = self.dist_handler.mixture_samples.shape[0]
        for eidx in range(num_examples):
            #Getting the point/"example" from the df
            point = self.dist_handler.mixture_samples.iloc[eidx]

            #get the probability of this sample on all targets
            sample_prob_sum=0.0
            for tidx,target in enumerate(all_target_keys):
                tprob = self.dist_handler.get_intervention_probability(
                                            sample=point,
                                            eval_do=list(target)
                )
                #Updating the prob in sample _posterior_pi
                sample_prob_sum+= tprob*all_target_pi[tidx]

            #Next we calculate the lop probability of sample
            overall_logprob += np.log(sample_prob_sum+log_epsilon)

        return overall_logprob/num_examples
    
    def _run_em_step_once(self,all_target_keys,all_target_pi,num_parallel_calls):
        '''
        This function will run the em step once by calculating the
        posterior and then updating the model parameters.
        '''
        #Caching all the targets so that we dont recreate do-graph for each example
        sample_point = self.dist_handler.mixture_samples.iloc[0]
        _,self.dist_handler = em_step_worker_kernel(dict(
                                                    all_target_keys=all_target_keys,
                                                    all_target_pi=all_target_pi,
                                                    point=sample_point,
                                                    dist_handler=self.dist_handler
        ))

        #Go through all the examples one by one to get posterior
        num_examples = self.dist_handler.mixture_samples.shape[0]
        #This could be parallelized on multiple threads.
        #Creating the task list
        task_config_list = []
        for eidx in range(num_examples):
            # print("Getting target distribution for sample: ",eidx,all_target_pi.shape)
            #Getting the point/"example" from the df
            point = self.dist_handler.mixture_samples.iloc[eidx]
            task_config_list.append(dict(
                                        all_target_keys = all_target_keys,
                                        all_target_pi = all_target_pi,
                                        point = point,
                                        dist_handler = self.dist_handler
            ))
        
        #Now running the job in parallel
        with Pool(num_parallel_calls) as p:
            production = p.map(em_step_worker_kernel,task_config_list)
        
        #Now getting the average posterior porbability
        all_posterior_pi = np.stack(
                    [
                        sample_posterior_pi for sample_posterior_pi,_ in production
                    ],
                    axis=0
        )
        posterior_pi = np.mean(all_posterior_pi,axis=0)

        return posterior_pi

    def _generate_all_possible_targets(self,max_target_order):
        '''
        This function will generate all possible intervention target
        upto a given order of intervention.
        '''
        max_target_order = min(max_target_order,
                                len(self.base_network.topo_i2n)
        )
        node_idxs = list(self.base_network.topo_i2n.keys())

        all_target_dict={}
        #Now creating the intervnetion targets order by order
        for oidx in range(0,max_target_order+1):
            #First of all we will generate all the nidx combination
            comb_nidx = it.combinations(node_idxs,oidx)

            #Next we need to generate all possible setting of all target nodes
            for tnidxs in comb_nidx:
                #Generating all possible assignment of this target nodes
                cidx_list=[range(
                        self.base_network.card_node[self.base_network.topo_i2n[nidx]]
                    )
                    for nidx in tnidxs
                ]
                #Getting all setting of all these nodes in this target
                tsettings = it.product(*cidx_list)

                #Now we will create them as separate target
                for tset in tsettings:
                    all_target_dict[(tnidxs,tset)]=0.0

        #Now we need to initialize these mixing coefficient randomly
        mixing_coeffs = np.squeeze(dirichlet.rvs(
                                size=1,
                                alpha=np.ones(len(all_target_dict)),
        ))
        # pdb.set_trace()
        for tidx,target in enumerate(all_target_dict):
            all_target_dict[target]=mixing_coeffs[tidx]

        return all_target_dict
    
    def solve_by_brute_force_sys_eq(self,zero_eps,num_parallel_calls):
        '''
        Here we will solve the problem via system of equation over all possible 
        intervnetion targets and marginal which are possible.

        Additonaly, instead of leaving the search of the exlcuded catgory upto 
        the model, we will serach over all the k^n possible zeros setting by brute
        force trying all of them.

        '''
        #First of all we need to generate the the A matrix
        A,b,all_target_list,actual_target_dict = self._get_full_Ab_matrix()

        #Getting all possible zeroing configuration
        zero_config_list = self._get_all_exclusion_config()

        #Now running the optimization for all possible zero configs
        all_problem_list=[]
        for zero_config in zero_config_list:
            all_problem_list.append(
                            dict(
                                exclusion_config = zero_config,
                                zero_eps = zero_eps,
                                A=A,
                                b=b,
                                all_target_list=all_target_list,
                                actual_target_dict=actual_target_dict
                            )
            )
        
        #Now we will start the solution in parallel
        with Pool(num_parallel_calls) as p:
            all_result_list = p.map(solve_for_a_exclusion_config_worker_kernel,all_problem_list)
            
        
        #Getting the best configuration
        all_result_list.sort(key=lambda x: x["opt_result"]["fun"],reverse=True)
        pprint(all_result_list)
        print("===================================")
        print("Actual Target Dict:")
        for tidx,target in enumerate(all_target_list):
            print("atpi:{:0.4f}\toptpi:{:0.4f}\ttarget:{}".format(
                                actual_target_dict[target],
                                all_result_list[-1]["opt_result"]["x"][tidx],
                                target
                    )
            )
        print("==================================")
        print("Optimal Balcklist Categories Found:")
        pprint(all_result_list[-1]["zero_config"])

        #Collecting all the result in standard pred_target_dict
        pred_target_dict = {}
        for tidx,target in enumerate(all_target_list):
            pred_target_dict["pt{}".format(tidx)] = (
                                                        target[0],
                                                        target[1],
                                                        all_result_list[-1]["opt_result"]["x"][tidx]
                                                    )
        return pred_target_dict

    def _get_full_Ab_matrix(self,):
        '''
        This function will genrerate the full A matrix with all the possilbe intervnetion 
        and the point locations.
                            target : t                     RHS: b
        point : x       |  p_t(x)-p(x)  |             | p_mix(x)-p(x) |
        '''
        #First of all we need the whole target list
        all_target_dict = self._generate_all_possible_targets(
                                        max_target_order=len(self.base_network.topo_i2n)
        )
        #Removing the base distribution 
        del all_target_dict[((),())]
        
        #Getting the names of the targets
        all_target_list = list(all_target_dict.keys())

        #Getting all the point of evaluation
        all_point_list = []
        for target in all_target_list:
            #Skipping the lower order marginals
            if len(target[0])<len(self.base_network.topo_i2n):
                continue
            
            #Convert the target to point
            point = {
                    self.base_network.topo_i2n[tidx]:cidx
                        for tidx,cidx in zip(target[0],target[1])
            }
            all_point_list.append(point)

        #Now we will create the full matrix
        A = np.zeros((len(all_point_list),len(all_target_list)))
        b = np.zeros(len(all_point_list))
        #Filling the matrix
        for ridx in range(A.shape[0]):
            point = all_point_list[ridx]

            #Filling the entry in the b_matrix
            pmix = self.dist_handler.get_mixture_probability(point,infinte_sample=False)
            p    = self.dist_handler.get_intervention_probability(
                                                    sample=point,
                                                    eval_do=[(),()]
            )
            b[ridx] = pmix - p

            #Filling up the columns of the matrix
            for cidx in range(A.shape[1]):
                #Getting the entry for this location
                target = all_target_list[cidx]
                print("Filling up the A: point:{}\ttarget:{}".format(point,target))
                p_t = self.dist_handler.get_intervention_probability(
                                                sample = point,
                                                eval_do = [target[0],target[1]]
                )
                
                A[ridx,cidx]= p_t - p
        
        #Getting the actual mixxing coefficient dict
        actual_target_dict={key:0.0 for key in all_target_dict.keys()}
        for tnodes,tcats,atpi in self.dist_handler.do_config:
            actual_target_dict[(tuple(tnodes),tuple(tcats))] = atpi
        print("Actual Target Dict")
        pprint(actual_target_dict)
        print("All Target List")
        pprint(all_target_list)
        
        return A,b,all_target_list,actual_target_dict
    
    def _get_all_exclusion_config(self,):
        '''
        This will generate all possible ways we could ipose paulis exclusion 
        assumption.
        '''
        #Generating all possible assignment of this target nodes
        cidx_list=[range(
                self.base_network.card_node[self.base_network.topo_i2n[nidx]]
            )
            for nidx in range(len(self.base_network.topo_i2n))
        ]
        #Getting all setting of all these nodes in this target
        zero_settings = it.product(*cidx_list)

        #Creting all possible zero config
        zero_config_list=[]
        for setting in zero_settings:
            config = {
                        nidx:setting[nidx] 
                            for nidx in range(len(self.base_network.topo_i2n))
            }
            #Appending to the zero config list
            zero_config_list.append(config)
        
        return zero_config_list

def em_step_worker_kernel(config):
    '''
    This function is the worker kernel which will be run by each of the example
    individually to get the strength of the mixing coefficient from their perspective
    and return the posteriaor porbability from of each mixing coefficient wrt to them.
    '''
    #Getting the required metadata to be used for finding the posterior porb
    all_target_keys = config["all_target_keys"]
    all_target_pi   = config["all_target_pi"]
    point           = config["point"]
    dist_handler    = config["dist_handler"]

    #Getting the posterior porbability
    #get the probability of this sample on all targets
    sample_posterior_pi = all_target_pi * 0.0
    for tidx,target in enumerate(all_target_keys):
        tprob = dist_handler.get_intervention_probability(
                                    sample=point,
                                    eval_do=list(target)
        )
        #Updating the prob in sample _posterior_pi
        sample_posterior_pi[tidx]=tprob

    #Next we multiply the pis and normalize
    sample_posterior_pi = sample_posterior_pi*all_target_pi
    sample_posterior_pi = sample_posterior_pi/np.sum(sample_posterior_pi)

    return sample_posterior_pi,dist_handler

def solve_for_a_exclusion_config_worker_kernel(worker_config):
    '''
    Given a particular exclusion setting, we want to solve the system of equation and
    get the final result.

    exclusion_config = {
                            nidx : blacklisted_cidx
    }
    '''
    exclusion_config    = worker_config["exclusion_config"]
    zero_eps            = worker_config["zero_eps"]
    A                   = worker_config["A"]
    b                   = worker_config["b"]
    all_target_list     = worker_config["all_target_list"]
    actual_target_dict  = worker_config["actual_target_dict"]

    print("\n\n======================================")
    print("Starting new run for zero config:")
    pprint(exclusion_config)

    def optimization_func(x,):
        #Getting the residual
        residual = np.sum((np.matmul(A,x)-b)**2)

        #Calculating the mse too
        step_mse = 0.0
        for tidx,target in enumerate(all_target_list):
            atpi = actual_target_dict[target]
            ptpi = x[tidx]

            step_mse += (atpi-ptpi)**2
        step_mse = step_mse/x.shape[0]

        #print("residual:{:0.3f}\tstep_mse:{:0.3f}".format(residual,step_mse))

        return residual 
    
    def get_excluded_target_pi_sum(x,):
        #Getting the index of target where the excluded category exist
        pi_sum = 0.0
        for ctidx,target in enumerate(all_target_list):
            for tnidx,tcat in zip(target[0],target[1]):
                if exclusion_config[tnidx]==tcat:
                    pi_sum+=x[ctidx]
                    break
        
        #print("eclusion violation level: {}".format(pi_sum))
        return zero_eps-pi_sum
    
    def get_sum_of_pis(x,):
        pi_sum = np.sum(x)
        #print("total pi sum: {}".format(pi_sum))

        return 1-pi_sum
    
    #Now we will define the constraints
    constraints = [
                        {
                            "type":"ineq",
                            "fun": get_sum_of_pis
                        },
                        {
                            "type":"ineq",
                            "fun": get_excluded_target_pi_sum
                        }
    ]
    bounds = [(0.0,1.0)]*A.shape[1]

    #Running the optimizer
    pi_trial = np.ones((A.shape[1],))/A.shape[1]
    opt_result = minimize(optimization_func,pi_trial,
                            method = "SLSQP",
                            bounds=bounds,
                            constraints=constraints
    )

    pprint(opt_result)

    result_dict = dict(
                    zero_config = exclusion_config,
                    opt_result=opt_result,
    )
    return result_dict


if __name__=="__main__":
    num_nodes=4
    node_card=4
    #Creating a random graph
    from graph_generator import GraphGenerator
    generator_args={}
    generator_args["scale_alpha"]=5
    graphGenerator = GraphGenerator(generator_args)
    modelpath = graphGenerator.generate_bayesian_network(num_nodes=num_nodes,
                                            node_card=node_card,
                                            num_edges=50,
                                            graph_type="SF")
    base_network=BnNetwork(modelpath)
    # pdb.set_trace()

    #Getting a random internvetion
    max_target_order = num_nodes
    target_generator = InterventionGenerator(S=16,
                                            max_nodes=num_nodes,
                                            max_cat=node_card,
                                            max_target_order=max_target_order,
                                            num_node_temperature=float("inf"),
                                            pi_dist_type="inverse",
                                            pi_alpha_scale=5)
    do_config = target_generator.generate_all_targets()

    #Testing by having the sample from mixture distribution
    infinite_sample_limit=False
    mixture_sample_size=float("inf")
    mixture_samples=None
    if not infinite_sample_limit:
        mixture_sample_size=10000
        mixture_samples = base_network.generate_sample_from_mixture(
                                            do_config=do_config,
                                            sample_size=mixture_sample_size)

    #Testing the general mixture solver
    solver = GeneralMixtureSolver(
                            base_network=base_network,
                            do_config=do_config,
                            infinite_sample_limit=infinite_sample_limit,
                            base_samples=None,
                            mixture_samples=mixture_samples,
                            pi_threshold=(1/16.0)*(0.25),
                            split_threshold=(-1e-10),
                            positivity_epsilon=1.0/mixture_sample_size,
                            positive_sol_threshold=1e-10,
            )
    
    #Solving using the EM algorithm
    # pred_target_dict_em,mse_overall_list,avg_logprob_list=solver.solve_by_em(
    #                             max_target_order=max_target_order,
    #                             epochs=30,
    #                             log_epsilon=1e-10,
    #                             num_parallel_calls=4
    # )
    #Plotting the evaluation metrics
    # plt.plot(range(len(mse_overall_list)),mse_overall_list,"o-")
    # plt.show()
    # plt.close()
    # plt.plot(range(len(avg_logprob_list)),avg_logprob_list,"o-")
    # plt.show()


    #Zero Threshold : The ones below this will be considered as worn out!
    zero_threshold = 1e-3

    #Solving using the Brute Force algorithm
    pred_target_dict_brute = solver.solve_by_brute_force_sys_eq(
                                                        zero_eps=zero_threshold, 
                                                        num_parallel_calls=4
    )
    print("Actual Balcklist Categories:")
    pprint(target_generator.blacklist_categories)



    #Now we will solve the mixture via our methods
    pred_target_dict_ours = solver.solve()


    #Now lets evaluate the solution
    evaluator = EvaluatePrediction(matching_weight=0.5)
    #Thresholding the predicted dicts
    pred_target_dict_brute = evaluator.threshold_target_dict(pred_target_dict_brute,zero_threshold)
    pred_target_dict_ours  = evaluator.threshold_target_dict(pred_target_dict_ours,zero_threshold)


    #Getting the evaluation score
    evaluation_brute = evaluator.get_evaluation_scores(pred_target_dict_brute,do_config)
    evaluation_ours = evaluator.get_evaluation_scores(pred_target_dict_ours,do_config)
    
    
    print("===================================")
    print("Brute Result:")
    pprint(evaluation_brute)
    print("===================================")
    print("Ours Result:")
    pprint(evaluation_ours)
    print("===================================")


