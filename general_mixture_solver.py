import numpy as np
import pdb
from pprint import pprint
from collection import defaultdict

from data_handle import BnNetwork
from non_overlap_intv_solver import DistributionHandler

class GeneralMixtureSolver():
    '''
    This class will be responsible for disentangling the general mixture of
    intervention with assumption that one category of each node is not present
    in the mixture at all.
    '''
    base_network=None           #Bayesian network corresponding to problem
    infinite_sample_limit=None  #boolean to denote to simulate infinite sample

    def __init__(self,base_network,do_config,
                infinite_sample_limit,mixture_samples):
        self.base_network=base_network
        self.infinite_sample_limit=infinite_sample_limit

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

        #Starting finding the merged targets topologically
        for vidx in range(len(self.base_network.topo_i2n)):
            v_bars,target_dict=self._split_all_targets(v_bars,vidx,target_dict)


    def _split_all_targets(self,v_bars,vidx,target_dict):
        '''
        This function will split the existing merged targets into new target
        by adding the possible internvetions present belonding to current node
        in those merged targets.
        '''
        #First of all we have to get the topological ordering among the targets
        tsorted_target_names=self._get_targets_topologically(target_dict)

        #Now we will go one by one in this order to split each of the target
        split_target_dict={}
        for tidx,tname in enumerate(tsorted_target_names):
            #Getting the all the targets which have been split before
            before_target_names = tsorted_target_names[0:tidx]
            self._split_target



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
        inv_adj_set=defaultdict(set)
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
        tsorted_target_names=toposort_flatten(inv_adj_set)

        return tsorted_target_names

    def _split_target(self,v_bars,vidx,tidx,target_dict,before_target_names,split_target_dict):
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
        curr_target = target_dict[tidx]

        #Now we will generate the system of equation for the current target
        node_id=self.base_network.topo_i2n[vidx]
        num_cats=self.base_network.card_node[node_id]
        #Initializing the system matrix
        A=np.zeros((num_cats,num_cats),dtype=np.float64)
        b=np.zeros((num_cats,),dtype=np.float64)

        #First of all we have to get the vt_bars U t' for current target
        vbtut=self._get_vbtut(v_bars,curr_target)

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
            A[cidx,:] = (-1*a)
            A[cidx,cidx] += 1

        #Now we are ready with our matrix





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
        This function will calcuate the b'' for the current cateogy
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
                                                    self.infinite_mix_sample)
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

    def
