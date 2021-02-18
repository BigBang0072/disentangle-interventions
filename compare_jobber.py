import numpy as np
import pandas as pd
import random
import pdb


import causaldag as cd
from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester



from graph_generator import GraphGenerator
from data_handle import BnNetwork
from non_overlap_intv_solver import DistributionHandler
from interventionGenerator import InterventionGenerator
from evaluator import EvaluatePrediction
from general_mixture_solver import GeneralMixtureSolver

class CompareJobber():
    '''
    We will compare the UT-IGSP and our Algorithm to knock the hell out of
    everyone.
    '''
    def __init__(self,graph_args,interv_args,eval_args):
        #Setting the Graph related args
        self.graph_args=graph_args
        #Iniitlaizing the graph generator
        self.graph_generator=GraphGenerator(graph_args)
        #Get ready with the graphs
        self.generate_graph(graph_args)

        #Setting the intervention args
        self.interv_args=interv_args
        self.generate_u_targets(graph_args,interv_args)

        #Setting the evaluation args
        self.eval_args=eval_args

    def generate_graph(self,graph_args):
        #Sampling the graph from Uhler's method
        print("Generating the Uhler's Graph")
        u_adj_mat=cd.rand.directed_erdos(
                        graph_args["num_nodes"],
                        graph_args["expt_nbrs"]/((graph_args["num_nodes"]-1)*1.0),
        )
        self.u_network = cd.rand.rand_weights(u_adj_mat)

        #Now converting the graph to our required format
        o_adj_mat = np.zeros([graph_args["num_nodes"]]*2,dtype=np.int32)
        #Now adding the entry one by one for all source
        for fidx in range(graph_args["num_nodes"]):
            #Getting all the childern
            all_child = u_adj_mat.children_of(fidx)
            for tidx in all_child:
                o_adj_mat[tidx][fidx]=1

        #Now we will use the intervention generateor to get the bayesian net
        print("Generating our network")
        o_network_path = self.graph_generator.generate_bayesian_network(
                                        num_nodes=graph_args["num_nodes"],
                                        node_card=graph_args["node_card"],
                                        num_edges=None,
                                        graph_type=o_adj_mat,
        )
        self.o_network    =   BnNetwork(o_network_path)



    def generate_u_targets(self,graph_args,interv_args):
        '''
        This will generate the targets to be used by uhlers method
        '''
        #Generating the random targets (node, they have assumed size same)
        print("Generating Uhler's target list")
        self.u_target_list = [random.sample(
                                list(range(graph_args["num_nodes"])),
                                interv_args["target_size"]
            ) for _ in range(interv_args["sparsity"])]

    def compare_given_sample_size(self,sample_size):
        '''
        We will be using the perfect internvetion, since our method is also
        based on the perfect internvetion, which could b a problem if we use
        the shift intervention.
        '''
        #Sampling the base distribution (from the u_graph)
        print("Generating base sample: size:{}".format(sample_size))
        base_samples = self.u_network.sample(sample_size)

        #Now we will generrate all the intervention targets location
        target_locs = [{
                        target_node: cd.GaussIntervention(
                            interv_args["iv_mean"],
                            interv_args["iv_var"]
                        )
                        for target_node in target
                    } for target in self.u_target_list
        ]
        #Generate all the intervention sample seperately
        print("Generating the intervention samples")
        target_sample_list = [
                    self.u_network.sample_interventional(loc,sample_size)

            for loc in target_locs
        ]

        #Now its time to solve the the problem using Uhler's approach
        pred_u_target_list = self.solve_uhler_approach(base_samples,
                                                    target_sample_list)

        #Now we will be solving with our approach
        pred_o_target_dict = self.solve_our_approach(base_samples,
                                                    target_sample_list)

        #

        pdb.set_trace()

    def solve_uhler_approach(self,base_samples,target_sample_list):
        '''
        This will be used to solve the problem using the uher's approach
        '''
        #Getting suffieient statistics
        base_suffstat = partial_correlation_suffstat(base_samples)
        invariance_suffstat = gauss_invariance_suffstat(base_samples,
                                                target_sample_list)

        #Getting the CI tester and invariance tester
        ci_tester = MemoizedCI_Tester(partial_correlation_test,
                                        base_suffstat,
                                        alpha=self.eval_args["u_alpha"])
        invariance_tester = MemoizedInvarianceTester(gauss_invariance_test,
                                        invariance_suffstat,
                                        alpha=self.eval_args["u_alpha_inv"])

        #Running their algorithm
        setting_list = [dict(known_interventions=[]) for _ in self.u_target_list]
        print("Solving the problem with Uhler's Method")
        _,pred_u_target_list = cd.unknown_target_igsp(setting_list,
                                                    set(range(self.graph_args["num_nodes"])),
                                                    ci_tester,
                                                    invariance_tester
                            )
        #Saving the predicted target list
        return pred_u_target_list

    def solve_our_approach(self,base_samples,target_sample_list):
        '''
        This function will be used to make prediction from our method
        '''
        #The first task is to discretize the nodes
        print("Binning the continuous sample")
        bin_base_samples,bin_mixture_samples = self._discretize_dataset(
                                                        base_samples,
                                                        target_sample_list
        )
        #Now we have to get the do config once

        #Now we are ready to call our solver
        print("Solving the problem with our method")
        #From now on we will be interested in absolute threshold
        infinite_sample_limit   =   False #Only work with finite numb of sample
        pi_threshold            =   self.eval_args["pi_threshold"]
        split_threshold         =   self.eval_args["split_threshold"]
        positivity_epsilon      =   self.eval_args["positivity_epsilon"]
        positive_sol_threshold  =   self.eval_args["positive_sol_threshold"]

        if positivity_epsilon=="one_by_sample_size":
            positivity_epsilon = 1.0/mixture_sample_size

        if infinite_sample_limit:
            pi_threshold=   1e-10   #Dont mess up with me here
        #Iniitlizing the solver
        solver          =   GeneralMixtureSolver(
                                base_network=self.o_network,
                                do_config=None,#Dont want inf sample prob
                                infinite_sample_limit=infinite_sample_limit,
                                base_samples=bin_base_samples,
                                mixture_samples=bin_mixture_samples,
                                pi_threshold=pi_threshold,
                                split_threshold=split_threshold,
                                positivity_epsilon=positivity_epsilon,
                                positive_sol_threshold=positive_sol_threshold
        )
        #Getting the prediction
        pred_target_dict    =   solver.solve()

        return pred_target_dict

    def _discretize_dataset(self,base_samples,target_sample_list):
        '''
        Here we will discretize the dataset
        '''
        #Converting to one big dataframe
        all_samples=pd.DataFrame(
                        np.concatenate([base_samples]+target_sample_list,axis=0)
                    )
        #Now cutting the data frame index by index
        for nidx in range(all_samples.shape[-1]):
            #Now lets cut this index
            all_samples[nidx] = pd.cut(all_samples[nidx],
                                        bins=graph_args["node_card"],
                                        labels=False,
                                )

        #Now its time to separate out the whole data frame as numpy
        all_binned_sample=all_samples.to_numpy()

        #Now we will remove out the base sample and keep the rest of sample
        binned_base_samples = all_binned_sample[0:base_samples.shape[0]].copy()
        binned_mixture_samples = all_binned_sample[base_samples.shape[0]:].copy()

        assert (
            (binned_base_samples.shape[0]+binned_mixture_samples.shape[0])\
            == (base_samples.shape[0]* (len(target_sample_list)+1))
        )

        return binned_base_samples, binned_mixture_samples


if __name__=="__main__":
    #Defininte the graph args
    graph_args={}
    graph_args["num_nodes"]=10
    graph_args["node_card"]=8
    graph_args["expt_nbrs"]=2
    graph_args["scale_alpha"]=2

    #Defining the intervention args
    interv_args={}
    interv_args["sparsity"]=4
    interv_args["target_size"]=3 #TODO: Maybe we use same graph
    interv_args["iv_mean"]=1
    interv_args["iv_var"]=0.1

    #Defining the eval args
    eval_args={}
    #Uhlers configs
    eval_args["u_alpha"]=1e-3
    eval_args["u_alpha_inv"]=1e-3
    #Ours configs
    eval_args["pi_threshold"]=1e-3
    eval_args["matching_weight"]=[1.0/3.0]
    eval_args["split_threshold"]=[-1e-10]
    eval_args["positivity_epsilon"]=["one_by_sample_size"]
    eval_args["positive_sol_threshold"]=[-1e-10]

    #Testing the Uhlers job
    CJ = CompareJobber(graph_args,interv_args,eval_args)
    CJ.compare_given_sample_size(1000)
