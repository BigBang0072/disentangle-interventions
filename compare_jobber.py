import numpy as np
np.random.seed(1)
import pandas as pd
import random
import pdb
from pprint import pprint
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import json
import pathlib


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

        #Initializing the result dict
        self.result_dict={}
        self.prediction_dict={}

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

        #Now we will generrate all the intervention targets location
        self.target_locs = [{
                        target_node: cd.GaussIntervention(
                            interv_args["iv_mean"],
                            interv_args["iv_var"]
                        )
                        for target_node in target
                    } for target in self.u_target_list
            ]

    def compare_given_sample_size(self,mode,sample_size):
        '''
        We will be using the perfect internvetion, since our method is also
        based on the perfect internvetion, which could b a problem if we use
        the shift intervention.
        '''
        #Sampling the base distribution (from the u_graph)
        print("Generating base sample: size:{}".format(sample_size))
        base_samples = self.u_network.sample(sample_size)

        #Generate all the intervention sample seperately
        print("Generating the intervention samples")
        target_sample_list = [
                    self.u_network.sample_interventional(loc,sample_size)

            for loc in self.target_locs
        ]

        #Now its time to solve the the problem using Uhler's approach
        ufpr, upred_list = self.solve_uhler_approach(base_samples,
                                                    target_sample_list)

        #Now we will be solving with our approach
        ofpr, opred_list = self.solve_our_approach(base_samples,
                                                    target_sample_list,
                                                    mode,
                                                    sample_size)

        #Getting the evaluation metrics
        # self.evaluate_both_predictions(pred_u_target_list,
        #                                 pred_o_target_dict,
        #                                 sample_size)
        #Saving the result both in json and in result dict
        self.result_dict[sample_size]={"uhler":ufpr,"ours":ofpr}
        self.prediction_dict[sample_size]={"ours":opred_list,"uhler":upred_list}
        # pdb.set_trace()


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
        #Converting the set prediction to list
        pred_u_target_list=[[int(node) for node in pred] for pred in pred_u_target_list]

        #Calculating the false positive rate
        true_positive=0.0
        for aidx,actual_target in enumerate(self.u_target_list):
            if self._check_target_equality(actual_target,pred_u_target_list[aidx]):
                true_positive+=1

        false_positive_rate = self._calculate_false_positive_rate(
                                                    true_positive,
                                                    len(self.u_target_list)
                                        )
        return false_positive_rate,pred_u_target_list

    def solve_our_approach(self,base_samples,target_sample_list,mode,sample_size):
        '''
        This function will solve with our the three approach and then give
        the merged result at once.

        Case 1: Give the full mixture at once to solve
        Case 2: Full mixture:
                        target 1 ---> solve separately  \ merge result
                        target 2 ---> solve separately  /
                        ....
                        ....

        Case 3: Full Mixutre:
                        target 1 ---> fixed mean interv ---> solve \ merge
                        target 2 ---> fixed mean interv ---> solve /
                        ....
                        ....
        '''
        #The first task is to discretize the nodes
        print("Binning the continuous sample")
        bin_base_samples,bin_mixture_samples,bin_interval_dict = \
                                    self._discretize_dataset(
                                                        base_samples,
                                                        target_sample_list
        )

        #Now based on the mode we will work accordingly
        all_prediction_list=[]
        false_positive_rate=None
        if mode == "all":
            #Solve every thing at once
            pred = self.solve_our_approach_once(bin_base_samples,
                                                bin_mixture_samples)

            #Getting the top-k prediction
            top_k_pred = self._get_top_n_predictions(pred,len(self.u_target_list))
            all_prediction_list = top_k_pred
            #Now getting the recall value
            false_positive_rate = self._get_mixture_fpr(top_k_pred.copy())
            # pdb.set_trace()

        elif mode =="single":
            true_positive=0.0
            #Solving each sample one by one
            for tidx,target in enumerate(self.u_target_list):
                #Splicing the sample for this particular target
                tbin_sample = bin_mixture_samples[tidx*sample_size:(tidx+1)*sample_size]

                #Now we will solve each of them one by one
                pred = self.solve_our_approach_once(bin_base_samples,tbin_sample)

                #Getting the top prediction
                pred_target = self._get_top_n_predictions(pred,1)[0]
                all_prediction_list.append(pred_target)

                true_positive += self._check_target_equality(pred_target,target)
                # pdb.set_trace()

            #Now calculatin the false postitive rate
            false_positive_rate=self._calculate_false_positive_rate(true_positive,
                                                len(self.u_target_list))


        elif mode =="single_mean":
            true_positive=0.0
            #Solving each sample one by one
            for tidx,target in enumerate(self.u_target_list):
                #Splicing the sample for this particular target
                tbin_sample = bin_mixture_samples[tidx*sample_size:(tidx+1)*sample_size]

                #First of all we have to make a clean fixed point interv (mean)
                for nidx in target:
                    #The value we have to put is
                    # search_val=self.interv_args["iv_mean"]

                    #Now we have to find the bin for this guy (any one bin is fine)
                    bin_num =int(self.graph_args["node_card"]/2)
                    # bin_num=len(bin_intervals)-2 #default is last bin
                    # for bidx in range(0,bin_intervals.shape[0]-1):
                    #     if search_val<=bin_intervals[bidx+1]:

                    #Putting this random intervention everywhere in this sample
                    tbin_sample[str(nidx)]=bin_num


                #Now we will solve each of them one by one
                pred = self.solve_our_approach_once(bin_base_samples,tbin_sample)
                #Getting the top prediction
                pred_target = self._get_top_n_predictions(pred,1)[0]
                all_prediction_list.append(pred_target)

                true_positive += self._check_target_equality(pred_target,target)

            #Now calculatin the false postitive rate
            false_positive_rate=self._calculate_false_positive_rate(true_positive,
                                                len(self.u_target_list))
        else:
            raise NotImplementedError

        #Now its time to merge all the bins in once single place
        # pdb.set_trace()
        return false_positive_rate,all_prediction_list

    def _get_mixture_fpr(self,pred_list):
        true_positive = 0.0
        for actual_target in self.u_target_list:
            for pred_target in pred_list:
                #Check if they are same
                if self._check_target_equality(actual_target,pred_target):
                    true_positive+=1
                    #Removing the guy so that we dont recount
                    pred_list.remove(pred_target)
                    break

        #fpr
        false_positive_rate = self._calculate_false_positive_rate(
                                                true_positive,
                                                len(self.u_target_list)
                                    )
        return false_positive_rate

    def _calculate_false_positive_rate(self,true_pos,num_actual):
        return (num_actual-true_pos)*1.0/num_actual

    def _check_target_equality(self,pt1,pt2):
        '''
        '''
        if (set(pt1)==set(pt2)):
            return True
        return False

    def _get_top_n_predictions(self,pred,n):
        '''
        Given a merged prediction dictionary we willl give the top n candidate
        based on the mixing coefficient.
        '''
        pred_list = [(target,pi) for target,pi in pred.items()]
        pred_list.sort(reverse=True, key=lambda x: x[-1])

        return [pred_list[idx][0] for idx in range(min(n,len(pred_list)))]

    def solve_our_approach_once(self,bin_base_samples,bin_mixture_samples):
        '''
        This function will be used to make prediction from our method
        '''
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
            positivity_epsilon = 1.0/bin_base_samples.shape[0]

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
        pred_target_dict    =   self._simplify_our_prediction(solver.solve())

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
        bin_interval_dict={}
        for nidx in range(all_samples.shape[-1]):
            #Now lets cut this index
            all_samples[nidx],bins = pd.cut(all_samples[nidx],
                                        bins=graph_args["node_card"],
                                        labels=False,
                                        retbins=True
                                )
            bin_interval_dict[nidx]=bins


        #Now its time to separate out the whole data frame as numpy
        all_binned_sample=all_samples
        all_binned_sample.columns=[str(idx)
                                    for idx in range(all_samples.shape[1])
                        ]

        # pdb.set_trace()
        #Now we will remove out the base sample and keep the rest of sample
        binned_base_samples = all_binned_sample.iloc[0:base_samples.shape[0]].copy()
        binned_mixture_samples = all_binned_sample.iloc[base_samples.shape[0]:].copy()

        assert (
            (binned_base_samples.shape[0]+binned_mixture_samples.shape[0])\
            == (base_samples.shape[0]* (len(target_sample_list)+1))
        )

        return binned_base_samples, binned_mixture_samples,bin_interval_dict

    def _simplify_our_prediction(self,pred_target_dict):
        '''
        Lets first merge all the prediction which have the same node as one
        and get thier mixing coefficient also
        '''
        simple_pred_target_dict={}
        for tname,(tnode,_,tpi) in pred_target_dict.items():
            if tuple(tnode) in simple_pred_target_dict:
                simple_pred_target_dict[tuple(tnode)]+=tpi
            else:
                simple_pred_target_dict[tuple(tnode)]=tpi

        return simple_pred_target_dict

    def evaluate_both_predictions(self,ulist,odict,sample_size):
        '''
        This function will get the recall for both the methods
        '''
        #Lets first hash all the targets
        actual_target_dict = {"at{}".format(tidx):set(target)
                                for tidx,target in enumerate(self.u_target_list)
                            }
        pred_uhler = {
                            "ut{}".format(tidx):set(target)
                            for tidx,target in enumerate(ulist)
        }
        pred_ours = {
                            "ot{}".format(tidx):set(target)
                            for tidx,target in enumerate(odict.keys())
        }

        #Get the recall of a prediction
        def get_recall(actual,pred):
            #Cehckin for all the targets(its possible that two target is same)
            true_pos=0.0
            for _,atarget in actual.items():
                #which of the target is present in the prediction
                for _,ptarget in pred.items():
                    if(atarget==ptarget):
                        true_pos+=1.0
                        break

            recall = true_pos / len(actual)
            return recall

        recall_uhler = get_recall(actual_target_dict,pred_uhler)
        recall_ours = get_recall(actual_target_dict,pred_ours)

        print("=========================\nActual Targets:")
        pprint(actual_target_dict)
        print("=========================\nUhler's Prediction:")
        pprint(pred_uhler)
        print("=========================\nOur's Prediction:")
        pprint(pred_ours)

        print("uhler:{}\tours:{}".format(recall_uhler,recall_ours))
        # pdb.set_trace()
        #Saving the result
        self.result_dict[sample_size]={"uhler":recall_uhler,"ours":recall_ours}


def job_runner(experiment_id,worker_id,
                graph_args,interv_args,eval_args,mode,
                all_sample_sizes,num_job,child_conn):
    result_list=[]
    for jidx in range(num_job):
        result_list.append(
                        job_kernel(experiment_id,
                                    worker_id,
                                    jidx,
                                    graph_args,
                                    interv_args,
                                    eval_args,
                                    mode,
                                    all_sample_sizes
            )
        )
    child_conn.send(result_list)
    child_conn.close()


def job_kernel(experiment_id,worker_id,job_id,
                graph_args,interv_args,eval_args,
                mode,all_sample_sizes):
    # modes=["all","single","single_mean"]
    CJ = CompareJobber(graph_args,interv_args,eval_args)
    for sample_size in all_sample_sizes:
        CJ.compare_given_sample_size(mode,sample_size)

    #Creating payload to dump in json
    dump_payload={}
    dump_payload["false_positive_rate"]=CJ.result_dict
    dump_payload["all_prediction_list"]=CJ.prediction_dict
    dump_payload["actual_target_list"]=CJ.u_target_list
    dump_payload["mode"]=mode
    dump_payload["graph_args"]=graph_args
    dump_payload["interv_args"]=interv_args
    dump_payload["eval_args"]=eval_args

    fname = "{}/w{}_j{}.json".format(experiment_id,worker_id,job_id)
    with open(fname,"w") as fp:
        json.dump(dump_payload,fp,indent="\t")

    return CJ.result_dict

if __name__=="__main__":
    #Defininte the graph args
    graph_args={}
    graph_args["num_nodes"]=4
    graph_args["node_card"]=3
    graph_args["expt_nbrs"]=2
    graph_args["scale_alpha"]=2

    #Defining the intervention args
    interv_args={}
    interv_args["sparsity"]=3
    interv_args["target_size"]=2 #TODO: Maybe we use same graph
    interv_args["iv_mean"]=1
    interv_args["iv_var"]=0.1

    #Defining the eval args
    eval_args={}
    #Uhlers configs
    eval_args["u_alpha"]=1e-3
    eval_args["u_alpha_inv"]=1e-3
    #Ours configs
    eval_args["pi_threshold"]=1e-2*0.5
    eval_args["matching_weight"]=1.0/3.0
    eval_args["split_threshold"]=(-1e-10)
    eval_args["positivity_epsilon"]="one_by_sample_size"
    eval_args["positive_sol_threshold"]=(-1e-10)

    #Testing the Uhlers job
    num_random_job=10.0
    mode = "single_mean"        #["all","single","single_mean"] lets do one by one
    all_sample_sizes=(100,200,400,800,1600,3200,6400)
    all_result_list=[]
    experiment_id="comp-exp-1"
    pathlib.Path(experiment_id).mkdir(parents=True,exist_ok=True)

    # job_runner(0,0,graph_args,interv_args,eval_args,mode,all_sample_sizes,
    #             1,None)


    #=========================================================================
    #=========================== RUN JOB Parallel ============================
    #=========================================================================
    #Running the jobs parallely
    num_cpu = mp.cpu_count()//2
    all_process_list=[]
    all_pipe_list=[]
    for widx in range(num_cpu):
        print("Started Job:{}".format(widx))
        #Get number of jobs to run
        num_jobs = int(np.ceil(num_random_job/num_cpu))

        #Staring the pipe
        parent_conn, child_conn = mp.Pipe()
        all_pipe_list.append(parent_conn)

        p = mp.Process(target=job_runner,
                        args=(experiment_id,widx,
                                graph_args,interv_args,eval_args,mode,
                                all_sample_sizes,num_jobs,child_conn)
            )
        p.start()
        all_process_list.append(p)

    #Now joining all the process
    results=[parent_conn.recv() for parent_conn in all_pipe_list]
    [parent_conn.close() for parent_conn in all_pipe_list]
    [p.join() for p in all_process_list]

    all_result_list=[]
    for one_result in results:
        all_result_list= all_result_list+one_result
    print("Total Number of Job ran: {}".format(len(all_result_list)))



    #=========================================================================
    #=========================== PLOTTING ====================================
    #=========================================================================
    #Now we could merge the results in one single plot
    all_uhler_result = np.array([
                [result[sample_size]["uhler"] for sample_size in all_sample_sizes]
                for result in all_result_list
    ])
    all_our_result = np.array([
                [result[sample_size]["ours"] for sample_size in all_sample_sizes]
                for result in all_result_list
    ])

    fig,ax = plt.subplots()

    def plot_graph(result,curve_name):
        x=list(range(len(all_sample_sizes)))
        mean=np.mean(result,axis=0)
        q20 =np.quantile(result,0.2,axis=0)
        q80 =np.quantile(result,0.8,axis=0)

        ax.errorbar(x,mean,yerr=(mean-q20,q80-mean),fmt='o-',alpha=0.6,
                        capsize=5,capthick=2,linewidth=2,label=curve_name)
        ax.fill_between(x,q20,q80,alpha=0.1)


    plot_graph(all_uhler_result,"uhlers")
    plot_graph(all_our_result,"ours")

    xlabels=[str(size) for size in all_sample_sizes]
    ax.set_xticks(list(range(len(all_sample_sizes))))
    ax.set_xticklabels(xlabels,rotation=45)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0,1.05)
    plt.show()
