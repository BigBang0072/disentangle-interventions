import numpy as np
np.random.seed(1)
import itertools
from pprint import pprint
import json
import multiprocessing as mp
import pathlib
import pdb
import random
import sys
import time

from graph_generator import GraphGenerator
from data_handle import BnNetwork
from non_overlap_intv_solver import DistributionHandler
from interventionGenerator import InterventionGenerator
from evaluator import EvaluatePrediction
from general_mixture_solver import GeneralMixtureSolver


class GeneralMixtureJobber():
    '''
    This class will be responsible for running the whole experiment.
    Problem Generation:
        1. Generate random BN
            1.1 Graph Type [SF, ER] *4
            1.2 num_nodes   *4
            1.3 node_card   *4
            1.4 num_edges   *4
            1.5 scale_alpha *4

        2. Generate Random Intervention
            2.1 Sparsity
            2.2 num of nodes in a target temperature
            2.3 Distribution type of the number of nodes
            2.4 Scale alpha for pi distribution

        3. Generate Random Mixture of Intervention
            3.1 Sparsity    [See how sample size varies this]
            3.2 Sample Size [actual metric of variation]
            3.3 pi_threshold [as a prior guess will be 1/|S| or 1/|2S|]

    Evaluation Criteria:
        1. (1/3)JS_node + (2/3)JS_overall

    '''
    def __init__(self,graph_args,interv_args,mixture_args,eval_args):
        #Initializing the flattened args
        self.flatten_args_key=[]
        self.flatten_args_val=[]

        #Initialing the evaluation ards
        self._generate_flatten_args(graph_args)
        self._generate_flatten_args(interv_args)
        self._generate_flatten_args(eval_args)

        #Now we wont outer product the sample size instead on same graph
        #We will run the expt for different sample size
        self.mixture_sample_sizes=tuple(mixture_args["mixture_sample_size"])

        #Now we will flatten all these arguments into list of problem instance
        self._get_all_problem_config()
        #pprint(self.all_problem_config)
        print("Total Number of jobs:{}".format(len(self.all_problem_config)))

    def _generate_flatten_args(self,arg_dict):
        '''
        '''
        for key,val in arg_dict.items():
            self.flatten_args_key.append(key)
            self.flatten_args_val.append(val)

    def _get_all_problem_config(self,):
        #Generating the names of all the key with index val
        self.flatten_i2a={idx:name
                            for idx,name in enumerate(self.flatten_args_key)}
        self.flatten_a2i={name:idx
                            for idx,name in self.flatten_i2a.items()}

        #First lets flatten all the problem instance
        problem_config_temp=list(
                            itertools.product(*self.flatten_args_val))
        #Now we have to generate different edge instance of the problem
        all_problem_config=[]
        for config in problem_config_temp:
            #Now we will create the problem dict
            config_dict={key:value
                            for key,value in zip(self.flatten_args_key,config)
                        }

            #Getting the number of nodes
            num_nodes       =   config_dict["num_nodes"]
            num_edge_sample =   config_dict["num_edge_sample"]
            num_edges_dist  =   config_dict["num_edges_dist"]
            #Now we will ge the number of edges
            edge_sample =   self._get_num_edges(num_nodes,num_edge_sample,
                                                    num_edges_dist)


            #Now we will update the num_edge_sample with num_edge
            for num_edge in edge_sample:
                #Adding the edge number
                new_config = config_dict.copy()
                new_config["num_edges"]=num_edge

                #Also adding all the sample sizes over which we have to run
                new_config["all_sample_sizes"]=self.mixture_sample_sizes

                #Adding this config to all problem list
                all_problem_config.append(new_config)

        #Saving all the problem list
        # pdb.set_trace()
        self.all_problem_config=all_problem_config

    def _get_num_edges(self,num_nodes,num_sample,num_dist):
        '''
        This fucntion will sample number of edges based on number of nodes
        '''
        edge_sample=None
        if num_dist=="uniform":
            #max_edges = (num_nodes*(num_nodes-1))//2
            min_edges = 1*num_nodes
            max_edges = min( (num_nodes*(num_nodes-1))//2 , 5*num_nodes)
            edge_sample = np.random.randint(min_edges,high=max_edges,
                                                size=num_sample)
        else:
            raise NotImplementedError

        return edge_sample.tolist()

    def run_job_parallely(self,experiment_id):
        '''
        This function will run the jobs parallely and save the result in the
        folder names experiment_id
        '''
        job_ids=list(range(len(self.all_problem_config)))
        exp_ids=[experiment_id]*len(self.all_problem_config)
        #Getting the problem list
        problem_list=list(zip(self.all_problem_config,
                                exp_ids,
                                job_ids,
                        )
        )
        random.shuffle(problem_list)
        random.shuffle(problem_list)

        #Running the jobs
        # num_cpu = mp.cpu_count()//2
        # with mp.Pool(num_cpu) as p:
        #     job_results = p.map(jobber,problem_list)
        # return job_results

        #Saving the actual stdout
        sysout = sys.stdout
        syserr = sys.stderr

        #Defining the worker kernel
        def worker_kernel(problem_list_shard,widx):
            #Iterating over all the problems in the list one by one
            for problem_args in problem_list_shard:
                #Now opening a new context for each of the problem
                _,expt_id,job_id=problem_args
                log_fname = "{}/logs_job-{}.txt".format(expt_id,job_id)
                with open(log_fname,"w") as log_port:
                    sys.stdout = sysout
                    sys.stderr = syserr
                    print("Saving logs for worker:{}\t job:{}\t to:{}".format(
                                                        widx,
                                                        job_id,
                                                        log_fname)
                    )
                    #Conneting the print pipe to log port
                    sys.stdout = log_port
                    sys.stderr = log_port
                    print("Starting in a new logfile for job:{}".format(job_id))
                    #Running the jobber for that problem
                    jobber(problem_args)

            return True

        #Sharding the problem_list and running them parallely
        num_cpu = (mp.cpu_count()//4)*3
        num_per_worker = int(np.ceil(len(problem_list)*1.0/num_cpu))
        process_list=[]
        for widx in range(num_cpu):
            #Sharding the problem list
            print("Worker Initialized:{}\tstart:{}\tend:{}".format(widx,
                                                    widx*num_per_worker,
                                                    (widx+1)*num_per_worker,
                                                )
                )
            problem_list_shard=problem_list[widx*num_per_worker:(widx+1)*num_per_worker]

            #Starting the process with worker kernel
            p = mp.Process(target=worker_kernel,
                            args=(problem_list_shard,widx),
                )
            p.start()
            process_list.append(p)

        #Now joining all the process
        [p.join() for p in process_list]


def jobber(problem_args):
    #Saving all the results in one place
    problem_config,experiment_id,job_id = problem_args
    print("==========================================================")
    print("==========================================================")
    print("==========================================================")
    print("exp_id:{} job_id:{}".format(experiment_id,job_id))
    pprint(problem_config)

    #Now let generate the problem instance
    generator_args={}
    generator_args["scale_alpha"]=problem_config["scale_alpha"]
    graphGenerator  =   GraphGenerator(generator_args)
    modelPath       =   graphGenerator.generate_bayesian_network(
                                num_nodes=problem_config["num_nodes"],
                                node_card=problem_config["node_card"],
                                num_edges=problem_config["num_edges"],
                                graph_type=problem_config["graph_type"],
    )
    #Now generating the bayesian network
    base_network    =   BnNetwork(modelPath)
    base_graph      =   base_network.base_graph.copy()




    #Generating the random internvention configuration
    target_generator    =   InterventionGenerator(
                                S=problem_config["sparsity"],
                                max_nodes=problem_config["num_nodes"],
                                max_cat=problem_config["node_card"],
                                num_node_temperature=problem_config["num_node_T"],
                                pi_dist_type=problem_config["pi_dist_type"],
                                pi_alpha_scale=problem_config["pi_alpha_scale"],
    )
    do_config           =   target_generator.generate_all_targets()

    #Now one by one we will try to run the problem for all the sample size
    for mixture_sample_size,pi_threshold_scale in problem_config["all_sample_sizes"]:
        #Adding that mixture size to the problem config
        problem_config["mixture_sample_size"]=mixture_sample_size
        problem_config["pi_threshold_scale"]=pi_threshold_scale
        #Resetting the base graph with actual distribution
        base_network.base_graph=base_graph.copy()

        #Now running the problem
        try:
            problem_config=jobber_runner(problem_config,
                                            base_network,
                                            do_config.copy())
        except Exception as e:
            print("Job terminated wrongly:")
            print(e)
            pprint(problem_args)
            problem_config["js_score"]=np.nan
            problem_config["avg_mse"]=np.nan


        #Writing the file to system
        fname   =   "{}/j{}_s{}_t{}.json".format(
                                        experiment_id,
                                        job_id,
                                        mixture_sample_size,
                                        pi_threshold_scale)
        with open(fname,"w") as fp:
            json.dump(problem_config,fp,indent="\t")


def jobber_runner(problem_config,base_network,do_config):
    '''
    This function will be called by each of the process to run the problem
    and get the result and save it in the JSON format.
    '''

    #Now we will generate the mixture data
    infinite_sample_limit   =   False
    mixture_samples         =   None
    mixture_sample_size     =   problem_config["mixture_sample_size"]
    print("Getting Mixture Sample")
    if ( mixture_sample_size == float("inf") ):
        infinite_sample_limit   =  True
    else:
        mixture_samples     = base_network.generate_sample_from_mixture(
                                    do_config=do_config,
                                    sample_size=mixture_sample_size
        )



    #Now we are ready to trigger the solver
    print("Solving the problem")
    #Prior knowledge in threshold
    # pi_threshold        =   problem_config["pi_threshold_scale"]\
    #                         *(1.0/len(do_config))

    #From now on we will be interested in absolute threshold
    pi_threshold            =   problem_config["pi_threshold_scale"]
    positivity_epsilon      =   problem_config["positivity_epsilon"]
    positive_sol_threshold  =   problem_config["positive_sol_threshold"]

    if positivity_epsilon=="one_by_sample_size":
        positivity_epsilon = 1.0/mixture_sample_size

    if infinite_sample_limit:
        pi_threshold=   1e-10   #Dont mess up with me here

    solver          =   GeneralMixtureSolver(
                            base_network=base_network,
                            do_config=do_config,
                            infinite_sample_limit=infinite_sample_limit,
                            base_samples= None,
                            mixture_samples=mixture_samples,
                            pi_threshold=pi_threshold,
                            split_threshold=problem_config["split_threshold"],
                            positivity_epsilon=positivity_epsilon,
                            positive_sol_threshold=positive_sol_threshold
    )
    #Solving the problem using em algorithm first
    em_start_time = time.time()
    pred_target_dict_em,mse_overall_list,avg_logprob_list=solver.solve_by_em(
                                max_target_order=problem_config["num_nodes"],
                                epochs=problem_config["num_em_epochs"],
                                log_epsilon=1e-10,
    )
    em_end_time = time.time()

    #Saving the results from the em into our problem config
    problem_config["pred_target_dict_em"] = pred_target_dict_em
    problem_config["mse_overall_list_em"] = mse_overall_list
    problem_config["avg_logprob_list_em"] = avg_logprob_list
    problem_config["em_execution_time"]   = em_end_time - em_start_time

    #Getting the prediction
    ours_start_time = time.time()
    pred_target_dict    =   solver.solve()
    ours_end_time  = time.time()

    #Now its time to evaluate
    evaluator           =   EvaluatePrediction(
                                matching_weight=problem_config["matching_weight"]
    )
    recall,_,avg_mse   =   evaluator.get_evaluation_scores(
                                pred_target_dict,
                                do_config
    )
    #Saving the actual and predicted target dict for later evaluations
    problem_config["pred_target_dict_ours"]=pred_target_dict
    problem_config["actual_target_dict"]=evaluator.actual_target_dict
    problem_config["ours_execution_time"] = ours_end_time - ours_start_time

    # if infinite_sample_limit:
    #     assert avg_score==1.0,"Problem in infinite sample limit"

    #Now we will save the config and the scores as a json file
    problem_config["js_score"]  =   recall
    problem_config["avg_mse"]   =   avg_mse

    return problem_config


if __name__=="__main__":
    #Initializing the graph args
    graph_args={}
    graph_args["graph_type"]=["SF"]
    graph_args["num_nodes"]=[4]
    graph_args["node_card"]=[3]
    graph_args["num_edges_dist"]=["uniform"] #dist to sample edge from
    graph_args["num_edge_sample"]=[1] #number of random edge per config
    graph_args["scale_alpha"]=[2]

    #Initializing the sparsity args
    interv_args={}
    interv_args["sparsity"]= np.random.randint(4,high=16,size=100).tolist()
    interv_args["num_node_T"]=[float("inf")]
    interv_args["pi_dist_type"]=["uniform"]
    interv_args["pi_alpha_scale"]=[2]

    #Initializing the sample distribution
    mixture_args={}
    mixture_args["mixture_sample_size"]= list(itertools.product(*
                                        [
                                            (2**np.arange(4,10)).tolist()+[float("inf")],
                                            [1e-3]
                                        ]
                                ))

    #Evaluation args
    eval_args={}
    eval_args["num_em_epochs"]=[30]       #Number of epochs to run the EM
    eval_args["matching_weight"]=[1.0/3.0]
    eval_args["split_threshold"]=[-1e-10]
    eval_args["positivity_epsilon"]=["one_by_sample_size"]
    eval_args["positive_sol_threshold"]=[-1e-10]

    #Now we are ready to start our experiments
    experiment_id="exp_patent_1.0"
    pathlib.Path(experiment_id).mkdir(parents=True,exist_ok=True)
    shantilal = GeneralMixtureJobber(graph_args,interv_args,mixture_args,eval_args)
    shantilal.run_job_parallely(experiment_id)
