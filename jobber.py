import numpy as np
np.random.seed(1)
import itertools
from pprint import pprint
import json
import multiprocessing as mp
import pathlib
import pdb

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
        self._generate_flatten_args(mixture_args)
        self._generate_flatten_args(eval_args)

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
                new_config = config_dict.copy()
                new_config["num_edges"]=num_edge
                #Adding this config to all problem list
                all_problem_config.append(new_config)

        #Saving all the problem list
        self.all_problem_config=all_problem_config

    def _get_num_edges(self,num_nodes,num_sample,num_dist):
        '''
        This fucntion will sample number of edges based on number of nodes
        '''
        edge_sample=None
        if num_dist=="uniform":
            max_edges = (num_nodes*(num_nodes-1))//2
            edge_sample = np.random.randint(1,high=max_edges,
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

        #Running the jobs
        num_cpu = mp.cpu_count()//2
        with mp.Pool(num_cpu) as p:
            job_results = p.map(jobber,problem_list)
        return job_results

def jobber(problem_args):
    return jobber_runner(problem_args)
    # try:
    #     return jobber_runner(problem_args)
    # except:
    #     pdb.set_trace()
    #     print("Job terminated wrongly:")
    #     pprint(problem_args)

    return None

def jobber_runner(problem_args):
    '''
    This function will be called by each of the process to run the problem
    and get the result and save it in the JSON format.
    '''
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
    pi_threshold    =   problem_config["pi_threshold_scale"]\
                            *(1.0/len(do_config))
    if infinite_sample_limit:
        pi_threshold=   1e-10   #Dont mess up with me here
    solver          =   GeneralMixtureSolver(
                            base_network=base_network,
                            do_config=do_config,
                            infinite_sample_limit=infinite_sample_limit,
                            mixture_samples=mixture_samples,
                            pi_threshold=pi_threshold,
                            split_threshold=problem_config["split_threshold"]
    )
    #Getting the prediction
    pred_target_dict    =   solver.solve()

    #Now its time to evaluate
    evaluator           =   EvaluatePrediction(
                                matching_weight=problem_config["matching_weight"]
    )
    avg_score,avg_mse   =   evaluator.get_evaluation_scores(
                                pred_target_dict,
                                do_config
    )

    if infinite_sample_limit:
        assert avg_score==1.0,"Problem in infinite sample limit"

    #Now we will save the config and the scores as a json file
    problem_config["js_score"]  =   avg_score
    problem_config["avg_mse"]   =   avg_mse
    #Writing the file to system
    fname   =   "{}/{}.json".format(experiment_id,job_id)
    with open(fname,"w") as fp:
        json.dump(problem_config,fp)

    return problem_config


if __name__=="__main__":
    #Initializing the graph args
    graph_args={}
    graph_args["graph_type"]=["ER","SF"]
    graph_args["num_nodes"]=[4,16]
    graph_args["node_card"]=[2,8]
    graph_args["num_edges_dist"]=["uniform"] #dist to sample edge from
    graph_args["num_edge_sample"]=[2] #number of random edge per config
    graph_args["scale_alpha"]=[2,8]

    #Initializing the sparsity args
    interv_args={}
    interv_args["sparsity"]=[4,16]
    interv_args["num_node_T"]=[10,float("inf")]
    interv_args["pi_dist_type"]=["uniform","inverse"]
    interv_args["pi_alpha_scale"]=[2,8]

    #Initializing the sample distribution
    mixture_args={}
    mixture_args["mixture_sample_size"]=[1000,10000,100000,float("inf")]
    mixture_args["pi_threshold_scale"]=[0.25,0.5,1] #to be multipled by (1/|S|)
    mixture_args["split_threshold"]=[-1e-10]

    #Evaluation args
    eval_args={}
    eval_args["matching_weight"]=[1.0/3.0]

    #Now we are ready to start our experiments
    experiment_id="exp4"
    pathlib.Path(experiment_id).mkdir(parents=True,exist_ok=True)
    shantilal = GeneralMixtureJobber(graph_args,interv_args,mixture_args,eval_args)
    shantilal.run_job_parallely(experiment_id)
