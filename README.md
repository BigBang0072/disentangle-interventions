# disentangle-mixture-of-interventions
This repository contains the code corresponding to the work done at the research internship at Adobe.


### Running Experiments for General-Mixture-Of-Interventions
1. First install all the dependencies using ```` pip install -r requirements.txt ````
2. Now to start the experiments and visualizing results:
	1. ```` python jobber.py ```` : For starting the experiment jobs. Please change the **experiment_id** in the main folder to avoid overwriting the result of old experiment. This will run **(num_cpu/2)** parallel processes.
  		* The ````__main__ ```` function contains all the experiment configuration. 
		* Please change following fields to add more experiments:
			* **graph_type** : ["ER", " SF"] supported (Erdos-Reneyi and Scale-Free)
			* **num_nodes** : Represents number of node in the graph.
			* **node_card** : Cardinality of each node
			* **num_edge_dist** : Only sampling from ["uniform"] distribution over [1,nC2] supported.
			* **num_edge_sample** : How many different edge number per configuration to be sampled. 
			* **sparsity** : Number of intervention performed on system.
			* **num_node_T** : Temperature for selection number of node present in each target.
			* **pi_dist_type** : ["uniform","inverse"] supported for sampleing the mixing coefficient.
			* **pi_scale_alpha** : alpha for drichlet distribution, (1,inf). Greater scale means sampling from center of simplex.
			* **mixture_sample_size** : The number of samples taken from mixture distribution.
			* **pi_threshold_scale** : scale * (1/|S|) for making a mixing coefficient zero if they go below this limit
			* **split_threshold** : No need to change this. This is for handling numerical error when splitting coefficient.
			* **matching_weight** : weight x JS_node + (1-weight) x JS_overall for calcualting Jaccard similarity.
			* **experiment_id** : Unique name for each experiments. The results jsons are stored in the folder with this name.
	2. ```` python plotter.py ```` : For visualizing the results for the experiments. Can be started even when the experiments are going on.
		* **experiment_id** : The name of the experiments to view the result.
  

### File Organization
+ **Data Handling:**
    - ````data_handle.py```` : Reading bif file, creating the Bayesian Network, creating internvetion graph, sampling from distirbution.
+ **VAE Implementation:**
    - **Dense Latent Space:**
        * ```` model.py ```` : Encoder and Decoder model for first order internvetions.
        * ```` run.py ```` : Running the VAE for first order intervetions.
    - **Sparse Latent Space:**
        * ````latent_spaces.py```` : Different Latent Space designs for sparse general order interventions.
        * ````model_sparse.py```` : Encoder and Decoder for genral order internvetions.
        * ````sparse_run.py```` : Running the VAE for sparse general order internvetions.
+ **Non Overlapping Internvetion:**
    - ````non_overlap_intv_solver.py```` : Implementation of the non-overlapping mixture of internvetions identifiability algorithm.
+ **Demo:**
    - ````app.py```` : Demo-1 shows the accuracy and similarity plot.
    - ````app2.py```` : Demo-2 used during the presentation for showing the usecase.
    - ````app_utils.py```` : Utilities used the application files.

### Running the Code
1. First install all the dependencies using ````pip install -r requirements.txt````
2. Running the Existing functionalities
	1. ````python run.py```` : For running VAE implementation of first-order-intervention. Please look at the main function for all the hyperparameter configuration.
	2. ````python sparse_run.py -options```` : For running VAE implementation of first-order-intervention. See the available options in the ````sparse_run.py```` file. Please look at the main function for all the hyperparameter configuration.
	3. ````python non_overlap_intv_solver.py```` : For running the identifiability algorithm. Please look at the main function for all the hyperparameter configuration.
	4. ````python app2.py```` : To run the app made to showing the use case.
