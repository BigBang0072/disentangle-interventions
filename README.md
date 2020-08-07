# disentangle-mixture-of-interventions
This repository contains the code corresponding to the work done at the research internship at Adobe.


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
