import numpy as np

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

        2. Generate Random Mixture of Intervention
            2.1 Sparsity    [See how sample size varies this]
            2.1 Sample Size [actual metric of variation]

    Evaluation Criteria:
        1.

    '''
