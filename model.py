import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(211)

def Encoder(keras.layers.Layer):
    '''
    This class will encode the given sample to the mixing coefficient space.
    '''
    def __init__(self,dense_config,coef_config):
        '''
        dense_config    : [[num_unit1,activation1],[num_unit2,activation2]...]
        coef_config     : the cardinality of each of the coefficient (pi^{..})
                            Ensure that one pi is for init-dist with 1 card.
        '''
        super(Encoder,self).__init__()
        #Now we will initialize the layers to be used in encoder
        dense_layers=[]
        for lid,config in enumerate(dense_config):
            units,activ=config
            dense_layers.append(keras.layers.Dense(units,activation=activ))
        self.dense_layers=dense_layers

        #Now we will initialize the final layer layers # TODO:
        assert coef_config[-1]==1,"Last config should be for base-dist"
        coef_layers=[]
        for cid,units in enumerate(coef_config):
            coef_layers.append(keras.layers.Dense(units,activation=None))
        self.coef_layers=coef_layers

    def call(self,input):
        '''
        We could encode out input categories both as one-hot OR
        continuous vaued number giving only one dimension per node.

        input   : tensor of shape [batch,#nodes*?]
        '''
        #First of all we apply the encoder layers
        X=input
        for layer in self.dense_layers:
            X=layer(X)

        #Now we will have to generate the distribution in latent variable
        coef_activations=[]
        normalization_const=0.0
        for coef_layer in self.coef_layers:
            #Applying the layer and exp to get unnormalized probability
            coef_actv=tf.math.exp(coef_layer(X))
            coef_activations.append(coef_actv)

            #Adding up the contribution to normalize later
            normalization_const+=tf.reduce_sum(coef_actv,axis=1,keep_dims=True)
        #Now we will normalize the output taking all the nodes into account
        coef_output=[coef_actv/normalization_const
                                        for coef_actv in coef_activations]
        #Now we have to get the average prediction of all the sample about mixt
        coef_output_avg=[tf.reduce_mean(coef_prob,axis=0)
                                        for coef_prob in coef_output]

        return coef_output_avg

def Decoder(keras.layers.Layer):
    '''
    Take the output of the encoder in the coefficient space and then will
    apply the deterministic operation to get the mixture distribution.

    Here we will have control over sparsity, choosing the top probable
    interventions in mixture and then calculating likliehood.
    '''
    def __init(self,sparsity_factor,coef_config,oracle):
        super(Decoder,self).__init__()
        #Initilaizing the oracle which is handling all PGM based work
        self.oracle=oracle
        #Parameters to choose the top-k interventions out of all
        self.sparsity_factor=sparsity_factor
        #Variable holding the number of category in each node (for unconcat)
        self.coef_config=coef_config
        #TODO: to be changed later when we are doing multiple interventions
        assert coef_config[-1]==1,"Last pi should be for init-distirbution"

    def call(self,coef_output_avg,input_samples):
        '''
        This function will select the top-|sparsity| number of node which says
        they are most probable for intervention to happen. Then use them to
        compute the likliehood.
        '''
        #First of all we have to stack all the output into one single tensor
        concat_output=tf.concat(coef_output_avg,axis=0)
        #Now we will retreive the tok-k position of interventions
        value,indices=tf.math.top_k(concat_output,k=self.sparsity_factor)

        #Now once we have indices we reconvert them to locations of interv
        interv_locs=self._get_intervention_locations(indices)

        #Now we are ready to calculate the likliehood of different samples
        #TODO: Assuming that we are generating actual probability here
        sample_loc_prob=oracle.get_sample_probability(interv_loc,
                                                    input_samples)

        #This is temporary #TODO
        interv_loc_prob=value
        return interv_loc_prob,sample_loc_prob

    def _get_intervention_locations(self,indices):
        '''
        Given a list of indices of 1-D tensor get the location of intervention
        node and categories. This is only valid for
        '''
        #Converting the indices to location
        indices=indices.numpy()
        #Creating a lookup table for locations
        lookup_table=[]
        for nidx in range(len(coef_config)):
            for cidx in range(coef_config[nidx]):
                lookup_table.append(([nidx],[cidx]))
        #Now we are ready to convert 1-D indices to intervention location
        interv_locs=[lookup_table[idx] for idx in indices]
        return interv_locs
