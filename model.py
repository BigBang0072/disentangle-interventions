import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(211)
import pdb
from pprint import pprint

from data_handle import BnNetwork

class Encoder(keras.layers.Layer):
    '''
    This class will encode the given sample to the mixing coefficient space.
    '''
    def __init__(self,dense_config,coef_config,**kwargs):
        '''
        dense_config    : [[num_unit1,activation1],[num_unit2,activation2]...]
        coef_config     : the cardinality of each of the coefficient (pi^{..})
                            Ensure that one pi is for init-dist with 1 card.
        '''
        super(Encoder,self).__init__(**kwargs)
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

    def call(self,inputs):
        '''
        We could encode out input categories both as one-hot OR
        continuous vaued number giving only one dimension per node.

        input   : tensor of shape [batch,#nodes*?]
        '''
        #First of all we apply the encoder layers
        X=inputs
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
            normalization_const+=tf.reduce_sum(coef_actv,axis=1,keepdims=True)
        #Now we will normalize the output taking all the nodes into account
        coef_output=[coef_actv/normalization_const
                                        for coef_actv in coef_activations]
        #Now we have to get the average prediction of all the sample about mixt
        coef_output_avg=[tf.reduce_mean(coef_prob,axis=0)
                                        for coef_prob in coef_output]

        return coef_output_avg

class Decoder(keras.layers.Layer):
    '''
    Take the output of the encoder in the coefficient space and then will
    apply the deterministic operation to get the mixture distribution.

    Here we will have control over sparsity, choosing the top probable
    interventions in mixture and then calculating likliehood.
    '''
    def __init__(self,sparsity_factor,coef_config,oracle,do_config,**kwargs):
        super(Decoder,self).__init__(**kwargs)
        #Initilaizing the oracle which is handling all PGM based work
        self.oracle=oracle
        self.do_config=do_config
        #Parameters to choose the top-k interventions out of all
        self.sparsity_factor=sparsity_factor
        #Variable holding the number of category in each node (for unconcat)
        self.coef_config=coef_config
        #TODO: to be changed later when we are doing multiple interventions
        assert coef_config[-1]==1,"Last pi should be for init-distirbution"

    def call(self,inputs,coef_output_avg):
        '''
        This function will select the top-|sparsity| number of node which says
        they are most probable for intervention to happen. Then use them to
        compute the likliehood.
        '''
        #First of all we have to stack all the output into one single tensor
        concat_output=tf.concat(coef_output_avg,axis=0)
        #Now we will retreive the tok-k position of interventions
        interv_loc_prob,indices=tf.math.top_k(concat_output,
                                                k=self.sparsity_factor)

        #Now once we have indices we reconvert them to locations of interv
        interv_locs=self._get_intervention_locations(indices)

        #Now we are ready to calculate the likliehood of different samples
        sample_loc_prob=self.oracle.get_sample_probability(interv_locs,
                                                    inputs)

        #This is temporary #TODO
        #TODO: Assuming that we are generating actual probability here
        interv_loc_prob=interv_loc_prob

        #Now getting the log probability of seeing the samples
        samples_logprob=self._calculate_sample_likliehood(interv_loc_prob,
                                                        sample_loc_prob)
        #Calculating the metrics to track progress
        doRecall=self._calculate_doRecall(interv_locs,self.do_config)
        self.add_metric(doRecall,name="doRecall",aggregation="mean")

        print("inerv_locs:",interv_locs)
        print("interv_loc_prob:",interv_loc_prob)
        print("do_config:",self.do_config)

        return samples_logprob

    def _get_intervention_locations(self,indices):
        '''
        Given a list of indices of 1-D tensor get the location of intervention
        node and categories. This is only valid for first order interventions
        '''
        #Converting the indices to location
        indices=indices.numpy()
        #Creating a lookup table for locations
        lookup_table=[]
        for nidx in range(len(self.coef_config)):
            for cidx in range(self.coef_config[nidx]):
                lookup_table.append(([nidx],[cidx]))
        #Now we are ready to convert 1-D indices to intervention location
        interv_locs=[lookup_table[idx] for idx in indices]
        return interv_locs

    def _calculate_sample_likliehood(self,interv_loc_prob,sample_loc_prob):
        '''
        Given the probability of the interventions and the probability of
        samples in those interventions, this function will calulcate the
        overall log-liklihood of the samples.

        Assumption:
        The interv_loc_distiribution is a valid distribution, i.e sum of
        probability of all the intervnetion done is equal to 1
        '''
        #Getting the overall sample probability of mixture
        sample_prob=tf.reduce_sum(interv_loc_prob*sample_loc_prob,axis=1)

        #Now we will calculate the overall log likliehood
        sample_logprob=tf.math.log(sample_prob)
        all_sample_logprob=tf.reduce_mean(sample_logprob)

        #Adding this to the layers losses
        return all_sample_logprob

    def _calculate_doRecall(self,interv_locs,do_config):
        '''
        This function will calculate the recall of the intervention done
        actually and then present in the top-Sparsity candidates.
        '''
        total_count=len(do_config)*1.0
        presence_count=0.0
        for do in do_config:
            nodes,cats,_=do
            if (nodes,cats) in interv_locs:
                presence_count+=1
        #Now we will also have to track the no intervention case
        _,_,pis=zip(*do_config)
        phi=1-sum(pis)
        no_interv_idx=len(self.coef_config)-1
        if phi>0:
            total_count+=1
            if ([no_interv_idx],[0]) in interv_locs:
                presence_count+=1
        #Now we are ready to calculate the recall
        recall=presence_count/total_count
        # pdb.set_trace()
        return recall

class AutoEncoder(keras.Model):
    '''
    This class will merge both the Encoder and Decoder inside it and calculate
    overall loss.
    '''
    def __init__(self,dense_config,coef_config,sparsity_factor,
                    oracle,do_config,**kwargs):
        super(AutoEncoder,self).__init__(**kwargs)
        #Now we will initialize our Encoder and Decoder
        self.encoder=Encoder(dense_config,coef_config)
        self.decoder=Decoder(sparsity_factor,coef_config,oracle,do_config)

    def call(self,inputs):
        #First of all calling the encoder to map us to latent space
        coef_output_avg=self.encoder(inputs)
        #Now we will get the likliehood of the samples (kept tack internally)
        samples_logprob=self.decoder(inputs,coef_output_avg)

        return samples_logprob
