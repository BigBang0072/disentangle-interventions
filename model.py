import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(211)
import pdb
import numpy as np
np.random.seed(211)
from pprint import pprint

from data_handle import BnNetwork

class Encoder(keras.layers.Layer):
    '''
    This class will encode the given sample to the mixing coefficient space.
    '''
    def __init__(self,dense_config,coef_config,
                temp_config,global_step,smry_writer,**kwargs):
        '''
        dense_config    : [[num_unit1,activation1],[num_unit2,activation2]...]
        coef_config     : the cardinality of each of the coefficient (pi^{..})
                            Ensure that one pi is for init-dist with 1 card.
        '''
        super(Encoder,self).__init__(**kwargs)
        self.smry_writer=smry_writer
        self.global_step=global_step
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

        #Initializing the temperature variable
        (self.soften,self.init_temp,
            self.temp_decay_rate,self.temp_decay_step)=temp_config
        if self.soften==True:
            assert 0.0<=self.temp_decay_rate<=1,"Decay rate in wrong range!!"

    def call(self,inputs,global_step):
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
            coef_actv=coef_layer(X)
            #Using temperature to smoothen out the probabilities
            if self.soften==True:
                #Now we have to decay this as the training goes on
                self.temperature=self.init_temp*tf.math.pow(
                                            self.temp_decay_rate,
                                            global_step/self.temp_decay_step)
                #Clipping the value to not go below 1
                self.temperature=tf.clip_by_value(self.temperature,
                                                clip_value_min=1.0,
                                                clip_value_max=self.init_temp)
                print("TEMPERATURE:",self.temperature)
                #Adding summary for temperature
                with self.smry_writer.as_default():
                    tf.summary.scalar("temperature",self.temperature,
                                        step=int(self.global_step.value()))

                #Applying the softening
                coef_actv=coef_actv/self.temperature

            #Now we will expoentitate to convert to probability
            coef_actv=tf.math.exp(coef_actv)#BEWARE: inf,use subtrac trick
            coef_activations.append(coef_actv)

            #Adding up the contribution to normalize later
            normalization_const+=tf.reduce_sum(coef_actv,axis=1,keepdims=True)
        #Now we will normalize the output taking all the nodes into account
        coef_output=[coef_actv/normalization_const
                                        for coef_actv in coef_activations]
        #Now we have to get the average prediction of all the sample about mixt
        coef_output_avg=[tf.reduce_mean(coef_prob,axis=0)
                                        for coef_prob in coef_output]

        #First of all we have to stack all the output into one single tensor
        concat_output=tf.concat(coef_output_avg,axis=0)

        return concat_output

class Decoder(keras.layers.Layer):
    '''
    Take the output of the encoder in the coefficient space and then will
    apply the deterministic operation to get the mixture distribution.

    Here we will have control over sparsity, choosing the top probable
    interventions in mixture and then calculating likliehood.
    '''
    lookup_i2l=None         #Lookup table to convert flat index to interv-loc
    lookup_l2i=None         #Convert loc to flat index

    def __init__(self,sparsity_factor,coef_config,oracle,do_config,
                sample_strategy,global_step,smry_writer,**kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.global_step=global_step
        self.smry_writer=smry_writer
        #Initilaizing the oracle which is handling all PGM based work
        self.oracle=oracle
        self.do_config=do_config
        #Parameters to choose the top-k interventions out of all
        self.sparsity_factor=sparsity_factor
        #Variable holding the number of category in each node (for unconcat)
        self.coef_config=coef_config
        #TODO: to be changed later when we are doing multiple interventions
        assert coef_config[-1]==1,"Last pi should be for init-distirbution"
        #Initializing the sampling strategy
        self.sample_strategy=sample_strategy
        assert sample_strategy in ["top-k","gumbel"],"Wrong sample strategy"

    def call(self,inputs,concat_output):
        '''
        This function will select the top-|sparsity| number of node which says
        they are most probable for intervention to happen. Then use them to
        compute the likliehood.
        '''
        if self.sample_strategy=="top-k":
            #Now we will retreive the tok-k position of interventions
            interv_loc_prob,indices=tf.math.top_k(concat_output,
                                                k=self.sparsity_factor)
        elif self.sample_strategy=="gumbel":
            #Here we will sample the indices instead of top-k directly
            total_dims=sum(self.coef_config)
            gumbel_samples=np.random.gumbel(0,1,size=total_dims)
            #Now we could add this pertubation to the logprob
            perturb_logprob=tf.math.log(concat_output)+gumbel_samples

            #Now we will have to select the top-k
            _,indices=tf.math.top_k(perturb_logprob,k=self.sparsity_factor)
            #Also we we need the log prob of those indices
            interv_loc_prob=tf.gather(concat_output,indices,
                                        axis=0,batch_dims=0)
        else:
            raise NotImplementedError

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
        #Calculating the mse of the actual intervened pi and predicted
        doMAE=self._calculate_doMSE(concat_output,self.do_config)
        self.add_metric(doMAE,name="doMAE",aggregation="mean")

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
        if self.lookup_i2l==None:
            #Creating a lookup table for locations
            lookup_i2l={}
            lookup_l2i={}
            counter=0
            for nidx in range(len(self.coef_config)):
                for cidx in range(self.coef_config[nidx]):
                    lookup_i2l[counter]=((nidx,),(cidx,))
                    lookup_l2i[((nidx,),(cidx,))]=counter
                    counter+=1
            #Now we will hash this for later use
            self.lookup_i2l=lookup_i2l
            self.lookup_l2i=lookup_l2i
        #Now we are ready to convert 1-D indices to intervention location
        interv_locs=[self.lookup_i2l[idx] for idx in indices]
        return interv_locs

    def _calculate_sample_likliehood(self,interv_loc_prob,sample_loc_prob,
                                    tolerance=1e-10):
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
        sample_logprob=tf.math.log(sample_prob+tolerance)
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

        #Adding the summary of recall
        with self.smry_writer.as_default():
            tf.summary.scalar("doRecall",recall,
                                step=int(self.global_step.value()))

        return recall

    def _calculate_doMSE(self,concat_output,do_config):
        '''
        To calculate the mse of actual itnervention coefficient and predicted
        coefficient.
        '''
        #Getting the actual and predicted pis
        actual_pis=[]
        pred_pis=[]
        adiff_pis=[]
        for nodes,cats,pi in do_config:
            actual_pis.append(pi)
            #Now getting the predicted pi
            pred_idx=self.lookup_l2i[(tuple(nodes),tuple(cats))]
            pi_hat=concat_output[pred_idx]
            pred_pis.append(pi_hat)

            #Now we will individually add absolute error for each component
            with self.smry_writer.as_default():
                name=str((nodes,cats,pi))
                adiff=abs(pi-pi_hat)
                adiff_pis.append(adiff)
                tf.summary.scalar(name,adiff,
                                    step=int(self.global_step.value()))

        #Also we will add the mean absolute error for all prediction
        mean_adiff=np.mean(adiff_pis)
        with self.smry_writer.as_default():
            tf.summary.scalar("mae",mean_adiff,
                                    step=int(self.global_step.value()))

        return mean_adiff

class AutoEncoder(keras.Model):
    '''
    This class will merge both the Encoder and Decoder inside it and calculate
    overall loss.
    '''
    def __init__(self,dense_config,coef_config,sparsity_factor,
                    oracle,do_config,temp_config,sample_strategy,
                    smry_writer,**kwargs):
        super(AutoEncoder,self).__init__(**kwargs)
        self.smry_writer=smry_writer
        #Now we will also maintain a global step for our decay
        self.global_step=tf.Variable(0.0,trainable=False,
                                    dtype="float32",name="gstep")

        #Now we will initialize our Encoder and Decoder
        self.encoder=Encoder(dense_config,coef_config,
                            temp_config,self.global_step,smry_writer)
        self.decoder=Decoder(sparsity_factor,coef_config,oracle,
                            do_config,sample_strategy,
                            self.global_step,smry_writer)

    def call(self,inputs):
        #First of all calling the encoder to map us to latent space
        concat_output=self.encoder(inputs,self.global_step)
        #Now we will get the likliehood of the samples (kept tack internally)
        samples_logprob=self.decoder(inputs,concat_output)
        #We have to update the global step after each update
        self.global_step.assign_add(1.0)

        return samples_logprob
