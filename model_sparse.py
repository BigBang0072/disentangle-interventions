import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb
tf.random.set_seed(211)
np.random.seed(211)
from pprint import pprint

from data_handle import BnNetwork
from latent_spaces import LatentSpace2 as LatentSpace

class Encoder(keras.layers.Layer):
    '''
    Create a stem-representation for the prediction of all the intervention
    using all the slots
    '''
    def __init__(self,dense_config,sparsity_factor,
                coef_config,sp_dense_config,sp_dense_config_base,
                temp_config,global_step,smry_writer,
                sample_strategy,cutoff_config,**kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.global_step=global_step
        self.smry_writer=smry_writer

        #First of all initializing all the layer to create stem-representation
        self.dense_layers=[]
        for units,actvn in dense_config:
            self.dense_layers.append(
                                keras.layers.Dense(units,activation=actvn))

        #Now we will initialize out Latent Output layer
        self.latent_space=LatentSpace(sparsity_factor=sparsity_factor,
                                coef_config=coef_config,
                                sp_dense_config=sp_dense_config,
                                sp_dense_config_base=sp_dense_config_base,
                                temp_config=temp_config,
                                global_step=global_step,
                                smry_writer=smry_writer,
                                sample_strategy=sample_strategy,
                                cutoff_config=cutoff_config)

    def call(self,inputs):
        '''
        Passing the input through encoder.
        '''
        #Passing though the initial dense layer for shared stem
        X=inputs
        for layer in self.dense_layers:
            X=layer(X)
        shared_stem=X

        #Now we are ready for the predictions of internvetions
        interv_locs,interv_locs_prob=self.latent_space(shared_stem)

        return interv_locs,interv_locs_prob

class Decoder(keras.layers.Layer):
    '''
    Here we will decode and calculate the sample log-probability for the,
    final objective to train on.
    '''

    def __init__(self,oracle,global_step,smry_writer,**kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.oracle=oracle
        self.global_step=global_step
        self.smry_writer=smry_writer

    def call(self,interv_locs,interv_locs_prob,inputs):
        '''
        Take the given intervention location and the mixture probbaility
        and conpute the sample log-prob.
        '''
        #First converting interv_loc to number
        def convert_np2int(interv_locs):
            new_interv_locs=[]
            for node_ids,cat_ids in interv_locs:
                #Converting the numpy integer to regular ones
                new_node_ids=[int(node_id) for node_id in node_ids]
                new_cat_ids=[int(cat_id) for cat_id in cat_ids]

                #Adding them to the locs list
                new_interv_locs.append(
                            (tuple(new_node_ids),tuple(new_cat_ids)))
            return new_interv_locs

        interv_locs=convert_np2int(interv_locs)
        pprint(interv_locs)
        pprint(interv_locs_prob)

        #Getting the probability of samples for each location
        sample_locs_prob=self.oracle.get_sample_probability(interv_locs,
                                                            inputs)
        #Now we will calculate the likliehood of whole sample
        sample_logprob=self._calculate_sample_likliehood(interv_locs_prob,
                                                        sample_locs_prob)

        return sample_logprob


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

class AutoEncoder(keras.Model):
    '''
    We will combine both the Encoder and Decoder in one for this front,
    interface of our whole model.
    '''
    def __init__(self,dense_config,sparsity_factor,
                coef_config,sp_dense_config,sp_dense_config_base,
                temp_config,smry_writer,
                sample_strategy,cutoff_config,
                oracle,**kwargs):
        super(AutoEncoder,self).__init__(**kwargs)
        self.smry_writer=smry_writer
        #Now we will also maintain a global step for our decay
        self.global_step=tf.Variable(0.0,trainable=False,
                                    dtype="float32",name="gstep")

        #Now we will create our encoder object
        self.encoder=Encoder(dense_config=dense_config,
                            sparsity_factor=sparsity_factor,
                            coef_config=coef_config,
                            sp_dense_config=sp_dense_config,
                            sp_dense_config_base=sp_dense_config_base,
                            temp_config=temp_config,
                            global_step=self.global_step,
                            smry_writer=smry_writer,
                            sample_strategy=sample_strategy,
                            cutoff_config=cutoff_config)
        #Creating our decoder
        self.decoder=Decoder(oracle=oracle,
                            global_step=self.global_step,
                            smry_writer=smry_writer)

    def call(self,inputs):
        #Getting the intervention location and mixture probability
        interv_locs,interv_locs_prob=self.encoder(inputs)

        #Now getting the likliehood of the sample
        samples_logprob=self.decoder(interv_locs,interv_locs_prob,inputs)

        #Incrementing the global step
        self.global_step.assign_add(1.0)

        return samples_logprob
