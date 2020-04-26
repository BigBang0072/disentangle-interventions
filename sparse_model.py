import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb
tf.random.set_seed(211)
np.random.seed(211)

from data_handle import BnNetwork

class LatentSlot(keras.layers.Layer):
    '''
    This class will design the representation and function for a particular,
    slot of the latent space. So, ultimately our latent space will contain
    S-of these slots.
    '''
    final_input=None        #This will be after additional dense is applied
    self.temperature=None   #Holding the current temperature
    self.tau=None           #The current cutoff value

    def __init__(self,coef_config,sp_dense_config,
                temp_config,global_step,smry_writer,
                sample_strategy,cutoff_config,**kwargs):
        '''
        coef_config     : same representation we used in previous Encoder
        sp_dense_config : config for further specialization of the shared inp
        '''
        super(LatentSlot,self).__init__(**kwargs)
        #Initializing our variabels
        self.coef_config=coef_config
        self.global_step=global_step
        self.smry_writer=smry_writer
        self.sample_strategy=sample_strategy

        #Initializing our dense layers if required
        self.sp_dense_layers=[]
        for units,actvn in sp_dense_config:
            self.sp_dense_layers.append(
                        keras.layers.Dense(units,activation=actvn))

        #Now we will initialize the node decision layer
        self.output_config=coef_config[:-1] #Will have no-intervention seperate
        self.node_decision=keras.layers.Dense(len(self.output_config),
                                                activation=None)
        #Now we will have probability distribution on separete node's category
        self.cout_layers=[]
        for cunits in self.output_config:
            self.cout_layers.append(keras.layers.Dense(cunits,activation=None))

        #Initializing the temperature variable
        (self.soften,self.init_temp,
            self.temp_decay_rate,self.temp_decay_step)=temp_config
        if self.soften==True:
            assert 0.0<=self.temp_decay_rate<=1,"Decay rate in wrong range!!"

        #Initializing the cutoff variable
        self.tau_max,self.scale_factor=cutoff_config
        assert 0<self.tau_max<1,"Give correct max-cutoff value"

    def call(self,hidden_inputs):
        '''
        Now, we will take the hidden shared representaion of our encoder,
        and:
            1. generate specialized representaion (maybe break symmetry)
            2. then generate the node decision and category dist
            3. finally we will sample the nodes and give out the single
                scalar and corresponding to node's pi estimate and
                corresponding config of intervention.
        '''
        #First of let us specialize the shared representation
        H=hidden_inputs
        for sp_layer in self.sp_dense_layers:
            H=sp_layer(H)



        #Now we will generate the node decision vector
        node_decision=self.node_decision(H)
        #Passing through sigmoid to let them separately generate decision
        node_decision=tf.math.sigmoid(node_decision)
        #We have to average over examples to get expected decision
        node_decision=tf.reduce_mean(node_decision,axis=0)



        #Calculating the temperature at this point of time
        self.temperature=self.init_temp*tf.math.pow(
                                self.temp_decay_rate,
                                self.global_step/self.temp_decay_step)
        #Clipping the value to not go below 1
        self.temperature=tf.clip_by_value(self.temperature,
                                        clip_value_min=1.0,
                                        clip_value_max=self.init_temp)
        #Adding summary for temperature
        with self.smry_writer.as_default():
            tf.summary.scalar("temperature",self.temperature,
                                step=int(self.global_step.value()))



        #Now we will generate the distribution over the category of each node
        all_cat_prob=[]
        all_cat_idx=[]
        for nidx,cout_layer,num_cat in enumerate(
                                zip(self.cout_layers,self.output_config)):
            #Getting a valid distribution among the nodes
            cout=cout_layer(H)
            #Now we will either soften if we want to or leave it as it is
            if self.soften==True:
                cout=cout/self.temperature

            cout=tf.nn.softmax(cout)
            #Getting the average distribution over the categories
            cout=tf.reduce_mean(cout,axis=0)

            #Now we will sample the node, either directly or by perturbing
            cat_prob,cat_idx=self._sample_categories(cout,num_cat)
            all_cat_prob.append(cat_prob)
            all_cat_idx.append((nidx,cat_idx))
        #Now we have intervened category for each node and their prob,
        #We will now get the score for this particulat intervention
        all_cat_prob=tf.concat(all_cat_prob,axis=0)#possible we have to stack


        #Now we have to get the node score for the node and cat pair
        #Prepare the schedule for the cutoff constant
        self.tau=self.tau_max*(1-tf.exp(-1*self.global_step/self.scale_factor))
        #Adding summary for cutoff value
        with self.smry_writer.as_default():
            tf.summary.scalar("cutoff_value",self.tau,
                                step=int(self.global_step.value()))

        #Lets get the bollean array of values above threshold
        above_tau=tf.math.greater(node_decision,self.tau)
        #Getting the score of this slot now
        #BEWARE: These number cold be small therefore could lead to van grad
        interv_score=(node_decision*tf.cast(above_tau,tf.float32)
                                        *all_cat_prob)
        #Adding up the score (alternatively we could multiply: tweak)
        interv_score=tf.reduce_sum(interv_score)



        #Now selecting the node-cat selected for this interventions
        interv_loc=self._get_intervention_locations(above_tau,all_cat_idx)

        return interv_loc,interv_score

    def _sample_categories(self,cout,num_cat):
        '''
        This function will sample a category of a node which is going to take
        part in the intervention.
        '''
        assert num_cat>1,"The no-intervention node is here by mistake"

        if self.sample_strategy=="max" or self.temperature.numpy()==1.0:
            cat_prob,cat_idx=tf.math.top_k(cout,k=1)
        elif self.sample_strategy=="gumbel":
            #Generating the samples from gumbel distribution
            gumbel_samples=np.random.gumbel(0,1,size=num_cat)
            #Perturbing the log-probability of samples
            perturb_logprob=tf.math.log(cout)+gumbel_samples

            #Now we will select the mx prob one
            _,cat_idx=tf.math.top_k(perturb_logprob,k=1)
            cat_prob=tf.gather(cout,cat_idx,axis=0,batch_dims=0)
        else:
            raise NotImplementedError

        return cat_prob,cat_idx

    def _get_intervention_locations(self,above_tau,all_cat_idx):
        '''
        This will take the category selected for each node, along with the
        decision that node is allowed to construct us the intervention,
        configuration combining them
        '''
        interv_nodes=[]
        interv_cats=[]
        above_tau_np=above_tau.numpy()
        for tidx in range(above_tau_np.shape[0]):
            if(above_tau_np[tidx]==True):
                node,cat=all_cat_idx[tidx]
                interv_nodes.append(node)
                interv_cats.append(cat.numpy())
        interv_loc=(tuple(interv_nodes),tuple(interv_cats))

        return interv_loc

class LatentSpace(keras.layers.Layer):
    '''
    This class will manage all the individual slots and give finally give
    us the intervention locations and their corresponding contribution,
    which will be then fed to decoder for calculating the logprobability.
    '''
    def __init__(self,sparsity_factor,
                coef_config,sp_dense_config,sp_dense_config_base,
                temp_config,global_step,smry_writer,
                sample_strategy,cutoff_config,**kwagrgs):
        super(LatentSpace,self).__init__(**kwargs)
        self.sparsity_factor=sparsity_factor
        self.base_num=len(coef_config)-1

        #First of all we will initialize all the latent slots we need
        self.latent_slots=[]
        for sidx in range(sparsity_factor):
            latent_slot=LatentSlot(coef_config=coef_config,
                                    sp_dense_config=sp_dense_config,
                                    temp_config=temp_config,
                                    global_step=global_step,
                                    smry_writer=smry_writer,
                                    sample_strategy=sample_strategy,
                                    cutoff_config=cutoff_config)
            self.latent_slots.append(latent_slot)

        #Now we have to desin one special slot for the no-intervention
        self.sp_dense_layers=[]
        for units,actvn in sp_dense_config_base:
                layer=keras.layers.Dense(units,activation=actvn)
                self.sp_dense_layers.append(layer)
        #Adding the last layer to give us one single score
        self.sp_dense_layers.append(keras.layers.Dense(units=1,
                                                    activation=None))

    def call(self,shared_input):
        '''
        Here we will generate all the prediction of possible intervention,
        along with the unnormalized score for that particular configuration.
        '''
        #First of all lets collect all the prediction from the other slots
        interv_locs=[]
        interv_locs_prob=[]
        for sidx in range(self.sparsity_factor):
            interv_loc,interv_score=self.latent_slots[sidx](shared_input)
            interv_locs.append(interv_loc)
            interv_locs_prob.append(interv_score)

        #Now we will generate score for the base-distirbution configuration
        H=shared_input
        for sp_layer in self.sp_dense_layers:
            H=sp_layer(H)
        #Now we will not pass it through sigmoid since we are normalizin ltr
        base_score=H

        #Adding this acore and loc to our list
        interv_locs.append(((self.base_num,),(0,)))
        interv_locs_prob.append(base_score)

        #Now we will normalize the interv_loc_prob
        interv_locs_prob=tf.concat(interv_locs_prob,axis=0)
        interv_locs_prob=tf.softmax(interv_locs_prob,axis=0)

        return interv_locs,interv_locs_prob

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
        latent_space=LatentSpace(sparsity_factor=sparsity_factor,
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
        X=input
        for layer in self.dense_layers:
            X=layer(X)
        shared_stem=X

        #Now we are ready for the predictions of internvetions
        interv_locs,interv_locs_prob=latent_space(shared_stem)

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
                            global_step=global_step,
                            smry_writer=smry_writer,
                            sample_strategy=sample_strategy,
                            cutoff_config=cutoff_config)
        #Creating our decoder
        self.decoder=Decoder(oracle=oracle,
                            global_step=global_step,
                            smry_writer=smry_writer)

    def call(self,inputs):
        #Getting the intervention location and mixture probability
        interv_locs,interv_locs_prob=self.encoder(inputs)

        #Now getting the likliehood of the sample
        samples_logprob=self.decoder(interv_locs,interv_locs_prob,inputs)

        #Incrementing the global step
        self.global_step.assign_add(1.0)

        return samples_logprob
