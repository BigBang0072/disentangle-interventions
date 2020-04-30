import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.random.set_seed(211)
np.random.seed(211)
from pprint import pprint
import pdb


#############################################################################
################################# SLOT-1 ####################################
#############################################################################
class LatentConfig2(keras.layers.Layer):
    '''
    This class will take over all the functionality of prediction and selection
    for a particular order or intervnetion.

    Also, this will also provide the config-score, to select the order of
    intervention from.
    '''
    config_logprob=None     #[batch,1]: to hold the score for this config
    interv_score=None       #The socre for the intervention sel in this config
    interv_loc=None         #The interv loc for this config
    temperature=None        #the current temperature received from above

    lookup_i2l=None         #To map the concatenated index to node-cat
    lookup_l2i=None         #Reverse mapping for the above map

    def __init__(self,order_interv,soften,
                sample_strategy,coef_config,**kwargs):
        '''
        order_interv    : the order of intervention.
        temperature     : the value of temperature to be used to smoothen
        '''
        super(LatentConfig2,self).__init__(**kwargs)
        self.order_interv=order_interv
        # self.temperature=temperature
        self.soften=soften
        self.sample_strategy=sample_strategy

        #Now we will initialize our projection layer to categories of each node
        self.output_config=coef_config[:-1]
        self.cout_layers=[]
        for cunits in self.output_config:
            self.cout_layers.append(keras.layers.Dense(cunits,activation=None))

    def call(self,slot_hidden,temperature):
        '''
        Now we will generate the distribution over the categories, sample
        the "order-interv" number of node-cat and give it to the slot to decide
        '''
        #Keeping track of current temperature
        self.temperature=temperature

        #Applying the final projection to the coef-layer
        couts_logprob=[]
        wtn_nodes_selection=[]
        for nidx,(cout_layer,num_cat) in enumerate(
                                zip(self.cout_layers,self.output_config)):
            #Getting the distribution for this node
            cout_logprob=cout_layer(slot_hidden)
            couts_logprob.append(cout_logprob)

        #Now we will merge them all in one single vector
        couts_logprob=tf.concat(couts_logprob,axis=1)
        #Now we will create our two branch,first for selection
        couts_logprob_temp=couts_logprob
        if self.soften==True:
            couts_logprob_temp=couts_logprob/temperature
        couts_temp=tf.nn.softmax(couts_logprob_temp,axis=1)
        couts_temp=tf.reduce_mean(couts_temp,axis=0)
        print("Node_Heat:",couts_temp)

        #Now we will ready our score generation branch
        couts_logprob_mean=tf.reduce_mean(couts_logprob,axis=0)

        #Now let's sample the indices and the location
        logscores,indices=self._sample(dist_logscore=couts_logprob_mean,
                                        dist_temp=couts_temp,
                                        k=self.order_interv)


        #Now its time to convert the indices into node-cat pair
        interv_loc=self._get_intervention_locations(indices)
        #Now we will calcuate one single value score for this loc
        interv_score=tf.reduce_sum(logscores,axis=0)

        #Assigning the score and loc to class
        self.interv_loc=interv_loc
        self.interv_score=interv_score
        # pdb.set_trace()

        assert len(interv_loc[0])==self.order_interv,"Loc Order mismatch"
        return self.interv_score,self.interv_loc

    def _sample(self,dist_logscore,dist_temp,k):
        '''
        dist_logscore   : the actual log-prob non-temperaturized
        dist_temp       : the distribution with temp enabled
        k               : number of samples
        dims            : the number of dimension in the distribution
        '''
        #Getting the dimension we obtain after stacking
        dims=sum(self.output_config)

        if self.sample_strategy=="top-k" or self.temperature.numpy()==1.0:
            _,indices=tf.math.top_k(dist_temp,k=k)
        elif self.sample_strategy=="gumbel":
            #Generating the gumbel samples for perturbing
            gumbel_samples=np.random.gumbel(0,1,size=dims)
            #Perturbing the actual distribution
            perturb_logprob=tf.math.log(dist_temp)+gumbel_samples

            #Now we will select the top-k
            _,indices=tf.math.top_k(perturb_logprob,k=k)
        else:
            raise NotImplementedError

        #Now we will select the score not from this softmaxed guys
        logscores=tf.gather(dist_logscore,indices,axis=0,batch_dims=0)
        # pdb.set_trace()

        return logscores,indices

    def _get_intervention_locations(self,indices):
        '''
        Based on the nodes selected, we will select the category for that node

        Changeslog:
        1. Now we will convert the indices gotten from concat of all indexes
            to node-cat pair.
            In case to cat from same node are selected we could remove them
        '''
        #First of all creating this lookup, for concat_indices to node-cat
        if self.lookup_i2l==None:
            #Creating a lookup table for locations
            lookup_i2l={}
            lookup_l2i={}
            counter=0
            for nidx in range(len(self.output_config)):
                for cidx in range(self.output_config[nidx]):
                    lookup_i2l[counter]=(nidx,cidx)
                    lookup_l2i[(nidx,cidx)]=counter
                    counter+=1
            #Saving this hash for later runs
            self.lookup_i2l=lookup_i2l
            self.lookup_l2i=lookup_l2i

        #Now we will reverse map the index to node-cat
        interv_nodes=[]
        interv_cats=[]
        indices=indices.numpy().tolist()
        # pdb.set_trace()
        for tidx in indices:
            #Retreiving cat and node from index
            node,cat=self.lookup_i2l[tidx]

            #Now we add the nodes and category
            #if node not in interv_nodes:   #So, that we dont repreat cats
            interv_nodes.append(np.int32(node))
            interv_cats.append(np.int32(cat))

        #Now putting all of then in one loc variabel
        interv_loc=(tuple(interv_nodes),tuple(interv_cats))

        return interv_loc

class LatentSlot2(keras.layers.Layer):
    '''
    This latent slot will incorporate our new idea, of having different
    slots for different order of intervention. One conjecture we have is
    that NN. should be good in comparison as compared to prediction.
    Hence this design will let us make comparison on various order of
    intervention and corresponding node selection rather than prediction.
    '''
    temperature=None        #Holding the current temperature

    def __init__(self,sp_dense_config,coef_config,
                soften,sample_strategy,**kwargs):
        '''
        '''
        super(LatentSlot2,self).__init__(**kwargs)
        self.soften=soften
        # self.temperature=temperature
        self.sample_strategy=sample_strategy

        #First of all initialize layers for specialization of stem of encoder
        self.sp_dense_layers=[]
        for units,actvn in sp_dense_config:
            self.sp_dense_layers.append(
                        keras.layers.Dense(units,activation=actvn))

        #Now let's initialize the all the differnet config (1---N)
        self.num_configs=len(coef_config)-1
        self.latent_configs=[]
        for cidx in [1]:#range(self.num_configs):
            config=LatentConfig2(order_interv=cidx+1,
                                soften=soften,
                                sample_strategy=sample_strategy,
                                coef_config=coef_config)
            self.latent_configs.append(config)

    def call(self,encoder_hidden,temperature):
        '''
        1. Take the stem from encoder and specialize to break symmetry (possib)
        2. Then generate the distribution over different config (order of I)
        3. Select the most probable config and its intervention
        '''
        #Keeping track of current temperature
        self.temperature=temperature

        #Lets first specialize the stem from encoder
        H=encoder_hidden
        for layer in self.sp_dense_layers:
            H=layer(H)

        #Now we are ready to generate the configs and their distribution
        configs_logprob=[]
        configs_scores=[]
        configs_locs=[]
        for config in self.latent_configs:
            wtn_score,wtn_loc=config(H,temperature)

            # configs_logprob.append(logprob)
            configs_scores.append(wtn_score)
            configs_locs.append(wtn_loc)

        #All these locs from each config will be treated as on location
        interv_scores=configs_scores
        interv_locs=configs_locs

        return interv_scores,interv_locs

    def _sample_config(self,configs_prob,num_config):
        '''
        Here, we could rescale the config (order probabilities) proportional
        to the size of search space for that order.
        '''
        if self.sample_strategy=="top-k" or self.temperature.numpy()==1.0:
            prob,config_idx=tf.math.top_k(configs_prob,k=1)
        elif self.sample_strategy=="gumbel":
            #Generating the samples from gumbel distribution
            gumbel_samples=np.random.gumbel(0,1,size=num_config)
            #Perturbing the log-probability of samples
            perturb_logprob=tf.math.log(configs_prob)+gumbel_samples

            #Now we will select the mx prob one
            _,config_idx=tf.math.top_k(perturb_logprob,k=1)
            prob=tf.gather(configs_prob,config_idx,axis=0,batch_dims=0)
        else:
            raise NotImplementedError

        return tf.squeeze(prob),tf.squeeze(config_idx)

class LatentSpace2(keras.layers.Layer):
    '''
    Finally this will culminate, all the slots into one class.
    '''
    def __init__(self,sparsity_factor,
                coef_config,sp_dense_config,sp_dense_config_base,
                temp_config,global_step,smry_writer,
                sample_strategy,cutoff_config,**kwargs):
        '''
        '''
        super(LatentSpace2,self).__init__(**kwargs)
        self.sparsity_factor=sparsity_factor
        self.base_num=len(coef_config)-1
        self.smry_writer=smry_writer
        self.global_step=global_step
        #self.cutoff_config=cutoff_config   #Kept as dummy

        #Initializing the temperature variable
        (self.soften,self.init_temp,
            self.temp_decay_rate,self.temp_decay_step)=temp_config
        if self.soften==True:
            assert 0.0<=self.temp_decay_rate<=1,"Decay rate in wrong range!!"

        #Now we will initialize the Latent slots
        self.latent_slots=[]
        for sidx in range(sparsity_factor):
            latent_slot=LatentSlot2(sp_dense_config=sp_dense_config,
                                    coef_config=coef_config,
                                    soften=self.soften,
                                    sample_strategy=sample_strategy)
            self.latent_slots.append(latent_slot)

        #Now we have to desin one special slot for the no-intervention
        self.sp_dense_layers=[]
        for units,actvn in sp_dense_config_base:
                layer=keras.layers.Dense(units,activation=actvn)
                self.sp_dense_layers.append(layer)
        #Adding the last layer to give us one single score
        self.sp_dense_layers.append(keras.layers.Dense(units=1,
                                                    activation=None))

    def call(self,encoder_hidden):
        '''
        '''
        #First of all we have to calculate the current temperature
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

        #Getting the predicted intervention from first-S slots
        interv_locs=[]
        interv_locs_prob=[]
        for sidx in range(self.sparsity_factor):
            interv_score,interv_loc=self.latent_slots[sidx](encoder_hidden,
                                                        self.temperature)

            interv_locs_prob.extend(interv_score)
            interv_locs.extend(interv_loc)

        #Now we will generate the contribution of the base distribution
        H=encoder_hidden
        for sp_layer in self.sp_dense_layers:
            H=sp_layer(H)
        #Now we will not pass it through sigmoid since we are normalizin ltr
        base_score=tf.reduce_mean(H)

        #Adding this acore and loc to our list
        interv_locs.append(((np.int32(self.base_num),),(np.int32(0),)))
        interv_locs_prob.append(base_score)

        #Now we will normalize the interv_loc_prob
        interv_locs_prob=tf.stack(interv_locs_prob,axis=0)
        print("sel_log_prog:",interv_locs_prob)
        interv_locs_prob=tf.nn.softmax(interv_locs_prob,axis=0)

        return interv_locs,interv_locs_prob


#############################################################################
################################# SLOT-1 ####################################
#############################################################################
class LatentSlot1(keras.layers.Layer):
    '''
    This class will design the representation and function for a particular,
    slot of the latent space. So, ultimately our latent space will contain
    S-of these slots.
    '''
    final_input=None        #This will be after additional dense is applied
    temperature=None        #Holding the current temperature
    tau=None                #The current cutoff value

    def __init__(self,coef_config,sp_dense_config,
                temp_config,global_step,smry_writer,
                sample_strategy,cutoff_config,**kwargs):
        '''
        coef_config     : same representation we used in previous Encoder
        sp_dense_config : config for further specialization of the shared inp
        '''
        super(LatentSlot1,self).__init__(**kwargs)
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
        print("node_decisions:",node_decision)



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
        for nidx,(cout_layer,num_cat) in enumerate(
                                zip(self.cout_layers,self.output_config)):
            #Getting a valid distribution among the nodes
            cout=cout_layer(H)
            #Now we will either soften if we want to or leave it as it is
            if self.soften==True:
                cout=cout/self.temperature

            cout=tf.nn.softmax(cout,axis=1)
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

        if self.sample_strategy=="top-k" or self.temperature.numpy()==1.0:
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

        return cat_prob,tf.squeeze(cat_idx)

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
                interv_nodes.append(np.int32(node))
                interv_cats.append(cat.numpy())
        interv_loc=(tuple(interv_nodes),tuple(interv_cats))

        return interv_loc

class LatentSpace1(keras.layers.Layer):
    '''
    This class will manage all the individual slots and give finally give
    us the intervention locations and their corresponding contribution,
    which will be then fed to decoder for calculating the logprobability.
    '''
    def __init__(self,sparsity_factor,
                coef_config,sp_dense_config,sp_dense_config_base,
                temp_config,global_step,smry_writer,
                sample_strategy,cutoff_config,**kwargs):
        super(LatentSpace1,self).__init__(**kwargs)
        self.sparsity_factor=sparsity_factor
        self.base_num=len(coef_config)-1

        #First of all we will initialize all the latent slots we need
        self.latent_slots=[]
        for sidx in range(sparsity_factor):
            latent_slot=LatentSlot1(coef_config=coef_config,
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
            #Excluding the location if none of the intervention is selected
            if len(interv_loc[0])==0 and len(interv_loc[1])==0:
                continue
            interv_locs.append(interv_loc)
            interv_locs_prob.append(interv_score)

        #Now we will generate score for the base-distirbution configuration
        H=shared_input
        for sp_layer in self.sp_dense_layers:
            H=sp_layer(H)
        #Now we will not pass it through sigmoid since we are normalizin ltr
        H=tf.nn.sigmoid(H)
        base_score=tf.reduce_mean(H)

        #Adding this acore and loc to our list
        interv_locs.append(((np.int32(self.base_num),),(np.int32(0),)))
        interv_locs_prob.append(base_score)

        #Now we will normalize the interv_loc_prob
        interv_locs_prob=tf.stack(interv_locs_prob,axis=0)
        interv_locs_prob=tf.nn.softmax(interv_locs_prob,axis=0)

        return interv_locs,interv_locs_prob
