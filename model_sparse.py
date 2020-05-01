import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb
tf.random.set_seed(211)
np.random.seed(211)
from pprint import pprint
from operator import itemgetter

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
    actual_dict=None        #For keeping the hash of doconfig and pi

    def __init__(self,oracle,global_step,smry_writer,do_config,**kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.oracle=oracle
        self.global_step=global_step
        self.smry_writer=smry_writer

        #Sorting the doconfig for comparison later
        nodes,cats,pis=zip(*do_config)
        #Now lets first sort them
        nodes,cats=zip(*self._sort_loc_order(list(zip(nodes,cats))))
        self.do_config=list(zip(nodes,cats,pis))

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

        #Calculating important metrics
        interv_locs=self._sort_loc_order(interv_locs)
        self._calculate_doRecall(interv_locs,interv_locs_prob,self.do_config)
        self._calculate_doMAE(interv_locs,interv_locs_prob,self.do_config)

        return sample_logprob


    def _calculate_sample_likliehood(self,interv_loc_prob,sample_loc_prob,tolerance=1e-10):
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

    def _calculate_doRecall(self,interv_locs,interv_probs,do_config):
        '''
        recall@I, i.e number of guys from the do-config in top-I output
        recall@S, number of guys from do-config who are atleast in S predicted
        '''
        #Keep inportant variables ready
        inodes,icats,pis=zip(*do_config)
        phi=1-sum(pis)
        last_node=len(self.oracle.topo_i2n)
        if phi>0:
            inodes.append((last_node,))
            icats.append((0,))
        #Getting total number intervention
        do_config=list(zip(inodes,icats))
        num_I=len(do_config)*1.0

        #First of all we will have to sort the intervention based on prob
        interv_probs=interv_probs.numpy().tolist()
        predicted_dict={}
        for tidx,nodes_cats in enumerate(interv_locs):
            #Merging the splitting of same prediction at multiple places
            if nodes_cats in predicted_dict:
                predicted_dict[nodes_cats]+=interv_probs[tidx]
            else:
                predicted_dict[nodes_cats]=interv_probs[tidx]
        #Creating the node,cats and prob in one list only
        ncp_list=[(key[0],key[1],value)
                        for key,value in predicted_dict.items()]
        #Now we will sort them based on their logprobability
        ncp_list=sorted(ncp_list,key=itemgetter(2),reverse=True)

        #Now we are ready to calculate recall@I
        num_present=0.0
        max_len=min(len(ncp_list),int(num_I))
        for idx in range(max_len):
            if ncp_list[idx][0:2] in do_config:
                num_present+=1
        recall_at_I=num_present/num_I

        #Now calculating the recall@S
        num_present=0.0
        for idx in range(len(ncp_list)):
            if ncp_list[idx][0:2] in do_config:
                num_present+=1
        recall_at_S=num_present/len(interv_locs) #We want to penalize split

        #Adding the result to tensorboard
        with self.smry_writer.as_default():
            tf.summary.scalar("Recall@I",recall_at_I,
                                step=int(self.global_step.value()))
            tf.summary.scalar("Recall@S",recall_at_S,
                                step=int(self.global_step.value()))

        #Adding the metrics to print also
        self.add_metric(recall_at_I,name="Recall@I",aggregation="mean")
        self.add_metric(recall_at_S,name="Recall@S",aggregation="mean")

        return recall_at_I,recall_at_S

    def _calculate_doMAE(self,interv_locs,interv_probs,do_config):
        '''
        Now we will calculate the MAE for the interventions done on this
        graph with their predicted value.
        '''
        #Lets first hash the do_config
        if self.actual_dict==None:
            actual_dict={}
            pi_sum=0.0
            for nodes,cats,pi in do_config:
                actual_dict[(nodes,cats)]=pi
                pi_sum+=pi
            if 1-pi_sum>0.0:
                last_node=len(self.oracle.topo_i2n)
                actual_dict[((last_node,),(0,))]=1-pi_sum
            #Saving it for later so that we dont recompute
            self.actual_dict=actual_dict
        actual_dict=self.actual_dict

        #Now lets hash the predicted locations
        interv_probs=interv_probs.numpy().tolist()
        predicted_dict={}
        for tidx,nodes_cats in enumerate(interv_locs):
            #Merging the splitting of same prediction at multiple places
            if nodes_cats in predicted_dict:
                predicted_dict[nodes_cats]+=interv_probs[tidx]
            else:
                predicted_dict[nodes_cats]=interv_probs[tidx]

        #Now we will calculate the MAE
        adiff_pis=[]
        # pdb.set_trace()
        for nodes_cats,pi in actual_dict.items():
            if nodes_cats in predicted_dict:
                pi_hat=predicted_dict[nodes_cats]
                adiff=abs(pi_hat-pi)
                adiff_pis.append(adiff)
            else:
                adiff=pi
                adiff_pis.append(pi)
            #Writing individual contribution to tensorboard
            with self.smry_writer.as_default():
                name="{}:{}".format(nodes_cats,pi)
                tf.summary.scalar(name,adiff,
                                    step=int(self.global_step.value()))
        #Now we will calculate the average trend of MSE
        mean_adiff=np.mean(adiff_pis)
        # pdb.set_trace()
        with self.smry_writer.as_default():
            tf.summary.scalar("mae",mean_adiff,
                                step=int(self.global_step.value()))

        #Adding this as metric
        self.add_metric(mean_adiff,name="MAE",aggregation="mean")
        return mean_adiff

    def _sort_loc_order(self,locations):
        '''
        This function will sort the location prediction and do configs
        to match them and not leae them if they are permuted.
        '''
        new_locations=[]
        for nodes,cats in locations:
            node_cat=list(zip(nodes,cats))
            node_cat=sorted(node_cat,key=itemgetter(0))
            #Now unzipping our nodes and cats
            nodes,cats=zip(*node_cat)
            new_locations.append((nodes,cats))
        return new_locations


class AutoEncoder(keras.Model):
    '''
    We will combine both the Encoder and Decoder in one for this front,
    interface of our whole model.
    '''
    def __init__(self,dense_config,sparsity_factor,
                coef_config,sp_dense_config,sp_dense_config_base,
                temp_config,smry_writer,
                sample_strategy,cutoff_config,
                oracle,do_config,**kwargs):
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
                            smry_writer=smry_writer,
                            do_config=do_config)

    def call(self,inputs):
        #Getting the intervention location and mixture probability
        interv_locs,interv_locs_prob=self.encoder(inputs)

        #Now getting the likliehood of the sample
        samples_logprob=self.decoder(interv_locs,interv_locs_prob,inputs)

        #Incrementing the global step
        self.global_step.assign_add(1.0)

        return samples_logprob
