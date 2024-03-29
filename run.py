import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb
tf.random.set_seed(211)
np.random.seed(211)

from data_handle import BnNetwork
from model import AutoEncoder

def get_coef_config(oracle):
    '''
    This function will generate the coef config for the latent space of
    encoder.
    '''
    coef_config=[]
    for idx in range(len(oracle.topo_i2n)):
        node=oracle.topo_i2n[idx]
        card=oracle.card_node[node]

        coef_config.append(card)

    #Adding one spot for initial distribution
    coef_config.append(1)

    return coef_config

def trainer(trainer_config):
    '''
    This is the call point of training our AutoEncoder.
    '''
    #First of all we will create out dataset
    oracle=BnNetwork(trainer_config["modelpath"])
    #Now we will generate the samples
    dataset=oracle.generate_sample_from_mixture(trainer_config["do_config"],
                                                trainer_config["sample_size"],
                                                trainer_config["savepath"])
    #Preparing the dataset as one-hot encoding
    dataset=oracle.encode_sample_one_hot(dataset)
    print("Dataset created with shape:",dataset.shape)
    #Now we will convert this numpy array into tensorflow dataset
    dataset=tf.data.Dataset.from_tensor_slices(dataset)
    dataset=dataset.repeat(trainer_config["epochs"])
    dataset=dataset.shuffle(trainer_config["shuffle_buffer"])
    dataset=dataset.batch(trainer_config["batch_size"])

    #Creating the tensorboard summary writer
    smry_writer=tf.summary.create_file_writer(trainer_config["smry_path"])

    #Now we create the model
    trainer_config["coef_config"]=get_coef_config(oracle)
    model=AutoEncoder(dense_config=trainer_config["dense_config"],
                        coef_config=trainer_config["coef_config"],
                        sparsity_factor=trainer_config["sparsity_factor"],
                        oracle=oracle,
                        do_config=trainer_config["do_config"],
                        temp_config=trainer_config["temp_config"],
                        sample_strategy=trainer_config["sample_strategy"],
                        smry_writer=smry_writer)
    #Creating our optimizer
    optimizer=tf.keras.optimizers.Adam(trainer_config["learning_rate"],
                                    decay=trainer_config["decay_rate"])

    #Now we are ready to run our training loop
    def run_training_step(X,model,oracle,optimizer):
        '''
        This function will run a single step of gradient update.
        '''
        with tf.GradientTape() as tape:
            #Getting the sample lop porbability
            samples_logprob=model(X)
            #Now we define our loss as negative log-likliehood
            loss=samples_logprob*(-1)

        #Calculating the gradient
        grads=tape.gradient(loss,model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        return loss

    #Starting to enumerate over the dataset
    losses=[]
    doRecalls=[]
    doMAEs=[]
    print("Starting the training steps:")
    for step,X in enumerate(dataset):
        #Running the training one step
        loss=run_training_step(X,model,oracle,optimizer)
        losses.append(float(loss))
        #Adding the loss to the summary
        with smry_writer.as_default():
            tf.summary.scalar("loss",loss,step=int(model.global_step.value()))
            #Adding summary for learning rate
            lr=(trainer_config["learning_rate"]
                /(1+trainer_config["decay_rate"]*model.global_step.value()))
            tf.summary.scalar("lr",lr,step=int(model.global_step.value()))

        #Getting the metrics from the model
        doRecall=float(model.metrics[0].result())
        doRecalls.append(doRecall)
        doMAE=float(model.metrics[1].result())
        doMAEs.append(doMAE)
        model.reset_metrics()
        # pdb.set_trace()

        #Printing ocassionally
        if step%trainer_config["verbose"]==0:
            print(
            "step:{0:} loss:{1:0.5f} doRecall:{2:0.5f} doMAE:{3:0.5f}\n\n".
                                    format(step,loss,doRecall,doMAE))

        #Stop after certain number of epochs
        # if step>=trainer_config["epochs"]:
        #     break
    return losses

if __name__=="__main__":
    #Setting up the parameters for the dataset
    graph_name="asia"
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    do_config=[
                ((0,),(0,),0.2),
                ((0,),(1,),0.3),
                ((7,),(0,),0.3),
                ((7,),(1,),0.2)
            ]

    #Deciding the configuration of the encoder
    dense_config=[
                    [100,"relu"],
                    [100,"relu"]
                ]

    #Setting up the configuration for the model
    trainer_config={}
    trainer_config["modelpath"]=modelpath
    trainer_config["do_config"]=do_config
    trainer_config["sample_size"]=10000
    trainer_config["savepath"]=None

    #Training related parameters
    trainer_config["shuffle_buffer"]=5000
    trainer_config["batch_size"]=1000
    trainer_config["dense_config"]=dense_config
    trainer_config["sparsity_factor"]=17
    trainer_config["learning_rate"]=1e-3
    trainer_config["decay_rate"]=1e-4
    trainer_config["verbose"]=5
    trainer_config["epochs"]=20

    #Parameters for sampling from the latent space
    soften=False                                #for using temperature
    trainer_config["sample_strategy"]="top-k"   #top-k or gumbel
    init_temp=1000
    temp_decay_rate=0.5
    temp_decay_step=10          #sample_size/batch_size = num steps per epoch
    trainer_config["temp_config"]=[soften,init_temp,
                                    temp_decay_rate,temp_decay_step]

    #Variables for tensorboard summary
    trainer_config["rnum"]="p2.3.2"
    trainer_config["smry_path"]="temp/dense/{}/new/{}/".format(graph_name,
                                                    trainer_config["rnum"])

    #Calling the trainer
    trainer(trainer_config)
