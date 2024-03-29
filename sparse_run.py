import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb
tf.random.set_seed(211)
np.random.seed(211)

from data_handle import BnNetwork
from model_sparse import AutoEncoder

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
                sparsity_factor=trainer_config["sparsity_factor"],
                coef_config=trainer_config["coef_config"],
                sp_dense_config=trainer_config["sp_dense_config"],
                sp_dense_config_base=trainer_config["sp_dense_config_base"],
                temp_config=trainer_config["temp_config"],
                smry_writer=smry_writer,
                sample_strategy=trainer_config["sample_strategy"],
                cutoff_config=trainer_config["cutoff_config"],
                oracle=oracle,
                do_config=trainer_config["do_config"])
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
    doRecalls_I=[]
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
        doRecall_I=float(model.metrics[0].result())
        doRecalls_I.append(doRecall_I)

        doRecall_S=float(model.metrics[1].result())

        doMAE=float(model.metrics[2].result())
        doMAEs.append(doMAE)
        model.reset_metrics()
        # pdb.set_trace()

        #Printing ocassionally
        if step%trainer_config["verbose"]==0:
            print(
            "step:{0:} loss:{1:0.5f} recall@I:{2:0.5f} recall@S:{3:0.5f} mae:{4:0.5f}\n\n".
                                    format(step,loss,doRecall_I,doRecall_S,doMAE))

        #Stop after certain number of epochs
        # if step>=trainer_config["epochs"]:
        #     break
    return losses

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("-gname",dest="graph_name")
    parser.add_argument("-dsize",dest="sample_size",type=int)
    parser.add_argument("-ssize",dest="shuffle_buffer",type=int)
    parser.add_argument("-bsize",dest="batch_size",type=int)
    parser.add_argument("-sfactor",dest="sparsity_factor",type=int)
    parser.add_argument("-epochs",dest="epochs",type=int)
    parser.add_argument("-rnum",dest="rnum")
    args=parser.parse_args()

    #Setting up the parameters for the dataset
    graph_name=args.graph_name
    modelpath="dataset/{}/{}.bif".format(graph_name,graph_name)
    do_config=[
                ((2,0),(0,1),0.2),
                ((6,3),(1,1),0.3),
                ((7,4),(0,0),0.3),
                ((5,1),(1,0),0.2)
            ]

    #Deciding the configuration of the encoder
    dense_config=[
                    [100,"relu"],
                    [100,"relu"]
                ]
    sp_dense_config=[
                    [100,"relu"]
                ]
    sp_dense_config_base=[
                ]

    #Setting up the configuration for the model
    trainer_config={}
    trainer_config["modelpath"]=modelpath
    trainer_config["do_config"]=do_config
    trainer_config["sample_size"]=args.sample_size
    trainer_config["savepath"]=None

    #Training related parameters
    trainer_config["shuffle_buffer"]=args.shuffle_buffer
    trainer_config["batch_size"]=args.batch_size
    trainer_config["dense_config"]=dense_config
    trainer_config["sp_dense_config"]=sp_dense_config
    trainer_config["sp_dense_config_base"]=sp_dense_config_base
    trainer_config["sparsity_factor"]=args.sparsity_factor
    trainer_config["learning_rate"]=1e-3
    trainer_config["decay_rate"]=1e-4
    trainer_config["verbose"]=1
    trainer_config["epochs"]=args.epochs

    #Parameters for sampling from the latent space
    soften=True                                #for using temperature
    trainer_config["sample_strategy"]="gumbel"   #top-k or gumbel
    init_temp=1024
    temp_decay_rate=0.5
    #num step per epoch,so in (1/2)^10 = 1024
    temp_delay=(args.epochs/10.0)*0.5        #after half of all epoch we chill
    temp_decay_step=temp_delay*(args.sample_size*1.0/args.batch_size)
    trainer_config["temp_config"]=[soften,init_temp,
                                    temp_decay_rate,temp_decay_step]

    #Parameters to control the cutoff value
    tau_max=0.1
    scale_factor=30
    trainer_config["cutoff_config"]=[tau_max,scale_factor]

    #Variables for tensorboard summary
    trainer_config["rnum"]=args.rnum
    trainer_config["smry_path"]="temp/sparse/{}/{}/".format(graph_name,
                                                    trainer_config["rnum"])

    #Calling the trainer
    trainer(trainer_config)
