import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import glob
import pdb
from pprint import pprint
from evaluator import EvaluatePrediction

class Plotter():
    '''
    This class will be responsible for plotting all the metrics for each of the
    experiment.
    '''

    def __init__(self,experiment_id):
        self.experiment_id  =   experiment_id
        self.all_expt_json  =   self._get_expt_jsons(experiment_id)

    def _get_expt_jsons(self,experiment_id):
        '''
        This function will read all the successful experiments done till
        now from the xperiment directory.
        '''
        expt_fnames = glob.glob("{}/*.json".format(experiment_id))
        print("Number of Expt Read:{}".format(len(expt_fnames)))

        #Now we will read the jsons one by one
        all_expt_json=[]
        for fname in expt_fnames:
            with open(fname) as json_file:
                expt_json = json.load(json_file)
                all_expt_json.append(expt_json)

        return all_expt_json

    def plot_evaluation_metrics(self,group_criteria):
        '''
        Now we have dict of all the experiment, with its config and the
        evaluation metrics
        '''
        tsv_path = "{}/expt_df.tsv".format(self.experiment_id)
        try:
            print("Loading the cached dataframe")
            #Directly loading the df from excel
            expt_df=pd.read_csv(tsv_path,sep="\t")
            # pdb.set_trace()
        except:
            #Converting the results into df
            expt_df = pd.DataFrame(self.all_expt_json)
            # pdb.set_trace()
            expt_df = self._calculate_other_js_scores(expt_df)
            #Saving the dataframe for faster loading later
            expt_df.to_csv(tsv_path,sep="\t")

        print("Size of Expt_df:{}".format(expt_df.shape))
        print(expt_df.head())
        # pdb.set_trace()

        #Getting the unique sample sizes
        sample_sizes=expt_df.mixture_sample_size.unique().tolist()
        sample_sizes.sort()
        # sample_sizes.remove(float("inf"))

        #Getting the evaluation metric by the group criteria
        js_dict,mse_dict,gratio_dict=\
                    self._get_metric_vs_sample_by(group_criteria,expt_df)
        #Now we will be plotting the metrics
        self._plot_metrics(js_dict,mse_dict,gratio_dict,group_criteria,
                            sample_sizes)

        # pdb.set_trace()

    def _calculate_other_js_scores(self,expt_df):
        '''
        This function will calculate the js_node and js_all and corresponding
        mse error in addition to the js_weighted.
        '''
        #Creating the evaluator
        evaluator = EvaluatePrediction(matching_weight=0.0)

        #Adding additional columns to df
        expt_df  = expt_df.assign(
                                  js_node=pd.Series(np.zeros(expt_df.shape[0])),
                                  mse_node=pd.Series(np.zeros(expt_df.shape[0])),
                                  js_all=pd.Series(np.zeros(expt_df.shape[0])),
                                  mse_all=pd.Series(np.zeros(expt_df.shape[0])),
        )
        expt_df = expt_df.rename(columns={
                                        "js_score":"js_weighted",
                                        "avg_mse":"mse_weighted",
                                }
        )

        #Going through the samples one by one
        for job_no in range(expt_df.shape[0]):
            config = expt_df.iloc[job_no]

            #Filling the nan positions with worst possible score
            if np.isnan(config["js_weighted"]):
                # expt_df.at[job_no,"js_weighted"]=0.0
                # expt_df.at[job_no,"mse_weighted"]=2.0/config["sparsity"]
                #
                # expt_df.at[job_no,"js_node"]=0.0
                # expt_df.at[job_no,"mse_node"]=2.0/config["sparsity"]
                #
                # expt_df.at[job_no,"js_all"]=0.0
                # expt_df.at[job_no,"mse_all"]=2.0/config["sparsity"]

                continue

            #Now we are ready to calculate the other js
            actual_target_dict = config["actual_target_dict"]
            pred_target_dict = config["pred_target_dict"]

            # #Now we are ready for evaluation for js_node
            # evaluator.matching_weight=1.0
            # js_node,mse_node = evaluator.get_evaluation_scores(
            #                                 pred_target_dict,
            #                                 actual_target_dict.values()
            # )
            # expt_df.at[job_no,"js_node"]=js_node
            # expt_df.at[job_no,"mse_node"]=mse_node

            #Now we are ready to calcualte the js_all
            evaluator.matching_weight=0.0
            recall,precision,mse_all = evaluator.get_evaluation_scores(
                                            pred_target_dict,
                                            actual_target_dict.values()
            )
            expt_df.at[job_no,"recall"]=recall
            expt_df.at[job_no,"precision"]=precision
            expt_df.at[job_no,"fscore"]=2*(precision*recall)/(precision+recall)
            expt_df.at[job_no,"mse_all"]=mse_all

        return expt_df

    def _get_metric_vs_sample_by(self,group_criteria,expt_df):
        '''
        This function will plot the metrics with respect to sample along
        with a certain fine variable given by group_criteria
        '''
        #Defining the custom percentile function
        def quantile25(x):
            return x.quantile(0.20)
        def quantile75(x):
            return x.quantile(0.80)

        #Getting the evaluation criteria
        js_dict = {}
        mse_dict = {}
        gratio_dict={}

        #Iterating over all the fine group
        for gname,gdf in expt_df.groupby(group_criteria):
            percentage_size = gdf.shape[0]/expt_df.shape[0]
            print("gname:{}\t percent_size:{}".format(gname,
                                                percentage_size)
            )
            #Now we will get the eval metrics based on sample size
            js_variation={}
            js_variation["mean"]=gdf.groupby("mixture_sample_size")["recall"]\
                            .agg("mean").to_dict()
            js_variation["std"]=gdf.groupby("mixture_sample_size")["recall"]\
                            .agg("std").to_dict()
            js_variation["q25"]=gdf.groupby("mixture_sample_size")["recall"]\
                            .agg(quantile25).to_dict()
            js_variation["q75"]=gdf.groupby("mixture_sample_size")["recall"]\
                            .agg(quantile75).to_dict()
            print("js_variation:")
            pprint(js_variation)

            #Getting the mse variation
            mse_variation={}
            mse_variation["mean"]=gdf.groupby("mixture_sample_size")["mse_all"]\
                            .agg("mean").to_dict()
            mse_variation["std"]=gdf.groupby("mixture_sample_size")["mse_all"]\
                            .agg("std").to_dict()
            mse_variation["q25"]=gdf.groupby("mixture_sample_size")["mse_all"]\
                            .agg(quantile25).to_dict()
            mse_variation["q75"]=gdf.groupby("mixture_sample_size")["mse_all"]\
                            .agg(quantile75).to_dict()
            print("mse_varitaion")
            pprint(mse_variation)

            #Adding them to overall dict with resepect to refined group
            js_dict[gname]=js_variation
            mse_dict[gname]=mse_variation
            gratio_dict[gname]=percentage_size

        # pdb.set_trace()
        return js_dict,mse_dict,gratio_dict

    def _plot_metrics(self,js_dict,mse_dict,gratio_dict,
                            group_criteria,sample_sizes):
        '''
        Here we will plot the metrics in one single plot with level curve
        of the finer group criteria.
        '''
        #Now lets start creating the subplot one by once
        fig, ax = plt.subplots(1,2)

        #Plotting the JS
        self._plot_on_one_axis(ax[0],sample_sizes,js_dict,gratio_dict,True,
                                                            group_criteria)
        #Plotting the mse
        self._plot_on_one_axis(ax[1],sample_sizes,mse_dict,gratio_dict,False,
                                                            group_criteria)

        fig.suptitle("Evaluation Metrics with level-curve wrt : {}".format(
                                                            group_criteria))
        plt.show()


    def _plot_on_one_axis(self,ax,sample_sizes,metric_dict,gratio_dict,is_js,
                            group_criteria):
        #One by one plotting each of the level curve for each of group id
        xval = list(range(len(sample_sizes)))
        for gname in metric_dict.keys():
            #Preparing the variation info for this group
            variation_dict=metric_dict[gname]
            yval = np.array([variation_dict["mean"][size] for size in sample_sizes])
            yerr = [variation_dict["std"][size]/2 for size in sample_sizes]
            yq25 = [yval[sidx]-variation_dict["q25"][size]
                                for sidx,size in enumerate(sample_sizes)]
            yq75 = [variation_dict["q75"][size]-yval[sidx]
                                for sidx,size in enumerate(sample_sizes)]

            #preparing the level-curve name
            curve_name = "{0:}={1:} : expt_ratio={2:0.2f}".format(
                                                        group_criteria,
                                                        gname,
                                                        gratio_dict[gname])


            ax.errorbar(xval,yval,yerr=(yq25,yq75),fmt='o-',alpha=0.6,
                            capsize=5,capthick=2,linewidth=2,label=curve_name)

            #Plotting by filling in between the error
            ax.fill_between(xval,yval-yq25,yval+yq75,alpha=0.1)


        #Now we are done with the plotting, lets beautiy it
        xlabels = [str(size) for size in sample_sizes]
        ax.set_xticks(xval)
        ax.set_xticklabels(xlabels,rotation=45)
        ax.set_xlabel("Sample Size")

        #Setting the title for the y-axis
        if is_js:
            ax.legend(loc="upper left")
            ax.set_ylim(0,1.05)
            ax.set_ylabel("Average Weighted-Jaccard-Similarity")
        else:
            ax.legend(loc="upper right")
            #ax.set_ylim(0,2.0/4.0)
            ax.set_ylabel("Average MSE in Mixing Coefficient")

        #Setting the grid
        ax.grid(True)


if __name__=="__main__":
    experiment_id="gasinha-exp11"
    plotter = Plotter(experiment_id)
    plotter.plot_evaluation_metrics(group_criteria="graph_type")
