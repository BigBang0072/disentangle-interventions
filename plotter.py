import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import glob
import pdb
from pprint import pprint

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
        #Converting the results into df
        expt_df = pd.DataFrame(self.all_expt_json)
        print("Size of Expt_df:{}".format(expt_df.shape))
        print(expt_df.head())
        # pdb.set_trace()

        #Getting the unique sample sizes
        sample_sizes=expt_df.mixture_sample_size.unique()
        sample_sizes.sort()

        #Getting the evaluation metric by the group criteria
        js_dict,mse_dict,gratio_dict=\
                    self._get_metric_vs_sample_by(group_criteria,expt_df)
        #Now we will be plotting the metrics
        self._plot_metrics(js_dict,mse_dict,gratio_dict,group_criteria,
                            sample_sizes)

        # pdb.set_trace()

    def _get_metric_vs_sample_by(self,group_criteria,expt_df):
        '''
        This function will plot the metrics with respect to sample along
        with a certain fine variable given by group_criteria
        '''
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
            js_variation["mean"]=gdf.groupby("mixture_sample_size")["js_score"]\
                            .agg("mean").to_dict()
            js_variation["std"]=gdf.groupby("mixture_sample_size")["js_score"]\
                            .agg("std").to_dict()
            print("js_variation:")
            pprint(js_variation)

            #Getting the mse variation
            mse_variation={}
            mse_variation["mean"]=gdf.groupby("mixture_sample_size")["avg_mse"]\
                            .agg("mean").to_dict()
            mse_variation["std"]=gdf.groupby("mixture_sample_size")["avg_mse"]\
                            .agg("std").to_dict()
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
            yval = [variation_dict["mean"][size] for size in sample_sizes]
            yerr = [variation_dict["std"][size]/2 for size in sample_sizes]

            #preparing the level-curve name
            curve_name = "{0:}={1:} : expt_ratio={2:0.2f}".format(
                                                        group_criteria,
                                                        gname,
                                                        gratio_dict[gname])


            ax.errorbar(xval,yval,yerr=yerr,fmt='o-',alpha=0.6,
                            uplims=True,lolims=True,label=curve_name)

        #Now we are done with the plotting, lets beautiy it
        xlabels = [str(size) for size in sample_sizes]
        ax.set_xticks(xval)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Sample Size")

        #Setting the title for the y-axis
        if is_js:
            ax.legend(loc="upper left")
            ax.set_ylim(0,1.2)
            ax.set_ylabel("Average Weighted-Jaccard-Similarity")
        else:
            ax.legend(loc="upper right")
            ax.set_ylabel("Average MSE in Mixing Coefficient")

        #Setting the grid
        ax.grid(True)


if __name__=="__main__":
    experiment_id="gasinha-exp1"
    plotter = Plotter(experiment_id)
    plotter.plot_evaluation_metrics(group_criteria="graph_type")
