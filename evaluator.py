import numpy as np
from pprint import pprint

class EvaluatePrediction():
    '''
    This class will be responsible for all the evaluation we will be going
    to do. The current evaluation supports:
        1. max Jaccard Similiarity (JS) between prediction and actual
        2. MSE between the matched prection and actual.
    '''
    def __init__(self,matching_weight,):
        self.matching_weight = matching_weight
        assert 0<=matching_weight<=1,"give correct weight"

    def get_evaluation_scores(self,pred_target_dict,do_config):
        '''
        This function will be responsible for all the comparison between the
        predicted targets and actual targets and then give a single number
        score for them.
        '''
        #First of all lets convert all the data into required format
        actual_target_dict = self._get_actual_target_dict(do_config)
        self.actual_target_dict=actual_target_dict
        print("Actual Target Dict:")
        pprint(actual_target_dict)
        print("Predicted Target Dict:")
        pprint(pred_target_dict)

        #First of all we have to get the matching score between them
        match_score_list = self._match_actual_to_pred(pred_target_dict,
                                                actual_target_dict)
        print("Match Socre List:")
        pprint(match_score_list)

        #Now we can match the actual to predicted targets
        matched_actual_dict={}
        matched_pred_dict={}
        matched_score_list=[]
        matched_pi_diff=[]
        for act_name,pred_name,score in match_score_list:
            #Stopping if we have mathed all possible config
            if (len(matched_actual_dict)==len(actual_target_dict)
                    or len(matched_pred_dict)==len(pred_target_dict)):
                break
            #Skipping if the actual target or the rpedicted target has matched
            if (pred_name in matched_pred_dict
                    or act_name in matched_actual_dict):
                continue

            #Skipping of the matching score is == 0.0 (actually we could break as
            #all the score will be zero from here)
            if (score==0.0):
                break

            #Now based on sir's comment to consider only match made in heaven
            if (score<1.0):
                break

            #Now whatever we have it's a match
            print("Match: actual:{}\t pred:{}\t score:{}".format(act_name,
                                                            pred_name,
                                                            score))
            #Adding the score to the matched list
            matched_score_list.append(score)
            #Getting the pi's for mse
            pi_diff = actual_target_dict[act_name][-1]\
                            - pred_target_dict[pred_name][-1]
            matched_pi_diff.append(pi_diff)

            #Now adding the names to matched dict
            matched_actual_dict[act_name]=True
            matched_pred_dict[pred_name]=True

        #Now its possible that some of targets are not matched
        actual_left_pi_diff=[]
        if len(matched_actual_dict)!=len(actual_target_dict):
            self._add_unmatched_target_score(actual_target_dict,
                                            matched_actual_dict,
                                            matched_score_list,
                                            actual_left_pi_diff)

        #Getting the pis of targets which are not matched in prediction
        pred_left_pi_diff=[]
        if len(matched_pred_dict)!=len(pred_target_dict):
            self._add_unmatched_target_score(pred_target_dict,
                                            matched_pred_dict,
                                            matched_score_list,
                                            pred_left_pi_diff)

        #Now calculating the average score and mse
        recall = np.sum(matched_score_list)/len(actual_target_dict)
        precision = np.sum(matched_score_list)/len(pred_target_dict)

        #Now segregating the mean squared error
        mse_all = {}
        mse_all["mse_present_actual"]  = self._get_l2_norm(actual_left_pi_diff)
        mse_all["mse_present_pred"]    = self._get_l2_norm(pred_left_pi_diff)
        mse_all["mse_present_both"]    = self._get_l2_norm(matched_pi_diff)
        mse_all["mse_overall"]  = self._get_l2_norm(matched_pi_diff+actual_left_pi_diff+pred_left_pi_diff)
        #avg_mse = (np.mean(np.array(matched_pi_diff)**2))**(0.5)

        print("Score: recall:{}\t precision:{}\t avg_mse:{}".format(recall,precision,mse_all))

        return recall,precision,mse_all

    def _get_l2_norm(self,arr_list):
        return np.mean(np.array(arr_list)**2)**(0.5)

    def _add_unmatched_target_score(self,target_dict,matched_names,matched_score_list,matched_pi_diff):
        #Iterating over the target dict
        for tname,(_,_,pi) in target_dict.items():
            if tname not in matched_names:
                print("Adding socre for unmatched target:{}".format(tname))
                matched_score_list.append(0)
                matched_pi_diff.append(pi)
        return

    def _get_actual_target_dict(self,do_config):
        actual_target_dict={}
        total_pi=0.0
        for cidx,config in enumerate(do_config):
            actual_target_dict["at{}".format(cidx)]=config
            total_pi+=config[-1]

        #Adding the empty config if its not there
        if (1-total_pi)>1e-5:
            empty_config=[(),(),1-total_pi]
            actual_target_dict["at{}".format(len(actual_target_dict))]\
                                    = empty_config
        return actual_target_dict

    def _match_actual_to_pred(self,pred_dict,actual_dict):
        '''
        This function will match all the actual target to the predicted target
        and give us a list of targets based on their score.
        '''
        #Gogin over all the actual target
        match_score_list = []
        for act_name,actual_target in actual_dict.items():
            #Comparing it with all the predicted target
            for pred_name,pred_target in pred_dict.items():
                #Now caluclate the matching soce between the targets
                matching_score = self._get_matching_score(actual_target,
                                                            pred_target)
                #Saving the match score
                match_score_list.append([act_name,pred_name,matching_score])

        #Now we will sort the list based on the matching score
        match_score_list.sort(key = lambda x: x[-1],reverse=True)
        return match_score_list

    def _get_matching_score(self,actual_target,pred_target):
        '''
        This function will calcuate the matching socre between the two target
        using the following two criteria:
            1. simple:
                1.1 here we just calcualte the IOU for both the location
                1.2 JS_overall
            2. ncorrected:
                2.1 here we also give imprtance to the atleast match in the node
                2.2 (matching_weight)JS_node * (1-matching_weight)JS_overall
        '''
        #Calcualting the overall JS
        merged_actual_loc = set(zip(actual_target[0],actual_target[1]))
        merged_pred_loc   = set(zip(pred_target[0],pred_target[1]))
        if len(merged_actual_loc)+len(merged_pred_loc)==0:
            JS_overall=1.0
        else:
            JS_overall = len(merged_actual_loc.intersection(merged_pred_loc))\
                        /len(merged_actual_loc.union(merged_pred_loc))

        #Now calculating the node JS
        actual_nodes = set(actual_target[0])
        pred_nodes   = set(pred_target[0])
        if len(actual_nodes)+len(pred_nodes)==0:
            JS_node=1.0
        else:
            JS_node = len(actual_nodes.intersection(pred_nodes))\
                        /len(actual_nodes.union(pred_nodes))

        matching_score = self.matching_weight*JS_node + \
                            (1-self.matching_weight)*JS_overall
        assert 0<=matching_score<=1,"convex sum is in convex set"

        return matching_score
    
    def threshold_target_dict(self,target_dict,zero_threshold):
        '''
        This fucntion will threshold the target dictupto the given threshold
        such that all the targets below the threshold are made zero and the
        rest of the targets will be renormalized to be the valid probability.
        '''
        rest_sum = 0.0
        dead_target_names = []
        for tname,target_config in target_dict.items():
            if target_config[-1]<zero_threshold:
                dead_target_names.append(tname)
            else:
                rest_sum+=target_config[-1]
        
        #Now we will remove the targets which are useless
        for dtname in dead_target_names:
            del target_dict[dtname]
        
        #Now its term for renormalization
        for tname in target_dict.keys():
            target_config = list(target_dict[tname])
            target_config[-1] /=rest_sum
            
            target_dict[tname] = tuple(target_config)
        

        return target_dict

if __name__=="__main__":
    #now lets test our evaluator
    do_config=[
        [[0,1],[1,2],0.3]
    ]
    pred_target_dict={
        "t0":[[],[],0.6],
        "t1":[[0,],[2,],0.2]
    }

    #Now initializing our solver
    evaluator = EvaluatePrediction(matching_weight=0.5)
    evaluator.get_evaluation_scores(pred_target_dict,do_config)
