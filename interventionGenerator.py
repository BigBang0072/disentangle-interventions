import numpy as np
np.random.seed(1)
from scipy.stats import dirichlet
import pdb
from pprint import pprint

class InterventionGenerator():
    '''
    This class will generate the random intervention for mixture generation
    possibly to cover the full space of interventions either with uniform
    probability or with some bias.

    Design:
        1. Given the sparsity |S| independently  we generate each component:
            1.1 Get the blacklisted set of cateogry to be consistent throughout
        2. For each component:
            2.1 Decide the number of nodes: |T|
                2.1.1 Uniform Probability
                2.1.2 inversly proportional to number of nodes (straight)
                2.1.3 with inverse probability (exponentially)
            2.2 Select |T| node from the graph:
                2.2.1 with Uniform Probability (every node is equally likely)
            2.3 For each of the node selected:
                2.3.1 Make it consistent with the blacklisted category
        3. Once all the targets location are ready: Generate pi's:
            3.1 Dirchlet distribution --> assymetric alpha
            3.2 Change the scale of alpha instead of direction
    '''

    def __init__(self,S,max_nodes,max_cat,
                    num_node_temperature,
                    pi_dist_type,pi_alpha_scale,
                    max_unique_tires=3):
        #Initializing the variable
        self.S = S
        self.max_nodes = max_nodes
        self.max_cat   = max_cat
        self.num_node_temperature = num_node_temperature
        self.pi_dist_type = pi_dist_type
        self.pi_alpha_scale = pi_alpha_scale
        self.max_unique_tires=max_unique_tires

    def generate_all_targets(self,):
        '''
        This function will be main point of comtact to generate the random
        intervention target for mixture generation.
        '''
        #Generating the blacklisted category for each of the node
        blist_cats = np.random.choice(self.max_cat,self.max_nodes,replace=True)
        print("blacklisted category of each node:")
        pprint(blist_cats)
        target_dict={}
        existing_locs={}
        for tidx in range(self.S):
            print("==========================================================")
            #Getting the target_location
            target_loc=self._generate_target_loc(blist_cats)
            num_try=0
            while ((target_loc in existing_locs)
                    and (num_try<self.max_unique_tires)):
                target_loc=self._generate_target_loc(blist_cats)
                num_try+=1
            #Now either we have unique or we skip trying
            if target_loc in existing_locs:
                continue
            else:
                #Got the unique location to park our Tesla
                target_dict["t{}".format(tidx)]=target_loc
                #Now removing the parking spot as its my spot!!
                existing_locs[target_loc]=True

        #Adding one empty target which will denote base distribution
        target_dict["t{}".format(self.S)]=[[],[]]

        #Now we add target pis for each of the target loc
        self._insert_pi_distribution(self.pi_dist_type,self.pi_alpha_scale,
                                    target_dict)

        print("Tragets Generated:")
        pprint(target_dict)
        return list(target_dict.values())

    def _generate_target_loc(self,blist_cats):
        '''
        This fucntion will generate one target location
        '''
        #First of all we have to decide on number of nodes in target
        num_node = self._get_num_node_in_target(self.num_node_temperature)

        #Now we will select these many nodes with uniform probability
        tnodes = np.random.choice(self.max_nodes,num_node,replace=False).tolist()

        #Now we have to choose the category for each of the node
        tcats = np.random.choice(self.max_cat-1,num_node,replace=True).tolist()
        tcats = [tcat if tcat<blist_cats[tnode] else tcat+1
                        for tnode,tcat in zip(tnodes,tcats)
                ]
        print("tnodes:{}".format(tnodes))
        print("tcats:{}".format(tcats))

        return (tuple(tnodes),tuple(tcats))

    def _get_num_node_in_target(self,temperature):
        '''
        This function will generate the number of nodes on the target based
        on the boltzamann distribution:
                exp(-n/T)
        For uniform probability of all node size give T = float("inf")
        '''
        num_dist = np.exp(-1.0*np.arange(self.max_nodes)/temperature)
        num_dist = num_dist/np.sum(num_dist)

        num_node = np.random.choice(np.arange(1,self.max_nodes+1),
                                    1,p=num_dist)[0]
        assert num_node<=self.max_nodes,"Number of nodes in target large"
        print("num_node:{}\tnum_dist:{}".format(num_node,num_dist))

        return num_node

    def _insert_pi_distribution(self,pi_dist_type,pi_alpha_scale,target_dict):
        '''
        This function will generate the distribution of pi for the target
        location. This comes in two flavour:
            1. uniform : alpha_vector = [1,1,...,1]
            2. inverse : alpha_vector = [1/|t1|, 1/||t2|, .... , 1\|tn|]*(|t|max)
        '''
        if pi_dist_type=="uniform":
            alpha = np.ones(len(target_dict))*pi_alpha_scale
            tpis = np.squeeze(dirichlet.rvs(size=1,alpha=alpha))
        elif pi_dist_type=="inverse":
            #Getting the target size list
            tsize = np.array(
                        [len(target_dict[tname][0])
                            if len(target_dict[tname][0])>0
                            else np.random.randint(self.max_nodes)+1
                                for tname in target_dict.keys()
                        ],
                        dtype=np.float32,
                    )
            max_tsize = np.max(tsize)
            #Getting the alpha
            alpha = (1/tsize)*max_tsize*pi_alpha_scale
            tpis = np.squeeze(dirichlet.rvs(size=1,alpha=alpha))
        else:
            raise NotImplementedError

        assert abs(np.sum(tpis)-1)<1e-10,"sum of pi/coefficient should be 1"
        #Now adding the tpis to the target locs
        for tidx,tname in enumerate(target_dict.keys()):
            target_dict[tname]=list(target_dict[tname])
            target_dict[tname].append(tpis[tidx])


if __name__=="__main__":
    target_generator = InterventionGenerator(S=10,
                                            max_nodes=5,
                                            max_cat=3,
                                            num_node_temperature=float("inf"),
                                            pi_dist_type="inverse",
                                            pi_alpha_scale=5)
    for idx in range(1000):
        target_generator.generate_all_targets()
    # target_generator.generate_all_targets()
