from distil.active_learning_strategies.strategy import Strategy
from torch.utils.data import DataLoader, Subset

import gc
import math
import multiprocessing as mp
import submodlib
import torch

def get_class_subset(dataset, class_to_retrieve, batch_size=64):

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    subset_idxs = []
    eval_idxs = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):

            matching_class_batch_idxs = torch.where(labels==class_to_retrieve)[0]
            matching_class_batch_idxs = matching_class_batch_idxs + eval_idxs
            subset_idxs.extend(matching_class_batch_idxs)
            eval_idxs += len(labels)

    return Subset(dataset, subset_idxs)

def parallel_select(self_strategy, class_num, budget, data_sijs, unlabeled_data_embedding):
    
    with open(F"/content/select{class_num}", "w") as f:
        f.write("heeey")
    
    # Get hyperparameters from args dict
    optimizer = self_strategy.args['optimizer'] if 'optimizer' in self_strategy.args else 'NaiveGreedy'
    metric = self_strategy.args['metric'] if 'metric' in self_strategy.args else 'cosine'
    eta = self_strategy.args['eta'] if 'eta' in self_strategy.args else 1
    gradType = self_strategy.args['gradType'] if 'gradType' in self_strategy.args else "bias_linear"
    stopIfZeroGain = self_strategy.args['stopIfZeroGain'] if 'stopIfZeroGain' in self_strategy.args else False
    stopIfNegativeGain = self_strategy.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self_strategy.args else False
    verbose = self_strategy.args['verbose'] if 'verbose' in self_strategy.args else False
    embedding_type = self_strategy.args['embedding_type'] if 'embedding_type' in self_strategy.args else "gradients"
    if(embedding_type=="features"):
        layer_name = self_strategy.args['layer_name'] if 'layer_name' in self_strategy.args else "avgpool"

    # Calculate the class budget to use in this selection
    budgets_to_use = [(budget * i) // self_strategy.target_classes for i in range(self_strategy.target_classes + 1)]
    class_budget = budgets_to_use[class_num + 1] - budgets_to_use[class_num]

    # Get all points in the query set that have sel_class as a label
    query_class_subset = get_class_subset(self_strategy.query_dataset, class_num)

    # Compute the feature embedding of this subset
    if(embedding_type == "gradients"):
        query_embedding = self_strategy.get_grad_embedding(query_class_subset, False, gradType)
    elif(embedding_type == "features"):
        query_embedding = self_strategy.get_feature_embedding(query_class_subset, False, layer_name)
    else:
        raise ValueError("Provided representation must be one of gradients or features")

    # Compute query-query kernel for LogDetMI if applicable
    if(self_strategy.args['smi_function']=='logdetmi'):
        query_query_sijs = submodlib.helper.create_kernel(X=query_embedding.cpu().numpy(), metric=metric, method="sklearn")
    else:
        query_query_sijs = None

    # Compute image-query kernel. Always needed.
    query_sijs = submodlib.helper.create_kernel(X=query_embedding.cpu().numpy(), X_rep=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")


    if(self_strategy.args['smi_function']=='fl1mi'):
        obj = submodlib.FacilityLocationMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                  num_queries=query_embedding.shape[0], 
                                                                  data_sijs=data_sijs , 
                                                                  query_sijs=query_sijs, 
                                                                  magnificationEta=eta)

    if(self_strategy.args['smi_function']=='fl2mi'):
        obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                  num_queries=query_embedding.shape[0], 
                                                                  query_sijs=query_sijs, 
                                                                  queryDiversityEta=eta)
    
    if(self_strategy.args['smi_function']=='com'):
        from submodlib_cpp import ConcaveOverModular
        obj = submodlib.ConcaveOverModularFunction(n=unlabeled_data_embedding.shape[0],
                                                                  num_queries=query_embedding.shape[0], 
                                                                  query_sijs=query_sijs, 
                                                                  queryDiversityEta=eta,
                                                                  mode=ConcaveOverModular.logarithmic)
    if(self_strategy.args['smi_function']=='gcmi'):
        obj = submodlib.GraphCutMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                  num_queries=query_embedding.shape[0],
                                                                  query_sijs=query_sijs, 
                                                                  metric=metric)
    if(self_strategy.args['smi_function']=='logdetmi'):
        lambdaVal = self_strategy.args['lambdaVal'] if 'lambdaVal' in self_strategy.args else 1
        obj = submodlib.LogDeterminantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                num_queries=query_embedding.shape[0],
                                                                data_sijs=data_sijs,  
                                                                query_sijs=query_sijs,
                                                                query_query_sijs=query_query_sijs,
                                                                magnificationEta=eta,
                                                                lambdaVal=lambdaVal)

    greedyList = obj.maximize(budget=class_budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                          stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)

    return greedyList

class SMIAutoLabeler(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, query_dataset, net, nclasses, args={}): #
        
        super(SMIAutoLabeler, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        
        self.query_dataset = query_dataset
        if "thread_count" not in args:
            args["thread_count"] = 5

    def _resolve_conflicts(self, greedy_list1, greedy_list2):

        # Get a position mapping for each list
        index_map1 = {greedy_list1[i][0]: i for i in range(len(greedy_list1))}
        index_map2 = {greedy_list2[i][0]: i for i in range(len(greedy_list2))}
	
        # Get all conflicting indices
        index_set1 = set([elem[0] for elem in greedy_list1])
        index_set2 = set([elem[0] for elem in greedy_list2])
        conflicting_indices = index_set1.intersection(index_set2)

        for conflicting_index in conflicting_indices:
	
            # Get the marginal gains of each element at the conflicting index
            greedy_list_index1 = index_map1[conflicting_index]
            greedy_list_index2 = index_map2[conflicting_index]
            _, marg_gain1 = greedy_list1[greedy_list_index1]
            _, marg_gain2 = greedy_list2[greedy_list_index2]

		        # Mark for deletion via -1 index
            if marg_gain1 > marg_gain2:
                greedy_list2[greedy_list_index2] = (-1,marg_gain2)
            else:
                greedy_list1[greedy_list_index1] = (-1,marg_gain1)

	      # Go through each list, working backwards, deleting marked locations
        for i in range(len(greedy_list1) - 1, -1, -1):
            if greedy_list1[i][0] == -1:
                del greedy_list1[i]
	
        for i in range(len(greedy_list2) - 1, -1, -1):
            if greedy_list2[i][0] == -1:
                del greedy_list2[i]

    def select(self, budget):
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	

        self.model.eval()

        # Get hyperparameters from args dict
        metric = self.args['metric'] if 'metric' in self.args else 'cosine'
        gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "gradients"
        if(embedding_type=="features"):
            layer_name = self.args['layer_name'] if 'layer_name' in self.args else "avgpool"

        # Compute unlabeled set embeddings
        if(embedding_type == "gradients"):
            unlabeled_data_embedding = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
        elif(embedding_type == "features"):
            unlabeled_data_embedding = self.get_feature_embedding(self.unlabeled_dataset, True, layer_name)
        else:
            raise ValueError("Provided representation must be one of gradients or features")
        
        # Compute image-image kernel. We only compute this once. It forms most of the running time.
        if(self.args['smi_function']=='fl1mi' or self.args['smi_function']=='logdetmi'): 
            data_sijs = submodlib.helper.create_kernel(X=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        else:
            data_sijs = None

        # =====================
        # BEGIN MULTIPROC
        # =====================

        process_argument_list = []
        worker_pool = mp.Pool(processes=self.args['thread_count'])

        for sel_class in range(self.target_classes):
            process_argument_list.append((self, sel_class, budget, data_sijs, unlabeled_data_embedding))

        selected_idx = worker_pool.map(parallel_select, process_argument_list)

        print(selected_idx)

        for i in range(self.target_classes):
            for j in range(i+1, self.target_classes):
                self._resolve_conflicts(selected_idx[i],selected_idx[j])

        return selected_idx

class PartitionStrategy(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}, query_dataset=None, private_dataset=None): #
        
        super(PartitionStrategy, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if "num_partitions" not in args:
            self.num_partitions = 1
        else:
            self.num_partitions = args["num_partitions"]
            
        if "wrapped_strategy_class" not in args:
            raise ValueError("args dictionary requires 'wrapped_strategy_class' key")
            
        self.wrapped_strategy_class = args["wrapped_strategy_class"]
        self.query_dataset = query_dataset
        self.private_dataset = private_dataset

    def select(self, budget):
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	
        
        # The number of partitions should be less than or equal to the budget.
        # This is because the budget is evenly divided among the partitions (roughly),
        # so having a smaller budget than the number of partitions results in one or 
        # more partitions having a 0 budget, which should not happen.
        if self.num_partitions > budget:
            raise ValueError("Budget cannot be less than the number of partitions!")
        
        # Furthermore, the number of partitions cannot be more than the size of the unlabeled set
        if self.num_partitions > len(self.unlabeled_dataset):
            raise ValueError("There cannot be more partitions than the size of the dataset!")
    
        # Calculate partition splits and budgets for each partition
        full_unlabeled_size = len(self.unlabeled_dataset)
        split_indices = [math.ceil(full_unlabeled_size * ((1+x) / self.num_partitions)) for x in range(self.num_partitions)]        
        partition_budget_splits = [math.ceil(budget * ((1+x) / self.num_partitions)) for x in range(self.num_partitions)]
                  
        beginning_split = 0
        
        selected_idx = [[] for x in range(self.target_classes)]
        
        for i in range(self.num_partitions):
            
            print("PARTITION", i)

            end_split = split_indices[i]
            
            # Create a subset of the original unlabeled dataset as a partition.
            partition_index_list = list(range(beginning_split, end_split))
            current_partition = Subset(self.unlabeled_dataset, partition_index_list)
            
            # Calculate the budget for this partition
            if i == 0:
                partition_budget = partition_budget_splits[i]
            else:
                partition_budget = partition_budget_splits[i] - partition_budget_splits[i - 1]
                
            # With the new subset, create an instance of the wrapped strategy and call its select function.
            if(self.query_dataset != None and self.private_dataset != None):
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.query_dataset, self.private_dataset, self.model, self.target_classes, self.args)
            elif(self.query_dataset != None):
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.query_dataset, self.model, self.target_classes, self.args)
            elif(self.private_dataset != None):
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.private_dataset, self.model, self.target_classes, self.args)
            else:
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.model, self.target_classes, self.args)
            selected_partition_idxs = wrapped_strategy.select(partition_budget)
            
            # Use the partition_index_list to map the selected indices w/ respect to the current partition to the indices w/ respect to the dataset
            for j, selected_per_class_idx in enumerate(selected_partition_idxs):
                new_list = []
                for (k, associated_gain) in selected_per_class_idx:
                    new_list.append((partition_index_list[k], associated_gain))

                selected_partition_idxs[j] = new_list
            
                selected_idx[j].extend(selected_partition_idxs[j])
            beginning_split = end_split
            
            del wrapped_strategy
            gc.collect()

        # Return the selected idx
        return selected_idx