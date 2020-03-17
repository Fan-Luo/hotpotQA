# ref: https://github.com/dsgissin/DiscriminativeActiveLearning
import os
from prepro import prepro
from run import train, run_predict_unlabel, run_evaluate_dev
import argparse
import numpy as np
import random
import torch
import gc
from util import get_buckets
import operator
import time
import shutil

def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(len(X_train))[np.logical_not(np.in1d(np.arange(len(X_train)), labeled_idx))]

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
            
# class QueryMethod:
#     """
#     A general class for query strategies, with a general method for querying examples to be labeled.
#     """

#     def __init__(self, model):
#         self.model = model

#     def query(self, X_train, Y_train, labeled_idx, amount):
#         """
#         get the indices of labeled examples after the given amount have been queried by the query strategy.
#         :param X_train: the training set
#         :param Y_train: the training labels
#         :param labeled_idx: the indices of the labeled examples
#         :param amount: the amount of examples to query
#         :return: the new labeled indices (including the ones queried)
#         """
#         return NotImplemented

#     def update_model(self, new_model):
#         del self.model
#         gc.collect()
#         self.model = new_model


class RandomSampling():
    """
    A random sampling query strategy baseline.
    """
    def __init__(self):
        pass
    # def __init__(self, model):
    #     super().__init__(model)

    def query(self, config, X_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))


class UncertaintySampling():
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
    """

    # def __init__(self, model):
    #     super().__init__(model)
    def __init__(self):
        pass
    def query(self, config, X_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_train_datapoints = list(operator.itemgetter(*unlabeled_idx)(X_train))    
        predictions = run_predict_unlabel(config, [unlabeled_train_datapoints] )
        
        # compare predictions[qids] with unlabeled_train_datapoints['id'] to ensure the ordered is same 
        print("predictions[qids][:10]", predictions[qids][:10])
        print("predictions[qids].shape", predictions[qids].shape)
        print("len(unlabeled_train_datapoints)", len(unlabeled_train_datapoints))
        print("unlabeled_train_datapoints[:10]", unlabeled_train_datapoints[:10])
        
        logit1_score = np.amax(predictions['softmax_logit1'], axis=-1)
        logit2_score = np.amax(predictions['softmax_logit2'], axis=-1)
        type_score = np.amax(predictions['softmax_type'], axis=-1)
        top2_sp_score = np.take_along_axis(predictions['predict_support_np'], np.argsort(predictions['predict_support_np'])[:,-2:], axis=-1)
        sp_score = np.average(top2_sp_score , axis =-1)
        
        unlabeled_predictions =  logit1_score + logit2_score + type_score + top2_sp_score + sp_score

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class CombinedSampling():
    """
    An implementation of a query strategy which naively combines two given query strategies, sampling half of the batch
    from one strategy and the other half from the other strategy.
    """
    
    def __init__(self, method1, method2):
        self.method1 = method1()
        self.method2 = method2()

    def query(self, config, X_train, labeled_idx, amount):
        labeled_idx = self.method1.query(config, X_train, labeled_idx, int(amount/2))
        return self.method2.query(config, X_train, labeled_idx, int(amount/2))

    def update_model(self):
        # del self.model
        # gc.collect()
        # self.model = new_model
        self.method1.update_model()
        self.method2.update_model()


def get_initial_idx(X_train, initial_idx_path, initial_size, seed):
    # load initial indices:
    if initial_idx_path is not None:
        idx_path = os.path.join(initial_idx_path, '{size}_{seed}.pkl'.format(size=initial_size, seed=seed))
        with open(idx_path, 'rb') as f:
            labeled_idx = pickle.load(f)
    else:
        print("No Initial Indices Found - Drawing Random Indices...")
        labeled_idx = np.random.choice(len(X_train), initial_size, replace=False)
    return labeled_idx


def set_query_method(method_name, method2_name=None):
    # set the first query method:
    if method_name == 'Random':
        method = RandomSampling
    # elif method == 'CoreSet':
    #     method = CoreSetSampling
    # elif method == 'CoreSetMIP':
    #     method = CoreSetMIPSampling
    # elif method == 'Discriminative':
    #     method = DiscriminativeSampling
    # elif method == 'DiscriminativeLearned':
    #     method = DiscriminativeRepresentationSampling
    # elif method == 'DiscriminativeAE':
    #     method = DiscriminativeAutoencoderSampling
    # elif method == 'DiscriminativeStochastic':
    #     method = DiscriminativeStochasticSampling
    elif method_name == 'Uncertainty':
        method = UncertaintySampling
    # elif method == 'Bayesian':
    #     method = BayesianUncertaintySampling
    # elif method == 'UncertaintyEntropy':
    #     method = UncertaintyEntropySampling
    # elif method == 'BayesianEntropy':
    #     method = BayesianUncertaintyEntropySampling
    # elif method == 'EGL':
    #     method = EGLSampling
    # elif method == 'Adversarial':
    #     method = AdversarialSampling

    # set the second query method:
    if method2_name is not None:
        print("Using Two Methods...")
        if method2_name == 'Random':
            method2 = RandomSampling
        # elif method2 == 'CoreSet':
        #     method2 = CoreSetSampling
        # elif method2 == 'CoreSetMIP':
        #     method2 = CoreSetMIPSampling
        # elif method2 == 'Discriminative':
        #     method2 = DiscriminativeSampling
        # elif method2 == 'DiscriminativeLearned':
        #     method2 = DiscriminativeRepresentationSampling
        # elif method2 == 'DiscriminativeAE':
        #     method2 = DiscriminativeAutoencoderSampling
        # elif method2 == 'DiscriminativeStochastic':
        #     method2 = DiscriminativeStochasticSampling
        elif method2_name == 'Uncertainty':
            method2 = UncertaintySampling
        # elif method2 == 'Bayesian':
        #     method2 = BayesianUncertaintySampling
        # elif method2 == 'UncertaintyEntropy':
        #     method2 = UncertaintyEntropySampling
        # elif method2 == 'BayesianEntropy':
        #     method2 = BayesianUncertaintyEntropySampling
        # elif method2 == 'EGL':
        #     method2 = EGLSampling
        # elif method2 == 'Adversarial':
        #     method2 = AdversarialSampling
        else:
            print("ERROR - UNKNOWN SECOND METHOD!")
            exit()
    else:
        method2 = None
        print("Only One Method Used...")

    # create the QueryMethod object:
    if method2 is not None:
        query_method = CombinedSampling(method, method2)
    else:
        query_method = method()
        
    return query_method
    

def evaluate_sample(config, training_function, X_train, iteration_idx):
    """
    A function that accepts a labeled-unlabeled data split and trains the relevant model on the labeled data, returning
    the model and it's accuracy on the test set.
    """

    # shuffle the training set:
    random.shuffle(X_train)

    # create the validation set:
    X_validation = X_train[:int(0.2*len(X_train))]
    X_train = X_train[int(0.2*len(X_train)):]

    # train and evaluate the model:
    training_function(config, [X_train], [X_validation], iteration_idx) # the best model in this iteration is saved in the model.pt file
    run_evaluate_dev(config, iteration_idx)

    
def active_train(config):

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py'])
    
    train_buckets = get_buckets(config.train_record_file)   # get_buckets returns [datapoints], and datapoints is a list, not numpy array
    random.shuffle(train_buckets)

    # warm-sart
    labeled_idx = get_initial_idx(train_buckets[0], config.initial_idx_path, config.initial_size, config.seed)
    query_method = set_query_method(config.method, config.method2)
    evaluate_sample(config, train, list(operator.itemgetter(*labeled_idx)(train_buckets[0])), -1) # will print evaluation result
    # query_method.update_model(model)
    
    # iteratively query
    for i in range(config.iterations):

        # get the new indices from the algorithm
        # old_labeled = np.copy(labeled_idx)
        labeled_idx = query_method.query(config, train_buckets[0], labeled_idx, config.label_batch_size)

        # # calculate and store the label entropy:
        # new_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, old_labeled))]
        # new_labels = Y_train[new_idx]
        # new_labels /= np.sum(new_labels)
        # new_labels = np.sum(new_labels, axis=0)
        # entropy = -np.sum(new_labels * np.log(new_labels + 1e-10))
        # entropies.append(entropy)
        # label_distributions.append(new_labels)
        # queries.append(new_idx)

        # evaluate the new sample:
        evaluate_sample(config, train, list(operator.itemgetter(*labeled_idx)(train_buckets[0])), i) 
        # query_method.update_model(model)
        metrics.append(metric)
    
    
