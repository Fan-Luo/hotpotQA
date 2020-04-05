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
from comet_ml import Experiment, ExistingExperiment

def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(len(X_train))[np.logical_not(np.in1d(np.arange(len(X_train)), labeled_idx))]

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path+'/pred')

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

    def query(self, config, X_train, labeled_idx, amount, iter_id, experiment):
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
    def query(self, config, X_train, labeled_idx, amount, iter_id, experiment):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_train_datapoints = list(operator.itemgetter(*unlabeled_idx)(X_train))    
        predictions = run_predict_unlabel(config, [unlabeled_train_datapoints] )
        # print("predictions in query:")
        # print("len(predictions['softmax_ans_start']) in predict() ", len(predictions['softmax_ans_start']))
        # print("len(predictions['softmax_ans_end']) in predict() ", len(predictions['softmax_ans_end']))
        # print("len(predictions['softmax_type']) in predict() ", len(predictions['softmax_type']))
        # print("len(predictions['predict_support_li']) in predict() ", len(predictions['predict_support_li']))
        # print("len(predictions['qids']) in predict() ", len(predictions['qids']))
        
        # print("len(unlabeled_train_datapoints)", len(unlabeled_train_datapoints))
       
        # compare predictions[qids] with unlabeled_train_datapoints['id'] to check if the ordered is same 
        # it turns out the order is different, so the predictions has to be remapped back as the order of unlabeled_idx has 
        # print("predictions['qids'][:10]", predictions['qids'][:10])
        # for i in range(10):
        #     print("unlabeled_train_datapoints[", i, "]['id']", unlabeled_train_datapoints[i]['id'])
        
        ans_start_score = []
        ans_end_score = []
        type_score = []
        sp_score = []
        qids = []
        for i in range(len(unlabeled_train_datapoints)):
            #map back to the same order as unlabeled_train_datapoints according to qid
            prediction_idx = predictions['qids'].index(unlabeled_train_datapoints[i]['id'])
            
            ans_start_score.append(np.max(predictions['softmax_ans_start'][prediction_idx]))
            ans_end_score.append(np.max(predictions['softmax_ans_end'][prediction_idx]))
            type_score.append(np.max(predictions['softmax_type'][prediction_idx]))
            
            predict_support = predictions['predict_support_li'][prediction_idx]
            top2_sp_score = predict_support.take(np.argsort(predict_support)[-2:])
            sp_score.append(np.average(top2_sp_score))
            
            qids.append(predictions['qids'][prediction_idx])
        
        experiment.log_histogram_3d(ans_start_score, name="ans_start_score", step=iter_id)
        experiment.log_histogram_3d(ans_end_score, name="ans_end_score", step=iter_id)
        experiment.log_histogram_3d(type_score, name="type_score", step=iter_id)
        experiment.log_histogram_3d(sp_score, name="sp_score", step=iter_id)
        # print("ans_start_score ", ans_start_score)
        # print("ans_end_score ", ans_end_score)
        # print("type_score ", type_score)
        # print("sp_score ", sp_score)
        # print("len(qids) ", len(qids))
        # print("qids[:10]", qids[:10])
        
        unlabeled_predictions = (1.0 - config.sp_uncertainty_lambda) * (np.array(ans_start_score) + np.array(ans_end_score) + np.array(type_score)) + config.sp_uncertainty_lambda * np.array(sp_score) # logit1_score + logit2_score
        # print("unlabeled_predictions.shape ", unlabeled_predictions.shape)
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

    def query(self, config, X_train, labeled_idx, amount, iter_id, experiment):
        labeled_idx = self.method1.query(config, X_train, labeled_idx, int(amount/2), iter_id, experiment)
        return self.method2.query(config, X_train, labeled_idx, int(amount/2), iter_id, experiment)

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
    

def evaluate_sample(config, training_function, X_validation, X_train, experiment_key, iteration_idx, experiment_iter):
    """
    A function that accepts a labeled-unlabeled data split and trains the relevant model on the labeled data, save
    the model to file and e evaluate its perfromance on the test set (dev in this case).
    """

    # shuffle the training set:
    random.shuffle(X_train)

    # train, (validation) dev
    # create the validation set:
    # X_validation = X_train[:int(0.2*len(X_train))]   #
    # X_train = X_train[int(0.2*len(X_train)):]

    T_before_train = time.time() # before train
    # train and evaluate the model:
    training_function(config, [X_train], [X_validation], iteration_idx, experiment_iter) # the best model in this iteration is saved in the model.pt file
    T_after_train = time.time() # after train
    T = T_after_train - T_before_train
    print("train in iteration ", iteration_idx, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T)))
    
    T_before_evaluate_dev = time.time() # before evaluate_dev
    metrics = run_evaluate_dev(config, iteration_idx)
    
    T_after_evaluate_dev = time.time() # before evaluate_dev
    T = T_after_evaluate_dev - T_before_evaluate_dev
    print("evaluate on dev in iteration ", iteration_idx, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T)))
    
    experiment = ExistingExperiment(api_key="Q8LzfxMlAfA3ABWwq9fJDoR6r",previous_experiment=experiment_key)
    experiment_key = experiment.get_key() # have to update the key
    for k in metrics.keys():
        experiment.log_metric(k, metrics[k], step=iteration_idx)
    
    return experiment_key
    
def active_train(config):

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py'])
    
    experiment = Experiment(api_key="Q8LzfxMlAfA3ABWwq9fJDoR6r", project_name="hotpotqa-al", workspace="fan-luo")
    experiment.set_name(config.run_name)   
    experiment_key = experiment.get_key()
    
    train_buckets = get_buckets(config.train_record_file)   # get_buckets returns [datapoints], and datapoints is a list, not numpy array
    #print("number of datapoints in train_buckets", len(train_buckets[0]))  #89791
    random.shuffle(train_buckets)

    T_before_warm_sart = time.time() # before warm-sart
    # default inital labeled size: 2.5% of training set 89791 * 2.5 % = 2,245
    # warm-sart
    labeled_idx = get_initial_idx(train_buckets[0], config.initial_idx_path, config.initial_size, config.seed)
    print("initial labeled_idx.shape", labeled_idx.shape)
    if(np.unique(labeled_idx).shape != labeled_idx.shape):
        print("!!! initial labeled_idx has duplicate elements")
        exit()
    
    validation_idx = random.sample(labeled_idx, int(0.5*len(labeled_idx))) 
    X_validation = list(operator.itemgetter(*validation_idx)(train_buckets[0]))
    print("X_validation[:10]['id'] in warm-sart: ")
    for j in range(10):
        print(X_validation[j]['id'])
    train_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, validation_idx))]
    X_train =  list(operator.itemgetter(*train_idx)(train_buckets[0]))
    
    experiment_iteration = []
    experiment_iteration.append(Experiment(api_key="Q8LzfxMlAfA3ABWwq9fJDoR6r", project_name="hotpotqa-al", workspace="fan-luo"))
    experiment_iteration[0].set_name(config.run_name + "iteration0")  
    query_method = set_query_method(config.method, config.method2)
    experiment_key = evaluate_sample(config, train, X_validation, X_train, experiment_key, 0, experiment_iteration[0]) # will print evaluation result
    T_after_warm_sart = time.time() # after warm-sart
    T_warm_sart = T_after_warm_sart - T_before_warm_sart
    print("warm sart time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T_warm_sart)))
    
    # iteratively query
    for iter in range(config.iterations):
        iter_id = iter + 1
        T_before_iteration = time.time() # before current iteration

        # get the new indices from the algorithm
        # old_labeled = np.copy(labeled_idx)
        labeled_idx = query_method.query(config, train_buckets[0], labeled_idx, config.label_batch_size, iter_id, experiment)
        print("labeled_idx.shape", labeled_idx.shape)
        T_after_query = time.time()
        T_query = T_after_query - T_before_iteration
        print("query in iteration ", iter_id, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T_query)))
        if(np.unique(labeled_idx).shape != labeled_idx.shape):
            print("!!!labeled_idx has duplicate elements in iteration", iter_id)
            exit()
        # # calculate and store the label entropy:
        # new_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, old_labeled))]
        # new_labels = Y_train[new_idx]
        # new_labels /= np.sum(new_labels)
        # new_labels = np.sum(new_labels, axis=0)
        # entropy = -np.sum(new_labels * np.log(new_labels + 1e-10))
        # entropies.append(entropy)
        # label_distributions.append(new_labels)
        # queries.append(new_idx)
        train_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, validation_idx))]
        X_train = list(operator.itemgetter(*train_idx)(train_buckets[0]))
        print("X_validation[:10]['id'] in iteration ", iter_id, " : ")
        for j in range(10):
            print(X_validation[j]['id'])
            
        experiment_iteration.append(Experiment(api_key="Q8LzfxMlAfA3ABWwq9fJDoR6r", project_name="hotpotqa-al", workspace="fan-luo"))
        experiment_iteration[iter_id].set_name(config.run_name + "iteration" + str(iter_id))
        
        # evaluate the new sample:
        experiment_key = evaluate_sample(config, train, X_validation, X_train, experiment_key, iter_id, experiment_iteration[iter_id]) #X_validation is constant, always be the of half of the initial
        T_after_iteration = time.time() # before current iteration
        T_iteration = T_after_iteration - T_before_iteration
        print("iteration ", iter_id, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T_iteration)))
    
