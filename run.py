from comet_ml import Experiment
import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model import Model #, NoCharModel, NoSelfModel
from sp_model import SPModel
# from normal_model import NormalModel, NoSelfModel, NoCharModel, NoSentModel
# from oracle_model import OracleModel, OracleModelV2
# from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from util import convert_tokens, evaluate
from util import get_buckets, DataIterator, IGNORE_INDEX
import time
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F
from hotpot_evaluate_v1 import eval as eval_all_metrics

nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduce=False, ignore_index=IGNORE_INDEX)

def train(config, train_buckets, validation_buckets, iteration_idx, experiment_iteration):

    # T_before_loading = time.time() # before 
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.validation_eval_file, "r") as fh:   # validation is 0.2 of train,  its eval file is actually 'train_eval.json'. It uses an id field to recognize question 
        validation_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    # config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    # create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py'])
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Train a model...")
    # train_buckets = get_buckets(config.train_record_file, config.example_portion)
    # dev_buckets = get_buckets(config.dev_record_file)

    def build_train_iterator():
        # print("para_size as parameter in build_train_iterator:" + str(config.para_limit))
        # print("ques_size as parameter in build_train_iterator:" + str(config.ques_limit))
        return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit)

    def build_validation_iterator():
        return DataIterator(validation_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    ori_model = model.cuda()
    model = nn.DataParallel(ori_model)
    print("next(model.parameters()).is_cuda: " + str(next(model.parameters()).is_cuda));
    print("next(ori_model.parameters()).is_cuda: " + str(next(ori_model.parameters()).is_cuda));

    lr = config.init_lr
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    cur_patience = 0
    total_loss = 0
    total_ans_loss = 0
    total_sp_loss = 0
    global_step = 0
    best_validation_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    
    # T_after_loading = time.time() # after  
    # T = T_after_loading - T_before_loading
    # print("loading data in train() in iteration ", iteration_idx, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T))) # about 2 minutes
    
    T_before_train = time.time() # before  
    model.train()
    for epoch in range(10000):
        T_before_epoch = time.time() # before 
        for data in build_train_iterator():
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            context_lens = Variable(data['context_lens'])
            y1 = Variable(data['y1'])
            y2 = Variable(data['y2'])
            q_type = Variable(data['q_type'])
            is_support = Variable(data['is_support'])
            start_mapping = Variable(data['start_mapping'])
            end_mapping = Variable(data['end_mapping'])
            all_mapping = Variable(data['all_mapping'])

            # T_before_forwardPass = time.time() # before 
            logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)
            # T_after_forwardPass = time.time() # after
            # T_forwardPass = T_after_forwardPass - T_before_forwardPass
            # print("Forward Pass in epoch ", epoch, " in iteration ", iteration_idx, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T_forwardPass)))  #takes 0s
            
            loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
            loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
            loss = loss_1 + config.sp_lambda * loss_2

            total_ans_loss += loss_1.data[0]
            total_sp_loss += loss_2.data[0]
            total_loss += loss.data[0]
            
            optimizer.zero_grad()
            T_before_backwardPass = time.time() # before 
            loss.backward()
            T_after_backwardPass = time.time() # after 
            T_backwardPass = T_after_backwardPass - T_before_backwardPass
            print("Backward Pass in epoch ", epoch, " in iteration ", iteration_idx, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T_backwardPass)))
            optimizer.step()
            
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                cur_ans_loss = total_ans_loss / config.period
                cur_sp_loss = total_sp_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f} | answer loss {:8.3f} | supporting facts loss {:8.3f} '.format(epoch, global_step, lr, elapsed*1000/config.period, cur_loss, cur_ans_loss, cur_sp_loss))
                experiment_iteration.log_metrics({'train loss':cur_loss, 'train answer loss':cur_ans_loss ,'train supporting facts loss':cur_sp_loss }, step=global_step)
                total_loss = 0
                total_ans_loss = 0
                total_sp_loss = 0
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                metrics = evaluate_batch(build_validation_iterator(), model, 0, validation_eval_file, config)
                model.train()

                logging('-' * 89)
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | validation loss {:8.3f} | answer loss {:8.3f} | supporting facts loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, metrics['loss'], metrics['ans_loss'], metrics['sp_loss'], metrics['exact_match'], metrics['f1']))
                logging('-' * 89)
                experiment_iteration.log_metrics({'validation loss':metrics['loss'], 'validation answer loss':metrics['ans_loss'] ,'validation supporting facts loss':metrics['sp_loss'], 'EM':metrics['exact_match'], 'F1': metrics['f1']}, step=global_step)

                eval_start_time = time.time()

                validation_F1 = metrics['f1']
                if best_validation_F1 is None or validation_F1 > best_validation_F1:
                    best_validation_F1 = validation_F1
                    torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0
   
        if stop_train: break
        
        T_after_epoch = time.time() # after  
        T_epoch = T_after_epoch - T_before_epoch
        print("epoch ", epoch, " in train() in iteration ", iteration_idx, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T_epoch))) 
        # for uncertainty sampling, each epoch takes 1 min in iteration0, 8min in iteration1, 15 min in teration2, 22min in teration3, 30 min in teration4,...1h12min in teration10
    
    
    T_after_train = time.time() # after  
    T_train = T_after_train - T_before_train
    print("train() in iteration ", iteration_idx, " takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T_train)))
    
    logging('best_validation_F1 {}'.format(best_validation_F1))

def evaluate_batch(data_source, model, max_batches, eval_file, config):
    answer_dict = {}
    sp_dict = {}
    total_loss, total_ans_loss, total_sp_loss, step_cnt = 0, 0, 0, 0
    iter = data_source
    for step, data in enumerate(iter):
        if step >= max_batches and max_batches > 0: break

        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        y1 = Variable(data['y1'], volatile=True)
        y2 = Variable(data['y2'], volatile=True)
        q_type = Variable(data['q_type'], volatile=True)
        is_support = Variable(data['is_support'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)

        logit1, logit2, predict_type, predict_support, yp1, yp2, _, _ = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) 
        loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
        loss = loss_1 + config.sp_lambda * loss_2
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        total_loss += loss.data[0]
        total_ans_loss += loss_1.data[0]
        total_sp_loss += loss_2.data[0]
        step_cnt += 1
    loss = total_loss / step_cnt
    ans_loss = total_ans_loss / step_cnt
    sp_loss = total_sp_loss / step_cnt
    metrics = evaluate(eval_file, answer_dict)
    metrics['loss'] = loss
    metrics['ans_loss'] = ans_loss
    metrics['sp_loss'] = sp_loss

    return metrics

def predict(data_source, model, eval_file, config, prediction_file):
    answer_dict = {}
    sp_dict = {}
    sp_th = config.sp_threshold
    m = nn.Softmax(dim=-1)
    softmax_ans_start = []
    softmax_ans_end = []
    softmax_type = []
    predict_support_li = []
    qids = []
    ques_embeds = np.array([])
    for step, data in enumerate(tqdm(data_source)):
        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)

        logit1, logit2, predict_type, predict_support, yp1, yp2, ans_start, ans_end, ques_embed = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        
        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        if prediction_file == 'Uncertainty':
            # softmax_logit1 = m(logit1).data.cpu().numpy()
            # softmax_logit2 = m(logit2).data.cpu().numpy()
            softmax_ans_start.extend(list(m(ans_start).data.cpu().numpy())) # list of numpy array
            softmax_ans_end.extend(list(m(ans_end).data.cpu().numpy()))
            softmax_type.extend(list(m(predict_type).data.cpu().numpy()))
            predict_support_li.extend(list(predict_support_np))
            qids.extend(data['ids'])
            # print("len(softmax_ans_start) in predict() ", len(softmax_ans_start))  # 不同batch中的softmax_ans_start的长度不一样,因为para_limit不一样
            # print("len(softmax_ans_start) in predict() ", len(softmax_ans_start))
            # print("len(softmax_type) in predict() ", len(softmax_type))
            # print("len(predict_support_li) in predict() ", len(predict_support_li))
            # print("len(qids) in predict() ", len(qids))
        elif prediction_file == 'CoreSet':
            qids.extend(data['ids'])
            if ques_embeds.size == 0:
                ques_embeds = ques_embed.data.cpu().numpy()
            else:
                ques_embeds = np.vstack((ques_embeds, ques_embed.data.cpu().numpy()))
        else:
            answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
            answer_dict.update(answer_dict_)
    
            # predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
            for i in range(predict_support_np.shape[0]):
                cur_sp_pred = []
                cur_id = data['ids'][i]
                for j in range(predict_support_np.shape[1]):
                    if j >= len(eval_file[cur_id]['sent2title_ids']): break
                    if predict_support_np[i, j] > sp_th:
                        cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
                sp_dict.update({cur_id: cur_sp_pred})
    
    if prediction_file == 'Uncertainty':
        predictions = dict.fromkeys(['softmax_ans_start', 'softmax_ans_end', 'softmax_type', 'predict_support_li', 'qids'],[]) #'softmax_logit1', 'softmax_logit2'
        predictions['softmax_ans_start'] = softmax_ans_start   
        predictions['softmax_ans_end'] = softmax_ans_end
        predictions['softmax_type'] = softmax_type
        predictions['predict_support_li'] = predict_support_li
        predictions['qids'] = qids
        print("len(predictions['softmax_ans_start']) in predict() ", len(predictions['softmax_ans_start']))
        print("len(predictions['softmax_ans_end']) in predict() ", len(predictions['softmax_ans_end']))
        print("len(predictions['softmax_type']) in predict() ", len(predictions['softmax_type']))
        print("len(predictions['predict_support_li']) in predict() ", len(predictions['predict_support_li']))
        print("len(predictions['qids']) in predict() ", len(predictions['qids']))
        print("predictions['predict_support_li'][0].shape in predict() ", predictions['predict_support_li'][0].shape)
        print("predictions['softmax_ans_start'][0].shape in predict() ", predictions['softmax_ans_start'][0].shape)
        print("predictions['softmax_type'][0].shape in predict() ", predictions['softmax_type'][0].shape)
        return predictions
    elif prediction_file == 'CoreSet':
        predictions = dict({'qids': [], 'question_embedding': np.array([])}) 
        predictions['qids'] = qids
        predictions['question_embedding'] = ques_embeds
        print("predictions['question_embedding'].shape in predict() ", predictions['question_embedding'].shape)
        return predictions
    else:
        prediction = {'answer': answer_dict, 'sp': sp_dict}
        # print("prediction_file", prediction_file, "in predict() before save")
        with open(prediction_file, 'w') as f:
            json.dump(prediction, f)
        print("saved prediction in prediction_file", prediction_file, "in predict()")
        
def load_model_data(config, data_split):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    if data_split == 'train':
        with open(config.train_eval_file, "r") as fh:
            eval_file = json.load(fh)
        para_limit = config.para_limit
        ques_limit = config.ques_limit
    elif data_split == 'dev':
        with open(config.dev_eval_file, "r") as fh:
            eval_file = json.load(fh)
        buckets = get_buckets(config.dev_record_file)
        para_limit = config.para_limit
        ques_limit = config.ques_limit
    else: #test
        with open(config.test_eval_file, 'r') as fh:
            eval_file = json.load(fh)
        buckets = get_buckets(config.test_record_file)
        para_limit = None
        ques_limit = None
        
    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)
    ori_model = model.cuda()
    # print("ori_model.linear_type.weight before load_state_dict", ori_model.linear_type.weight)
    ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
    # print("ori_model.linear_type.weight after load_state_dict", ori_model.linear_type.weight)
    model = nn.DataParallel(ori_model)
    # print("model.module.linear_type.weight after load_state_dict", model.module.linear_type.weight)
    if data_split == 'train':
        return model, eval_file, para_limit, ques_limit
    else:      
        return model, buckets, eval_file, para_limit, ques_limit

def test(config, data_split, iteration_idx):
    model, buckets, eval_file, para_limit, ques_limit = load_model_data(config, data_split)
    # print("model.module.linear_type.weight in test() after load_model_data()", model.module.linear_type.weight)
    model.eval()
    def build_iterator():
        return DataIterator(buckets, config.batch_size, para_limit, ques_limit, config.char_limit, False, config.sent_limit)
    print("config.save before prediction_file: ", config.save)
    if config.mode == 'train':
        prediction_file = config.save + '/pred/' + config.run_name + '_iter' + str(iteration_idx) + '.json'
    else:
        prediction_file = config.save + '/pred/' + config.run_name + '.json'
    print("prediction_file", prediction_file, "in test() at iteration: ", iteration_idx)
    predict(build_iterator(), model, eval_file, config, prediction_file)

    if config.mode == 'train':
        return prediction_file

def run_predict_unlabel(config, buckets, predict_type):
    model, eval_file, para_limit, ques_limit = load_model_data(config, config.data_split)
    # print("model.module.linear_type.weight in run_predict_unlabel() after load_model_data()", model.module.linear_type.weight)
    model.eval()
    print("unlabeled datapoints in buckets in run_predict_unlabel()",len(buckets[0]))
    def build_iterator():
        return DataIterator(buckets, config.batch_size, para_limit, ques_limit, config.char_limit, False, config.sent_limit)
    predictions = predict(build_iterator(), model, eval_file, config, predict_type)
    return predictions
    
def run_evaluate_dev(config, iteration_idx):
    prediction_file = test(config, 'dev', iteration_idx)
    print("prediction_file", prediction_file, "in run_evaluate_dev() at iteration: ", iteration_idx)
    metrics = eval_all_metrics(prediction_file, config.dev_gold)
    return metrics
         