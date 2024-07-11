from builtins import NotImplementedError
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore TF log
import sys
import random
import copy
import os.path as osp
import glog as log
import json
import math
import time
import argparse
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import precision_score
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.append("..")
from util.file_util import load_data
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from train_Controller import Controller, LinearsHead
import bleurt
from bleurt import score
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from tree_utils import *
from sent_utils import add_fullstop, remove_fullstop
from evaluate_metric import *
from sent_utils import LCstring, sent_overlap
import progressbar


##### experiment utils
def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname


def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def uncapitalize(sent):
    if len(sent) > 1:
        return sent[0].lower() + sent[1:]
    else:
        return sent.lower()


class RetDataset(Dataset):
    def __init__(self, data_set, args):
        item_list = list()
        for item in data_set:
            path_states = item["goals"]
            path_labels = item["sorted_paths"][1]
            upd_path_states = list()
            for path_idx, states in enumerate(path_states):
                states_info = list()
                for state in states:
                    states_info.append((state[0], state[1], state[2]))
                #     input(state[0])
                #     input(state[1])
                #     input(state[2])
                #     input("==========state one")
                # print("========== path one", path_labels[path_idx])
                # input()
                upd_path_states.append(states_info)
            assert len(upd_path_states) == len(path_labels)
            item_list.append((upd_path_states, path_labels))
        self.dataset = item_list

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


class RetrieveM(nn.Module):
    def __init__(self, args, config):
        super(RetrieveM, self).__init__()
        self.args = args
        self.config = config
        self.controller_model_ = Controller(args, config)
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        self.retrieve_fact_scorer = LinearsHead(config, input_num=3)  # state_seq, hypo, current_fact
        self.path_scorer = LinearsHead(config, input_num=1)  # state_seq, hypo, current_fact

        self.device = self.encoder.device
        # encoder_layer = nn.TransformerEncoderLayer(d_model=4096, nhead=4)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # src = torch.rand(10, 32, 512)
        # transformer_encoder(src)

        if self.args.state_method == 'fact_cls_learn':
            lambda_weight = torch.FloatTensor([0.5, 0.5])
            lambda_weight = nn.Parameter(lambda_weight, requires_grad=True)
            self.lambda_weight = lambda_weight
            assert self.lambda_weight.requires_grad == True
        self.margin_ranking_loss = nn.MarginRankingLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()
        self.pad_fact_rep = None

    def train_(self, batch_data, max_state_facts, max_state_pairs):
        """ Borrow the parameters of the controller as our retriever, tuning the parameter during retrieving.
            Data: A batch (B) of paths with max_len L1 for each path, each path contains a group of states and each
                  state has a max length L2.
                  For each path, we input the states info (L1, L2) into the state scorer of Controller for representation
                  h = (L1, H1). Then we input the data into a pre-trained XXX to learn the CoT information inside the
                  path, obtaining (B, L1, H2).

            For retrieving:
                  1. We pair the hidden states (B, L1, H2) with the representation of (1 Hypothesis & L1 groups of
                     candidates) obtaining good cases (B, L1, H3) and bad cases (B, L1, H3). MarginRankingLoss
                  2. We pair the hidden states (B, L1, H2) with the representation of (B, L1, C) candidates, obtaining
                     (B, N, L1, H2) -> softmax->(B, L1, C) vs (B, L1) best premises. NLL_Loss

            For chain selection:
                  (Good chains lead to better CoT, faster and more accurate)
                  1. Based on the state representations (B, L1, H2), we will get a batch of path info (B, H4). On the
                     contrary, we have a batch representation of bad paths (B, H4). MarginRankingLoss
        """
        batch_input, batch_goal_label = batch_data
        # print("===============================")
        # print("Path number: ", len(batch_input))
        # print("States per path: ", max_state_facts)
        # print("Fact pairs per path: ", max_state_pairs)
        state_rep = self.state_encoding(batch_input)  # h (B, L, H1), sent (B, L, H1)

        retrieving_good_h, retrieving_bad_h = None, None  # 1. MarginRankingLoss for retrieving
        retrieving_facts_all, facts_mask, retrieving_facts_best = None, None, list()  # 2. NLL_Loss for retrieving

        path_info_true = list()
        path_info_false = list()
        for path_id, path_state_rep in enumerate(state_rep):
            # path_state_rep means the data for a parsing path
            fact_features, target_features, state_cls = path_state_rep
            batch_labels, path_label = batch_goal_label[path_id]
            # states
            state_seq = None
            state_seq_h = None
            for state_idx in range(len(state_cls)):
                facts_all, target_all, fact_pairs_, fact_best_ = batch_labels[state_idx]
                state_fact_features = fact_features[state_idx]
                state_target_features = target_features[state_idx][0]  # only one target
                # state_seq = state_cls[:state_idx + 1]
                one_state = state_cls[state_idx].unsqueeze(0)
                state_seq = one_state if state_seq is None else torch.cat((state_seq, one_state), 0)
                # state_seq_h = self.transformer_encoder(state_seq)  # select the last one, or avg them with attention
                state_seq_h = torch.mean(state_seq, 0)
                true_ones, false_ones = fact_pairs_
                for true_id, false_id in zip(true_ones, false_ones):
                    # print(facts_all, true_id, false_id)
                    true_idx = facts_all.index(true_id)
                    false_idx = facts_all.index(false_id)
                    # training goal A
                    good_fact_h = torch.cat((state_target_features, state_seq_h, state_fact_features[true_idx]), 0).unsqueeze(0)  # H1
                    bad_fact_h = torch.cat((state_target_features, state_seq_h, state_fact_features[false_idx]), 0).unsqueeze(0)
                    retrieving_good_h = good_fact_h if retrieving_good_h is None else torch.cat((retrieving_good_h, good_fact_h), 0)
                    retrieving_bad_h = bad_fact_h if retrieving_bad_h is None else torch.cat((retrieving_bad_h, bad_fact_h), 0)

                # facts_rep
                state_facts_rep, facts_mask_one = None, list()
                fact_id = 0
                for fact_rep in state_fact_features:
                    fact_rep = torch.cat((state_target_features, state_seq_h, fact_rep), -1)
                    fact_rep = fact_rep.unsqueeze(0)
                    state_facts_rep = fact_rep if state_facts_rep is None else torch.cat((state_facts_rep, fact_rep), 0)
                    facts_mask_one.append(1)
                    fact_id += 1

                while fact_id < max_state_facts:
                    if self.pad_fact_rep is None:
                        self.pad_fact_rep = nn.Parameter(torch.empty(1, fact_rep.size(1), dtype=torch.float)).to(device)
                        nn.init.xavier_normal_(self.pad_fact_rep)
                    # padding
                    state_facts_rep = self.pad_fact_rep if state_facts_rep is None else torch.cat((state_facts_rep, self.pad_fact_rep), 0)
                    facts_mask_one.append(0)
                    fact_id += 1

                if target_all is not None:
                    # training goal B
                    if fact_best_ is not None and len(fact_best_) > 0:
                        # print(fact_best_)
                        state_facts_rep = state_facts_rep.unsqueeze(0)
                        retrieving_facts_all = state_facts_rep if retrieving_facts_all is None else torch.cat((retrieving_facts_all, state_facts_rep), 0)
                        facts_mask_one = torch.tensor(facts_mask_one).unsqueeze(0).to(device)
                        facts_mask = facts_mask_one if facts_mask is None else torch.cat((facts_mask, facts_mask_one), 0)
                        best_item = random.sample(fact_best_, 1)[0]  # 我们只考虑学习 最优者，多轮学习，最终会涉及所有
                        retrieving_facts_best.append(facts_all.index(best_item))
            
            # training goal C
            if path_label:
                path_info_true.append(state_seq_h)
            else:
                path_info_false.append(state_seq_h)
        path_pairs = list()
        true_path_h, false_path_h = None, None
        # print("****** HIGHLIGHT ******")
        # print(len(path_info_true), len(path_info_false))
        for true_path_ in path_info_true:
            true_path_ = true_path_.unsqueeze(0)
            for false_path_ in path_info_false:
                false_path_ = false_path_.unsqueeze(0)
                true_path_h = true_path_ if true_path_h is None else torch.cat((true_path_h, true_path_), 0)
                false_path_h = false_path_ if false_path_h is None else torch.cat((false_path_h, false_path_), 0)

        ret_mrl_loss = ret_nll_loss= path_loss = 0.

        # score & loss: good and bad retrieving cases
        if retrieving_good_h is not None:
            # print("Training goal A.")
            # print(retrieving_good_h.size(), retrieving_bad_h.size())
            good_h_scores = self.retrieve_fact_scorer(retrieving_good_h).view(-1).sigmoid() # logit -> 0~1
            bad_h_scores = self.retrieve_fact_scorer(retrieving_bad_h).view(-1).sigmoid() # logit -> 0~1
            # print(good_h_scores.size(), bad_h_scores.size())
            good_h_scores = good_h_scores.view(-1)
            bad_h_scores = bad_h_scores.view(-1)
            assert good_h_scores.size() == bad_h_scores.size()
            target = torch.ones_like(good_h_scores)
            ret_mrl_loss = self.margin_ranking_loss(good_h_scores, bad_h_scores, target) # retrieving + MarginRankingLoss
            # print(ret_mrl_loss)

        # score & loss: best retrieving cases
        if retrieving_facts_all is not None:
            # print("Training goal B.")
            h_scores = self.retrieve_fact_scorer(retrieving_facts_all).squeeze(-1)
            # h_scores = h_scores.squeeze(-1).view(-1, retrieving_facts_all.size(-1)) # (B * L, K), B paths, L states, K facts
            h_scores = self.log_softmax(h_scores)  # (B * L, K)
            # print(retrieving_facts_all.size(), h_scores.size(), facts_mask.size())
            retrieving_facts_best = torch.tensor(retrieving_facts_best, dtype=torch.long).view(-1).to(device)
            # print(h_scores.size(), retrieving_facts_best.size())
            # mask for h_scores
            h_scores = h_scores * facts_mask + -1e8 * (1 - facts_mask)
            ret_nll_loss = self.nll_loss(h_scores, retrieving_facts_best)  # retrieving + NLL_Loss
            # print(ret_nll_loss)

        # score & loss: best retrieving path
        if true_path_h is not None:
            # print("Training goal C.")
            # print(true_path_h.size(), false_path_h.size())
            good_p_scores = self.path_scorer(true_path_h).view(-1).sigmoid() # logit -> 0~1
            bad_p_scores = self.path_scorer(false_path_h).view(-1).sigmoid() # logit -> 0~1
            # print(good_p_scores.size(), bad_p_scores.size())
            good_p_scores = good_p_scores.view(-1)
            bad_p_scores = bad_p_scores.view(-1)
            assert good_p_scores.size() == bad_p_scores.size()
            target = torch.ones_like(good_p_scores)
            path_loss = self.margin_ranking_loss(good_p_scores, bad_p_scores, target) # path selection + MarginRankingLoss
            # print(path_loss)
        return ret_mrl_loss, ret_nll_loss, path_loss

    def state_encoding(self, batch_input):
        """ [
        [('sent1 & sent2 & sent3 -> hypothesis', (['sent1', 'sent2'], ['sent3', 'sent3']), None, 'ded: sent1 & sent3 -> int1'), ('int1 & sent2 -> hypothesis', (['sent2'], ['sent3']), ['sent2'], 'ded: int1 & sent2 -> hypothesis')], [('sent1 & sent2 & sent3 -> hypothesis', (['sent1', 'sent2'], ['sent3', 'sent3']), None, 'abd: sent2 & hypothesis -> int1'), ('sent1 & sent3 -> int1', ([], []), ['sent1', 'sent3'], 'ded: sent1 & sent3 -> int1')]
        ]
        """
        path_state_info = list()
        for path_data in batch_input:
            batch_encoder_inputs, offsets_all = path_data
            fact_offsets_all, target_offsets_all = offsets_all
            # path (L1, L2)
            outputs = self.controller_model_.encoder(**batch_encoder_inputs)
            batch_size = outputs['last_hidden_state'].size(0)

            target_features, fact_features, state_rep = list(), list(), list()
            for batch_index in range(batch_size):
                seq_features = outputs['last_hidden_state'][batch_index]
                state_rep.append(seq_features[0])

                target_offsets_one = target_offsets_all[batch_index]
                targets_rep = list() 
                for target_offset in target_offsets_one:
                    targets_rep.append(torch.mean(seq_features[target_offset[0]:target_offset[1], :], dim=0))
                target_features.append(targets_rep)

                fact_offsets_one = fact_offsets_all[batch_index]
                fact_rep = list()
                for fact_offset in fact_offsets_one:
                    fact_rep.append(torch.mean(seq_features[fact_offset[0]:fact_offset[1], :], dim=0))
                fact_features.append(fact_rep)

            path_state_info.append((fact_features, target_features, state_rep))
        return path_state_info


    def forward(self, batch_encoder_inputs, batch_label_info):
        raise NotImplementedError


def create_optimizer(model, args):
    # decay if not LayerNorm or bias
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps": 1e-6,
        }
    optimizer_kwargs["lr"] = args.lr

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer


def create_scheduler(optimizer, args):
    warmup_steps = math.ceil(args.num_training_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    return lr_scheduler


def tokenize_target_fact_sents(target_sents, fact_sents, tokenizer, pad_max_length=None):
    target_offsets = []
    fact_offsets = []
    input_ids = []
    attention_mask = []
    # make input ids
    input_ids.append(tokenizer.cls_token_id)

    for s in target_sents:
        new_token = tokenizer.encode(s, add_special_tokens=False)
        new_offsets = [len(input_ids), len(input_ids) + len(new_token)]
        target_offsets.append(new_offsets)
        input_ids += new_token
        input_ids += [tokenizer.sep_token_id]
    input_ids += [tokenizer.sep_token_id]

    for s in fact_sents:
        new_token = tokenizer.encode(s, add_special_tokens=False)
        new_offsets = [len(input_ids), len(input_ids) + len(new_token)]
        fact_offsets.append(new_offsets)
        input_ids += new_token
        input_ids += [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    if pad_max_length:
        attention_mask += [0] * (pad_max_length - len(input_ids))
        input_ids += [tokenizer.pad_token_id] * (pad_max_length - len(input_ids))
    return input_ids, attention_mask, target_offsets, fact_offsets


def token_encoding(target_sents, fact_sents, tokenizer, pad_max_length=None):
    target_offsets = []
    fact_offsets = []
    input_ids = []
    attention_mask = []
    # make input ids
    input_ids.append(tokenizer.cls_token_id)

    for s in target_sents:
        new_token = tokenizer.encode(s, add_special_tokens=False)
        new_offsets = [len(input_ids), len(input_ids) + len(new_token)]
        target_offsets.append(new_offsets)
        input_ids += new_token
        input_ids += [tokenizer.sep_token_id]
    input_ids += [tokenizer.sep_token_id]

    for s in fact_sents:
        new_token = tokenizer.encode(s, add_special_tokens=False)
        new_offsets = [len(input_ids), len(input_ids) + len(new_token)]
        fact_offsets.append(new_offsets)
        input_ids += new_token
        input_ids += [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    if pad_max_length:
        attention_mask += [0] * (pad_max_length - len(input_ids))
        input_ids += [tokenizer.pad_token_id] * (pad_max_length - len(input_ids))
    return input_ids, attention_mask, target_offsets, fact_offsets


def get_batch_data(batch, tokenizer, args, device):
    """
    [[('sent1 & sent2 & sent3 -> hypothesis', (['sent1', 'sent2'], ['sent3', 'sent3']), None, 0 / 1),
      ('int1 & sent2 -> hypothesis', (['sent2'], ['sent3']), ['sent2'], 'ded: int1 & sent2 -> hypothesis')],
     [('sent1 & sent2 & sent3 -> hypothesis', (['sent1', 'sent2'], ['sent3', 'sent3']), None, 'abd: sent2 & hypothesis -> int1'),
      ('sent1 & sent3 -> int1', ([], []), ['sent1', 'sent3'], 'ded: sent1 & sent3 -> int1')]]
    """
    batch_input_g, batch_goal_label_g = list(), list()
    batch_input_b, batch_goal_label_b = list(), list()
    path_data_all, path_label_all = batch[0], batch[1]

    path_num_g, path_num_b = list(), list()
    for idx, path_label in enumerate(path_label_all):
        if path_label:
            path_num_g.append(idx)
        else:
            path_num_b.append(idx)
    path_idx_g = random.sample(path_num_g, min(len(path_num_g), 4))
    path_idx_b = random.sample(path_num_b, min(len(path_num_b), 4))

    data_idx = 0
    for path_data, path_label in zip(path_data_all, path_label_all):
        if data_idx not in path_idx_g and data_idx not in path_idx_b:
            continue
        # path (L1, L2)
        l1 = len(path_data)
        batch_encoder_inputs = {'input_ids': [], 'attention_mask': []}
        batch_goals = []
        fact_offsets_all, target_offsets_all = list(), list()
        for state_idx, state in enumerate(path_data):
            ref_info = state[0]
            fact_target = ref_info.split(" -> ")
            facts = fact_target[0].split(" & ")
            state_ids, state_masks, target_offsets, fact_offsets = token_encoding(fact_target[1], facts, tokenizer,
                                                                                  pad_max_length=args.max_token_length)
            fact_offsets_all.append(fact_offsets)
            target_offsets_all.append(target_offsets)
            batch_encoder_inputs['input_ids'].append(state_ids)
            batch_encoder_inputs['attention_mask'].append(state_masks)

            # goals
            facts_all = state[0].split(" -> ")[0].split(" & ")
            target_all = state[0].split(" -> ")[1]
            fact_pairs_ = state[1]
            fact_best_ = state[2]
            batch_goals.append((facts_all, target_all, fact_pairs_, fact_best_))
        batch_encoder_inputs['input_ids'] = torch.LongTensor(batch_encoder_inputs['input_ids']).to(device)
        batch_encoder_inputs['attention_mask'] = torch.LongTensor(batch_encoder_inputs['attention_mask']).to(device)
        if path_label:
            batch_input_g.append((batch_encoder_inputs, (fact_offsets_all, target_offsets_all)))
            batch_goal_label_g.append((batch_goals, path_label))
            # print("AAAAA")
            # print(batch_input_g)
            # input(batch_goal_label_g)
        else:
            batch_input_b.append((batch_encoder_inputs, (fact_offsets_all, target_offsets_all)))
            batch_goal_label_b.append((batch_goals, path_label))
            # print("BBBBB")
            # print(batch_input_b)
            # input(batch_goal_label_b)
        data_idx += 1
    # print(len(batch_input_g), len(batch_input_b))
    # input("check aaa")
    sampled_batch_input = batch_input_g + batch_input_b
    sampled_batch_goal_label = batch_goal_label_g + batch_goal_label_b

    max_state_facts, max_state_pairs = get_mask_info(sampled_batch_goal_label)
    return (sampled_batch_input, sampled_batch_goal_label), max_state_facts, max_state_pairs


def eval_model_by_loss(data_loader, model, tokenizer, args):
    model.eval()
    metric_collecter = defaultdict(list)
    batch_num = len(data_loader)
    ret_mrl_loss_, ret_nll_loss_, path_loss_ = 0., 0., 0.
    metric_collecter = {"ret_mrl_loss": list(), "ret_nll_loss": list(), "path_loss": list()}
    with torch.no_grad():
        p = progressbar.ProgressBar()
        p_idx = 0
        p.start(len(data_loader))
        for batch in data_loader:
            p_idx += 1
            p.update(p_idx)
            batch_data, max_state_facts, max_state_pairs = get_batch_data(batch, tokenizer, args, device)
            ret_mrl_loss, ret_nll_loss, path_loss = model.train_(batch_data, max_state_facts, max_state_pairs)

            ret_mrl_loss_show = ret_mrl_loss if isinstance(ret_mrl_loss, float) else ret_mrl_loss.detach().clone().cpu().data
            metric_collecter["ret_mrl_loss"].append(ret_mrl_loss_show)
            ret_nll_loss_show = ret_nll_loss if isinstance(ret_nll_loss, float) else ret_nll_loss.detach().clone().cpu().data
            metric_collecter["ret_nll_loss"].append(ret_nll_loss_show)
            path_loss_show = path_loss if isinstance(path_loss, float) else path_loss.detach().clone().cpu().data
            metric_collecter["path_loss"].append(path_loss_show)
        p.finish()
    for k, v in metric_collecter.items():
        metric_collecter[k] = np.mean(v)
    return dict(metric_collecter)


def get_mask_info(batch_goal_label):
    """ We only build mask for 1-batch data with K paths, get max inside the K paths.
    """
    max_state_pairs, max_state_facts = 0, 0
    for one_path_labels in batch_goal_label:
        path_labels, _ = one_path_labels
        for state_labels in path_labels:
            facts_all, target_all, fact_pairs_, fact_best_ = state_labels
            max_state_facts = max(max_state_facts, len(facts_all))
            max_state_pairs = max(max_state_pairs, len(fact_pairs_[0]))
    return max_state_facts, max_state_pairs


def run(args, retrieve_model):
    """ Load a pre-trained controller for retrieving tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retrieve_model = retrieve_model.to(device)

    log.info("Loading data...")
    train_, dev_, _ = load_data(args.retrieve_data_file)
    train_dataset = RetDataset(train_, args)
    dev_dataset = RetDataset(dev_, args)

    def collect_compare(batch):
        new_batch = []
        for group in batch:
            new_batch += group
        return new_batch

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collect_compare)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collect_compare)

    log.info(f"Length of training dataest: {len(train_dataset)}")
    log.info(f"Length of dev dataest: {len(dev_dataset)}")
    log.info(f"number of iteration each epoch : {len(train_loader)}")
    args.eval_iter = round(args.eval_epoch * len(train_loader))
    args.report_iter = round(args.report_epoch * len(train_loader))
    args.num_training_steps = args.epochs * len(train_loader)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    with open(osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
        json.dump(vars(retrieve_model.config), f, sort_keys=False, indent=4)

    optimizer = create_optimizer(retrieve_model, args)
    lr_scheduler = create_scheduler(optimizer, args)

    log.info("start training")
    global_iter = 0
    epoch_init = 1
    loss_collecter = defaultdict(list)
    best_metric = -100
    for epoch_i in range(epoch_init, args.epochs + 1):
        p = progressbar.ProgressBar()
        p_idx = 0
        p.start(len(train_loader))
        for batch in train_loader:
            batch_data, max_state_facts, max_state_pairs = get_batch_data(batch, tokenizer, args, device)
            p_idx += 1
            p.update(p_idx)
            try:
                ret_mrl_loss, ret_nll_loss, path_loss = retrieve_model.train_(batch_data, max_state_facts, max_state_pairs)
                loss = 0.5 * ret_mrl_loss + 0.5 * ret_nll_loss + 0.0 * path_loss
                if not isinstance(loss, float):
                    loss.backward()
                # add gradient clip
                nn.utils.clip_grad_norm_(retrieve_model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            except RuntimeError as exception:
                ret_mrl_loss = ret_nll_loss = path_loss = 0.
                if "out of memory" in str(exception):
                    print('solving the oom problem, jump the item')
                if hasattr(torch.cuda, 'empty_cache'):
                    print('solved!')
                    torch.cuda.empty_cache()
                else:
                    raise exception
            
            ret_mrl_loss_show = ret_mrl_loss if isinstance(ret_mrl_loss, float) else ret_mrl_loss.detach().clone().cpu().data
            ret_nll_loss_show = ret_nll_loss if isinstance(ret_nll_loss, float) else ret_nll_loss.detach().clone().cpu().data
            path_loss_show = path_loss if isinstance(path_loss, float) else path_loss.detach().clone().cpu().data

            global_iter += 1

            if not global_iter % args.eval_iter:  # 50 times larger than report_iter
                log.info(f"----- state evaluate -----")
                metric_collecter = eval_model_by_loss(dev_loader, retrieve_model, tokenizer, args)

                metric_str = ""
                for k, v in metric_collecter.items():
                    metric_str += f" {k} {v:.4f}"
                log.info(f"Iteration {global_iter} dev {metric_str}")

                if args.save_model:
                    # save_path = osp.join(args.exp_dir, f'model_{global_iter}.pth')
                    # torch.save(retrieve_model.state_dict(), save_path)
                    last_path = osp.join(args.exp_dir, f'model_last.pth')
                    torch.save(retrieve_model.state_dict(), last_path)
                    log.info(f"Iteration {global_iter} save model")
            if not global_iter % args.report_iter:
                show_str = ""
                for k, v in loss_collecter.items():
                    show_str += f" {k} {np.mean(v):.4f} "
                log.info(f"Epoch {global_iter / len(train_loader):.2f} training loss {show_str}")
                loss_collecter = defaultdict(list)
            else:
                loss_collecter["ret_mrl_loss"].append(ret_mrl_loss_show)
                loss_collecter["ret_nll_loss"].append(ret_nll_loss_show)
                loss_collecter["path_loss"].append(path_loss_show)
        log.info(f"Epoch {epoch_i} finished")
        p.finish()

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Training BART')

    # dateset
    parser.add_argument("--retrieve_data_file", type=str,
                        default='', help="training data file")
    parser.add_argument("--pre_m", type=str,
                        default='', help="pre-trained controller")
    parser.add_argument("--pre_m_t", type=str,
                        default='', help="pre-trained controller-t")
    parser.add_argument("--train_data_file", type=str, 
                        default='', help="training data file")
    parser.add_argument("--dev_data_file", type=str, 
                        default='', help="dev data file")
    parser.add_argument("--test_data_file", type=str, 
                        default='', help="test data file")  
    parser.add_argument("--data_loading_type", type=str,
                        default='orig', help="test data file")
    parser.add_argument("--compare_group_len", type=int,
                        default=1, help="")
    parser.add_argument("--max_pre_training", type=int,
                        default=100, help="")
    parser.add_argument("--abd_compare", action='store_true', default=False)
    parser.add_argument('--compare_strategy', type=int,
                        nargs='+', default=[3], help='compare_strategy')
    parser.add_argument("--sample_strategy", type=int, default=1, help="")
    parser.add_argument('--ratio_train', type=float, default=1.0)

    # model
    parser.add_argument("--model_name_or_path", type=str,
                        default="facebook/bart-large", help="")
    parser.add_argument("--resume_path", type=str,
                        default="", help="")
    parser.add_argument("--task_name", type=str,
                        default="", help="")
    parser.add_argument("--num_qa", type=int,
                        default=0, help="")
    parser.add_argument("--num_abd", type=int,
                        default=0, help="")
    parser.add_argument("--state_method", type=str,
                        default='fact_cls_learn', help="")
    parser.add_argument("--num_max_fact", type=int,
                        default=20, help="")
    parser.add_argument("--max_token_length", type=int,
                        default=450, help="")
    parser.add_argument("--step_func", type=str,
                        default='softmax_all', help="")

    # optimization
    parser.add_argument('--bs', type=int, default=5, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')

    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adafactor', action='store_true', default=False)
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--eval_epoch', type=float, default=1.0)

    parser.add_argument('--loss_weight', type=float,
                        nargs='+', default=[1.0, 1.0, 1.0, 1.0, 0.01], help='state_fact_step_stepabd')
    parser.add_argument('--add_state_cls_loss', type=float, default=0.0)
    parser.add_argument('--add_fact_cls_loss', type=float, default=1.0)
    parser.add_argument('--add_step_cls_loss', type=float, default=1.0)

    parser.add_argument('--margins', type=float, nargs='+', default=[0.1, 0.1, 0.1],
                        help='state_fact_step_loss_margin')

    # seed
    parser.add_argument('--seed', type=int, default=1260, metavar='S',
                        help='random seed')

    # exp and log
    parser.add_argument("--exp_dir", type=str, default='./exp')
    parser.add_argument("--code_dir", type=str, default='./code')
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--report_epoch', type=float, default=1.0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_params()
    if args.seed == 0:
        args.seed = random.randint(1, 1e4)
    os.makedirs(args.exp_dir, exist_ok=True)

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # make metrics.json for logging metrics
    args.metric_file = osp.join(args.exp_dir, 'metrics.json')
    open(args.metric_file, 'a').close()

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    os.system(f'cp -r {args.code_dir} {args.exp_dir}')

    log.info('Python info: {}'.format(os.popen('which python').read().strip()))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    # load pre-trained model
    load_flag = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log.info("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # controller_model_ = Controller(args, config)  # We should keep the arguments the same for the shared module
    retrieve_model_ = RetrieveM(args, config)
    
    if load_flag == 0: 
        # load parameters from the saved one directly
        model_dict = retrieve_model_.state_dict()
        state_dict = torch.load(args.pre_m_t, map_location='cpu')
        model_dict.update(state_dict)
        retrieve_model_.load_state_dict(model_dict)
    elif load_flag == 1:
        # load parameters from the saved one, and update the tuned controller's parameters
        model_dict = retrieve_model_.state_dict()
        state_dict = torch.load(args.pre_m_t, map_location='cpu')
        model_dict.update(state_dict)
        retrieve_model_.load_state_dict(model_dict)

        model_dict_ = retrieve_model_.controller_model_.state_dict()
        state_dict_ = torch.load(args.pre_m, map_location='cpu')
        model_dict_.update(state_dict_)
        retrieve_model_.controller_model_.load_state_dict(model_dict_)
    elif load_flag == 2:
        # initial stage, load the controller model param only
        model_dict = retrieve_model_.controller_model_.state_dict()
        state_dict = torch.load(args.pre_m, map_location='cpu')
        model_dict.update(state_dict)
        retrieve_model_.controller_model_.load_state_dict(model_dict)
    else:
        pass
    run(args, retrieve_model_)
    open(osp.join(args.exp_dir, 'done'), 'a').close()
