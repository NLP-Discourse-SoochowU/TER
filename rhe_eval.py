# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2019.1.24
@Description: The trainer of our parser.
"""
import os
import logging
import random
import numpy as np
import progressbar
import torch
import torch.nn as nn
from util.DL_tricks import *
import torch.optim as optim
from rhetorical_model.model import RhetoricalModel
from transformers import *
from sentence_transformers import SentenceTransformer, util
from util.file_util import *

random.seed(17)
torch.manual_seed(17)
np.random.seed(17)
device = 'cuda'
label2id, id2label = load_data("data/rhetorical_label2id.pkl")
BATCH_SIZE = 128

def gen_batch_iter(dataset):
    instances = dataset
    random_instances = np.random.permutation(instances)
    num_instances = len(instances)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset + BATCH_SIZE)]
        batch_size = len(batch)
        left_all, right_all, label_all = list(), list(), np.zeros([batch_size], dtype=np.long)
        for batch_idx, (left_text, right_text, label) in enumerate(batch):
            left_all.append(left_text)
            right_all.append(right_text)
            label_all[batch_idx] = label2id[label]
        label_all = torch.from_numpy(label_all).long().to(device)
        yield left_all, right_all, label_all
        offset = offset + BATCH_SIZE


def evaluate(model, model_xl, tokenizer):
    """ evaluate the performance of the system
    """
    _, _, test_set = load_data("data/rhetorical_explanation_upd.pkl")

    batch_iter_eval = gen_batch_iter(test_set)
    all_num = 0.
    correct_num = 0.
    for n_batch, (left_all, right_all, label_all) in enumerate(batch_iter_eval, start=1):
        predictions = model(left_all, right_all, model_xl, tokenizer)
        label_all = label_all.cpu().detach().numpy()
        print(predictions)
        input(label_all)
        correct_ = np.sum(predictions == label_all)
        all_num += predictions.shape[0]
        correct_num += correct_
    return correct_num / all_num


if __name__ == "__main__":
    # rhetorical relation
    rhe_model, rhe_rep = torch.load("data/rhetorical_model/best.model", map_location='cuda')
    rhe_tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
    rhe_model.eval()
    rhe_rep.eval()
    evaluate(rhe_model, rhe_rep, rhe_tokenizer)

