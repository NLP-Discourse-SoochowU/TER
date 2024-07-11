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

class Trainer:
    def __init__(self, args):
        self.args = args
        train_data, dev_data, test_data = load_data(args.path_dataset)
        self.train_set = train_data
        self.dev_set = dev_data
        self.test_set = test_data
        label2id, id2label = load_data(args.label_ids)
        self.label2id = label2id
        self.id2label = id2label
        self.model = RhetoricalModel(args).to(device)
        if args.cword_rep == "xlnet":
            self.tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
            self.model_xl = XLNetModel.from_pretrained("xlnet-large-cased").to(device)
        elif args.cword_rep == "transformer":
            self.model_xl = RobertaModel.from_pretrained(args.language_model).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        else:
            self.model_xl = SentenceTransformer('all-MiniLM-L6-v2').to(device)
            self.tokenizer = None

    def train(self):
        n_iter = 0
        optimizer = optim.Adam([{'params': self.model.parameters()},
                                {'params': self.model_xl.parameters(), 'lr': 1e-5}], lr=self.args.learning_rate,
                               weight_decay=1e-3)
        optimizer.zero_grad()
        p = progressbar.ProgressBar()

        eval_loss_log = 1e8
        for epoch in range(1, self.args.epoch + 1):
            self.model.train()
            self.model_xl.train()
            p.start(len(self.train_set) // self.args.BATCH_SIZE)
            batch_iter = self.gen_batch_iter()
            for n_batch, (left_all, right_all, label_all) in enumerate(batch_iter, start=1):
                # p.update(n_batch)
                n_iter += 1
                loss = self.model.loss(left_all, right_all, label_all, self.model_xl, self.tokenizer)
                print_("Training Loss: " + str(loss.item()), "train_log.txt")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if epoch % self.args.valid_every == 0:
                new_loss = self.estimate()
                print_("Epoch: " + str(epoch) + " -- Temporary-Loss: " + str(new_loss)
                       + " -- Min-Loss: " + str(eval_loss_log), "train_log.txt")
                if new_loss < eval_loss_log:
                    eval_loss_log = new_loss
                    torch.save((self.model, self.model_xl), "data/rhetorical_model/best.model")
                    acc_score = self.evaluate()
                    print_("Accuracy for the current dev model: " + str(acc_score) + " Epoch: " + str(epoch),
                           "train_log.txt")
            p.finish()

    def gen_batch_iter(self, dataset=None):
        instances = self.train_set if dataset is None else dataset
        random_instances = np.random.permutation(instances)
        num_instances = len(instances)
        offset = 0
        while offset < num_instances:
            batch = random_instances[offset: min(num_instances, offset + self.args.BATCH_SIZE)]
            batch_size = len(batch)
            left_all, right_all, label_all = list(), list(), np.zeros([batch_size], dtype=np.long)
            for batch_idx, (left_text, right_text, label) in enumerate(batch):
                left_all.append(left_text)
                right_all.append(right_text)
                label_all[batch_idx] = self.label2id[label]
            label_all = torch.from_numpy(label_all).long().to(device)
            yield left_all, right_all, label_all
            offset = offset + self.args.BATCH_SIZE

    def estimate(self):
        """ evaluate the performance of the model.
        """
        self.model.eval()
        self.model_xl.eval()
        batch_iter_eval = self.gen_batch_iter(self.dev_set)
        loss_all = 0.
        ite_count = 0
        for n_batch, (left_all, right_all, label_all) in enumerate(batch_iter_eval, start=1):
            loss = self.model.loss(left_all, right_all, label_all, self.model_xl, self.tokenizer)
            loss_all += loss.item()
            ite_count += 1
        return loss_all / ite_count

    def evaluate(self):
        """ evaluate the performance of the system
        """
        batch_iter_eval = self.gen_batch_iter(self.test_set)
        all_num = 0.
        correct_num = 0.
        for n_batch, (left_all, right_all, label_all) in enumerate(batch_iter_eval, start=1):
            predictions = self.model(left_all, right_all, self.model_xl, self.tokenizer)
            label_all = label_all.cpu().detach().numpy()
            correct_ = np.sum(predictions == label_all)
            all_num += predictions.shape[0]
            correct_num += correct_
        return correct_num / all_num
