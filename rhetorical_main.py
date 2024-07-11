"""
Author: Longyin
Date: 2023.1
Email:
"""
import logging
from sys import argv
import argparse
from util.file_util import *
import numpy as np
import progressbar
import torch
import pandas as pd
import spacy
import os, sys
import time
import torch.nn as nn
import json
import re
import random
from rhetorical_model.trainer import Trainer
import spacy

torch.manual_seed(17)
np.random.seed(17)
random.seed(17)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--cversion", default=1, type=int)  # different clustering methods
    arg_parser.add_argument("--device", default="cuda:3")  # cpu, cuda:x
    arg_parser.add_argument("--path_dataset", default="data/rhetorical_explanation_upd.pkl")
    arg_parser.add_argument("--language_model", default="roberta-large")
    arg_parser.add_argument("--learning_rate", default=4e-4)
    arg_parser.add_argument("--epoch", default=100)
    arg_parser.add_argument("--valid_every", default=1)
    arg_parser.add_argument("--label_ids", default="data/rhetorical_label2id.pkl")
    arg_parser.add_argument("--label_space", default=3)
    arg_parser.add_argument("--BATCH_SIZE", default=128)
    arg_parser.add_argument("--BERT_MAX_LEN", default=512)
    arg_parser.add_argument("--cword_rep", default="xlnet")  # xlnet, transformer, sentence-transformer
    arg_parser.add_argument("--hidden_size", default=1024)
    arg_parser.set_defaults(use_gpu=True)
    args = arg_parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
    # fine-tune the language model or not
