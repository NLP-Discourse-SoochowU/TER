# coding: UTF-8
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import *

torch.manual_seed(17)
np.random.seed(17)
device = 'cuda'


class RhetoricalModel(nn.Module):
    def __init__(self, args):
        super(RhetoricalModel, self).__init__()
        self.args = args
        hidden_size, label_space = args.hidden_size, args.label_space
        self.scorer = nn.Linear(hidden_size * 2, label_space, bias=True)

    def encode_sent(self, text_batch, model_xl, tokenizer_xl):
        text_reps = None
        for text_piece in text_batch:
            inputs = tokenizer_xl(text_piece, max_length=self.args.BERT_MAX_LEN,
                                  truncation=True, return_tensors="pt").to(device)
            outputs = model_xl(**inputs)
            last_hidden_states = outputs.last_hidden_state
            output = torch.mean(last_hidden_states, dim=1)
            output = output.detach()
            text_reps = output if text_reps is None else torch.cat((text_reps, output), 0)
        return text_reps

    def encode_sent_transformer(self, text_batch, model_xl):
        corpus_embeddings = model_xl.encode(text_batch, batch_size=64, show_progress_bar=True, convert_to_tensor=True).to(device)
        return corpus_embeddings

    def encode_sent_xlnet(self, text_batch, model_xl, tokenizer_xl):
        text_reps = None
        for text_piece in text_batch:
            inputs = tokenizer_xl(text_piece, return_tensors="pt").to(device)
            outputs = model_xl(**inputs)
            last_hidden_states = outputs.last_hidden_state
            output = torch.mean(last_hidden_states, dim=1)
            output = output.detach()
            text_reps = output if text_reps is None else torch.cat((text_reps, output), 0)
        return text_reps

    def loss(self, left_all, right_all, label_all, model_xl, tokenizer):
        if self.args.cword_rep == "xlnet":
            left_rep = self.encode_sent_xlnet(left_all, model_xl, tokenizer)
            right_rep = self.encode_sent_xlnet(right_all, model_xl, tokenizer)
        elif self.args.cword_rep == "transformer":
            left_rep = self.encode_sent(left_all, model_xl, tokenizer)
            right_rep = self.encode_sent(right_all, model_xl, tokenizer)
        else:
            left_rep = self.encode_sent_transformer(left_all, model_xl)
            right_rep = self.encode_sent_transformer(right_all, model_xl)
        sons_rep = torch.cat((left_rep, right_rep), dim=-1)
        output = self.scorer(sons_rep)
        predictions = output.log_softmax(dim=-1)
        ground_labels = label_all.view(-1)
        loss = func.nll_loss(predictions, ground_labels)
        loss = loss / ground_labels.size(0)  # the average loss of each instance
        return loss

    def forward(self, left_all, right_all, model_xl, tokenizer):
        if self.args.cword_rep == "xlnet":
            left_rep = self.encode_sent_xlnet(left_all, model_xl, tokenizer)
            right_rep = self.encode_sent_xlnet(right_all, model_xl, tokenizer)
        elif self.args.cword_rep == "transformer":
            left_rep = self.encode_sent(left_all, model_xl, tokenizer)
            right_rep = self.encode_sent(right_all, model_xl, tokenizer)
        else:
            left_rep = self.encode_sent_transformer(left_all, model_xl)
            right_rep = self.encode_sent_transformer(right_all, model_xl)
        output = self.scorer(torch.cat((left_rep, right_rep), dim=-1))
        scores = output.softmax(dim=-1)
        pred_labels = torch.argmax(scores, dim=-1)
        pred_labels = pred_labels.cpu().detach().numpy()
        return pred_labels
