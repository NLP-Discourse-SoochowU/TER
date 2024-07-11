import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # ignore TF log
import os.path as osp
import json
import argparse
import sys
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BartForConditionalGeneration,BartTokenizer
from transformers import T5ForConditionalGeneration,T5Tokenizer


from transformers.optimization import Adafactor,AdamW,get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F

import bleurt 
from bleurt import score
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

from tree_utils import *
from evaluate_metric import *
from sent_utils import add_fullstop,sent_overlap

# ================================================================
# SPECIAL
# ================================================================
# Alpaca_lora
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
llama_base = 'decapoda-research/llama-7b-hf'  # llm_alpaca_base means alpaca-7B,  decapoda-research/llama-7b-hf means llama-7B
llama_version = "../llm_llama_upd"  # llm_alpaca_upd means alpaca-7B,  llm_llama_upd means llama-7B
llm_tokenizer = LlamaTokenizer.from_pretrained(llama_base)
model_ = LlamaForCausalLM.from_pretrained(llama_base, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
llm_model = PeftModel.from_pretrained(model_, llama_version, torch_dtype=torch.float16)
llm_model.config.pad_token_id = llm_tokenizer.pad_token_id = 0  # unk
llm_model.config.bos_token_id = 1
llm_model.config.eos_token_id = 2
llm_model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    llm_model = torch.compile(llm_model)

# ----- reasoning_module ----- 
def load_reasoning_module(exp_dir):
    # read config
    config = json.load(open(osp.join(exp_dir,'config.json')))
    model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
    args = argparse.Namespace(**config)

    # load model
    if args.model_name_or_path in ['facebook/bart-large']:
        try:
            model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        except:
            model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path in ['t5-large','t5-base','t5-small']:
        try:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        except:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.sep_token = tokenizer.eos_token
    else:
        raise NotImplementedError

    model.config.update(model_config)

    # load trained parameters
    state_dict = torch.load(osp.join(exp_dir,'best_model.pth'), map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model,tokenizer,args


def module_generate(input_sents, model, tokenizer, args, num_return=1):
    model.eval()
    with torch.no_grad():
        if args.input_join:
            input_sents = [' '.join(sents) for sents in input_sents]
        # print("Checking ............... ")
        # input(input_sents)
        input_batch = tokenizer(
                input_sents,
                add_special_tokens=True,
                return_tensors='pt',
                padding='longest',
                max_length=args.max_src_length,
                truncation=True,
            )
        input_batch = input_batch.to(model.device)
        # generate
        generated = model.generate(
            input_ids = input_batch['input_ids'],
            attention_mask = input_batch['attention_mask'],
            top_p = 0.9,
            do_sample = True,
            max_length= 50, 
            num_return_sequences = num_return, 
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return decoded


def inference_alltype_batch(input_sents, reasoning_module_tuple, bs = 1, num_return=1):
    reasoning_modules, tokenizer_module, args_module = reasoning_module_tuple
    # input_sents [num_sent]
    # output_sents [num_sent, num_outputs]
    # cls_types [num_sent, num_outputs]
    # use corresponding module to infer result
    output_sents = [[] for _ in range(len(input_sents))]
    cls_types = [[] for _ in range(len(input_sents))]
    
    for module_type, infer_module in reasoning_modules.items():
        infer_model = infer_module["model"]
        tmp_sents = input_sents
        tmp_outputs = []
        for batch in chunk(tmp_sents,bs):
            generated_sents = module_generate(batch, model=infer_model, tokenizer=tokenizer_module, args=args_module,
                                              num_return=num_return)
            # generated_sents len = num_return*bs
            for i in range(0,len(generated_sents),num_return):
                tmp_outputs.append(list(set(generated_sents[i:i+num_return])))
        for index, outs in enumerate(tmp_outputs):
            for out in outs:
                output_sents[index].append(out)
                cls_types[index].append(module_type)
    return output_sents, cls_types


def inference_alltype_batch_with_buffer(input_sents,reasoning_module_tuple, num_return=1, buffer_dict=None):
    # input_sents [num_sent]
    # output_sents [num_sent, num_outputs]
    # cls_types [num_sent, num_outputs]

    reasoning_modules,tokenizer_module,args_module = reasoning_module_tuple

    if buffer_dict is None:
        buffer_dict = {}

    for module_type in reasoning_modules.keys():
        if module_type not in buffer_dict:
            buffer_dict[module_type] = {}

    output_sents = [[] for _ in range(len(input_sents))]
    cls_types = [[] for _ in range(len(input_sents))]
    
    # use corresponding module to infer result
    for module_type, infer_module_info in reasoning_modules.items():
        infer_module = infer_module_info['model']
        task_prefix = infer_module_info['task_prefix']
        
        if len(task_prefix) == 0:
            tmp_sents = input_sents
        else:
            tmp_sents = [[task_prefix] + input_ for input_ in input_sents]

        # generate num_return sents for this module_type for each input
        tmp_outputs = [[] for _ in range(len(input_sents))]
        for index, input_ in enumerate(tmp_sents):
            prediction_type = input_[0].split()[0]

            buffer_key = "+".join(input_)

            if buffer_key in buffer_dict[module_type].keys():
                tmp_outputs[index] = buffer_dict[module_type][buffer_key]
            else:
                # get output
                facts = input_[1:]
                if prediction_type == "deductive":
                    instruct = "In the deductive mode, given the following facts: " + "; ".join([f"Fact {idx + 1}. {facts[idx].strip()}" for idx in range(len(facts))]) + ". Please generate an inference."
                else:
                    tmp_facts = facts[1:] if len(facts) > 1 else []
                    instruct = f"In the abductive mode, given the hypothesis: '{facts[0]}' and the following facts: " + "; ".join([f"Fact {idx + 1}. {facts[idx].strip()}" for idx in range(len(tmp_facts))]) + ". Please generate another fact to support this hypothesis."
                
                with torch.no_grad():
                    instruct_ids = llm_tokenizer(instruct, return_tensors="pt").input_ids.to("cuda")
                    
                    llm_output_ = llm_model.generate(input_ids=instruct_ids, max_new_tokens=50, num_beams=1, num_return_sequences=1, do_sample=True)
                    llm_output_ = llm_tokenizer.batch_decode(llm_output_)
                    bad_1, bad_2, bad_3 = "The inference is that ", "Answer: The inference is that ", "Inference: "
                    llm_output_ = [item.replace("<unk>", "").replace("\n", "").replace("</s>", "").replace(bad_1, "").replace(bad_2, "").replace(bad_3, "") for item in llm_output_]
                    llm_output = list()
                    for item in llm_output_:
                        if prediction_type == "deductive":
                            llm_output.append(item.split(". Please generate an inference.")[1])
                        else:
                            llm_output.append(item.split(". Please generate another fact to support this hypothesis.")[1])
                    tmp_outputs[index] = llm_output

                    buffer_dict[module_type][buffer_key] = tmp_outputs[index]

        for index, outs in enumerate(tmp_outputs):
            for out in outs:
                output_sents[index].append(out)
                cls_types[index].append(module_type)
    
    return output_sents, cls_types, buffer_dict
