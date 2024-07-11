# Given the gold structure, estimate the relation between tree-node depth and bleurt-score. 
import json, re, torch, sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import progressbar
import os.path as osp
import argparse

blt_tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
blt_model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512").to('cuda')
blt_model.eval()


def load_gold():
    """ get the gold Task1 data lines, get the proofs, then for each tree, have a list with tuples 
        [(height, str_pred, str_gold)]
        Save all the nodes in the dataset. 
        "proof": "sent2 & sent3 -> int1: the northern hemisphere is a kind of place; int1 & sent1 -> hypothesis; "
    """
    pattern_sent = re.compile(r'sent\d+')
    path_ = "data/entailment_trees_emnlp2021_data_v2/dataset/task_1/test.jsonl"
    llm_data = list()
    with open(path_, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_dict = json.loads(line)
            # parse the proof
            context_txt = line_dict["context"]

            sentence_split = re.split('sent\d+: ', context_txt)[1:]
            node_dict = dict()
            node_height_dict = dict()
            for sent_id, sent in enumerate(sentence_split):
                node_dict["sent" + str(sent_id + 1)] = sent
                node_height_dict["sent" + str(sent_id + 1)] = 1

            # parse the proofs
            proofs = line_dict["proof"].split("; ")[:-1]
            hypothesis = line_dict["hypothesis"]
            for proof_item in proofs:
                facts_llm = [node_dict[item_one] for item_one in proof_item.split("->")[0].strip().split(" & ")]

                facts_heights = [node_height_dict[item_one] for item_one in proof_item.split("->")[0].strip().split(" & ")]
                current_height = max(facts_heights) + 1

                int_ = proof_item.split("->")[1]
                if int_.strip() != "hypothesis":
                    key_value = int_.split(": ")
                    target_gold = key_value[1].strip()
                    int_name = key_value[0].strip()
                    node_dict[int_name] = target_gold
                    node_height_dict[int_name] = current_height
                else:
                    target_gold = hypothesis
                llm_data.append((current_height, facts_llm, target_gold))
    return llm_data


def blt(pred_sent, gold_sent):
    with torch.no_grad():
        tokenized_data = blt_tokenizer([pred_sent], [gold_sent], return_tensors='pt', padding=True)
        tokenized_data = tokenized_data.to(blt_model.device)
        score = blt_model(**tokenized_data)[0].squeeze().item()
        return score



def accuracy_per_height(upd_list):
    """
        Print height to accuracy.
    """
    height_to_acc = dict()
    for item in upd_list:
        height_, pred_, gold_ = item
        bleurt_score = blt(pred_, gold_)
        if bleurt_score >= 0.28:
            if height_ in height_to_acc.keys():
                acc_info = height_to_acc[height_]
                height_to_acc[height_] = (acc_info[0] + 1, acc_info[1] + 1)
            else:
                height_to_acc[height_] = (1, 1)
        else:
            if height_ in height_to_acc.keys():
                acc_info = height_to_acc[height_]
                height_to_acc[height_] = (acc_info[0], acc_info[1] + 1)
            else:
                height_to_acc[height_] = (0, 1)
    print("Height to accuracy:")
    print(height_to_acc)


def eval_ftx():
    llm_data = load_gold()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("llm_flt/checkpoint-5154", use_cache=False)
    llm_model = llm_model.to("cuda")
    llm_model.eval()

    upd_list = list()

    for instance in progressbar.progressbar(llm_data):
        current_height, facts_llm, target_gold = instance

        with torch.no_grad():
            instruct = "In the deductive mode, given the following facts: " + "; ".join([f"Fact {idx + 1}. {facts_llm[idx].strip()}" for idx in range(len(facts_llm))]) + ". Please generate an inference."

            instruct_ids = llm_tokenizer(instruct, return_tensors="pt").input_ids.to("cuda")
                
            llm_output = llm_model.generate(instruct_ids, max_new_tokens=500, num_return_sequences=1, do_sample=True)
            llm_output = llm_tokenizer.batch_decode(llm_output)
            target_pred = [item.replace("<pad> ", "").replace("<pad>", "").replace("</s>", "") for item in llm_output][0]
        upd_list.append((current_height, target_pred, target_gold))
    accuracy_per_height(upd_list)


def eval_llama():
    llm_data = load_gold()

    from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
    from peft import PeftModel
    llama_base, llama_version = 'decapoda-research/llama-7b-hf', "llm_llama_upd"
    llm_tokenizer = LlamaTokenizer.from_pretrained(llama_base)
    model_ = LlamaForCausalLM.from_pretrained(llama_base, load_in_8bit=False, torch_dtype=torch.float16, device_map="auto")
    llm_model = PeftModel.from_pretrained(model_, llama_version, torch_dtype=torch.float16)
    llm_model.config.pad_token_id = llm_tokenizer.pad_token_id = 0  # unk
    llm_model.config.bos_token_id = 1
    llm_model.config.eos_token_id = 2
    llm_model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        llm_model = torch.compile(llm_model)

    upd_list = list()
    for instance in progressbar.progressbar(llm_data):
        current_height, facts_llm, target_gold = instance

        with torch.no_grad():
            instruct = "In the deductive mode, given the following facts: " + "; ".join([f"Fact {idx + 1}. {facts_llm[idx].strip()}" for idx in range(len(facts_llm))]) + ". Please generate an inference."

            instruct_ids = llm_tokenizer(instruct, return_tensors="pt").input_ids.to("cuda")
                    
            llm_output_ = llm_model.generate(input_ids=instruct_ids, max_new_tokens=50, num_beams=1, num_return_sequences=1, do_sample=True)
            target_pred = llm_tokenizer.batch_decode(llm_output_)[0]
            bad_1, bad_2, bad_3 = "The inference is that ", "Answer: The inference is that ", "Inference: "
            target_pred = target_pred.replace("<unk>", "").replace("\n", "").replace("</s>", "").replace(bad_1, "").replace(bad_2, "").replace(bad_3, "").split(". Please generate an inference.")[1]
        upd_list.append((current_height, target_pred, target_gold))
    accuracy_per_height(upd_list)


def eval_alpaca():
    llm_data = load_gold()

    from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
    from peft import PeftModel
    llama_base, llama_version = 'llm_alpaca_base', "llm_alpaca_upd"
    llm_tokenizer = LlamaTokenizer.from_pretrained(llama_base)
    model_ = LlamaForCausalLM.from_pretrained(llama_base, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
    llm_model = PeftModel.from_pretrained(model_, llama_version, torch_dtype=torch.float16)
    llm_model.config.pad_token_id = llm_tokenizer.pad_token_id = 0  # unk
    llm_model.config.bos_token_id = 1
    llm_model.config.eos_token_id = 2
    llm_model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        llm_model = torch.compile(llm_model)

    upd_list = list()
    for instance in progressbar.progressbar(llm_data):
        current_height, facts_llm, target_gold = instance

        with torch.no_grad():
            instruct = "In the deductive mode, given the following facts: " + "; ".join([f"Fact {idx + 1}. {facts_llm[idx].strip()}" for idx in range(len(facts_llm))]) + ". Please generate an inference."
            print(instruct)
            instruct_ids = llm_tokenizer(instruct, return_tensors="pt").input_ids.to("cuda")
                    
            llm_output_ = llm_model.generate(input_ids=instruct_ids, max_new_tokens=50, num_beams=1, num_return_sequences=1, do_sample=True)
            target_pred = llm_tokenizer.batch_decode(llm_output_)[0]
            bad_1, bad_2, bad_3 = "The inference is that ", "Answer: The inference is that ", "Inference: "
            target_pred = target_pred.replace("<unk>", "").replace("\n", "").replace("</s>", "").replace(bad_1, "").replace(bad_2, "").replace(bad_3, "").split(". Please generate an inference.")[1]
            input(target_pred)
        upd_list.append((current_height, target_pred, target_gold))
    accuracy_per_height(upd_list)


def load_reasoning_module(exp_dir):
    from transformers import T5ForConditionalGeneration,T5Tokenizer
    # read config
    config = json.load(open(osp.join(exp_dir,'config.json')))
    model_config = json.load(open(osp.join(exp_dir,'model.config.json')))
    args = argparse.Namespace(**config)

    # load model
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-large")
    except:
        model = T5ForConditionalGeneration.from_pretrained("t5-large", local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    tokenizer.sep_token = tokenizer.eos_token

    model.config.update(model_config)

    # load trained parameters
    state_dict = torch.load(osp.join(exp_dir,'best_model.pth'), map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model, tokenizer, args


def module_generate(input_sents, model, tokenizer, args, num_return=1):
    model.eval()
    with torch.no_grad():
        input_sents = [' '.join(sents) for sents in input_sents]
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


def eval_t5():
    llm_data = load_gold()

    module_all, tokenizer_module, args_module = load_reasoning_module("exp/Module_all/para_etree_all/Acdpaxg6")

    upd_list = list()
    for instance in progressbar.progressbar(llm_data):
        current_height, facts_llm, target_gold = instance
        with torch.no_grad():
            target_pred = module_generate([facts_llm], module_all, tokenizer_module, args_module)[0]
        upd_list.append((current_height, target_pred, target_gold))
    accuracy_per_height(upd_list)


if __name__ == '__main__':
    # eval_t5()
    # eval_ftx()
    eval_llama()
    # eval_alpaca()