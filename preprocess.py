"""
Author: Longyin
Date: 2023-2024
Email: zhangly@i2r.a-star.edu.sg

The binary tree is saved in entailment_trees_binary
"""
from util.file_util import *
import json
import re
from tqdm import tqdm
import nltk
from random import sample
from nltk.corpus import stopwords
from string import punctuation
import inflect

p = inflect.engine()

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

pattern_sent = re.compile(r'sent\d+')
pattern_int = re.compile(r'int\d+')

global_control = 0
global_paths_ = list()

global llm_data
llm_data = list()


def get_state_ref(state, other_sentences):
    left_elements, right_elements = [], []
    for state_item in state:
        left_right = state_item.split("->")
        left_elements += pattern_sent.findall(left_right[0])
        left_elements += pattern_int.findall(left_right[0])
        right_elements += pattern_int.findall(left_right[1])
        if "hypothesis" in left_right[1]:
            right_elements.append("hypothesis")
    # leaves should not exist in the right part
    # hypothesis should not exist in the left part
    left_set = set(left_elements)
    right_set = set(right_elements)
    leaves = list(left_set - right_set)
    hyp = list(right_set - left_set)
    if len(hyp) != 1:
        print(leaves)
        print(hyp)
        input(state)
    leaves = sorted(leaves + other_sentences)
    state = " & ".join(leaves) + " -> " + hyp[0]
    return state


def path_parse(other_sentences, sentences_all, node_dict, explanation, state, next_state, last_chosen=None, cns=[], good_cases=None):
    """ parse the path to get all gold nodes, parse similar ones as good, others as bad;
        parse the path to get a best node for training.
    """
    state_ref = get_state_ref(state, other_sentences)  # 参考的内容如此，我们从参考的内容中 retrieve ? 还是从所有 句子 中 retrieve
    current_leaves = state_ref.split(" -> ")[0].split(" & ")

    if last_chosen is not None:
        last_cns = pattern_int.findall(last_chosen[0])
        cns_all_str = " ".join(cns)
        all_cns = pattern_int.findall(cns_all_str)
        last_cns = [n for n in last_cns if n not in all_cns]
        cns += last_chosen
        if len(last_cns) > 1:
            input("error_A")
        else:
            last_cns = last_cns[0]
        explanation = explanation + " & " + node_dict[last_cns]

    pos_out = nltk.pos_tag(nltk.word_tokenize(explanation))
    exp_words = [p.singular_noun(word[0]) or word[0] for word in pos_out
                 if (word[1] in ['NN', 'NNS', 'NNP', 'NNPS'] and word[0] not in punctuation
                     and word[0] not in stop_words)]
    gold_premises = list()
    sentences = pattern_sent.findall(" ".join(state))
    for sent in sentences:
        sent_text = node_dict[sent]
        # print(sent, sent_text)
        tokens = nltk.word_tokenize(sent_text)
        pos_out = nltk.pos_tag(tokens)
        sent_words = [p.singular_noun(word[0]) or word[0] for word in pos_out
                      if (word[1] in ['NN', 'NNS', 'NNP', 'NNPS'] and word[0] not in punctuation
                          and word[0] not in stop_words)]
        if len(set(sent_words).intersection(set(exp_words))) > 0:
            gold_premises.append(sent)
    if good_cases is None:
        good_cases = set(gold_premises[:])
    else:
        good_cases = good_cases | set(gold_premises[:])  # keep updating
    bad_cases = list()
    for sent in current_leaves:
        if sent not in good_cases and "int" not in sent:
            bad_cases.append(sent)
    bad_num, gold_num = len(bad_cases), len(gold_premises)
    if bad_num > gold_num:
        bad_premises = sample(bad_cases, gold_num)
    elif bad_num != 0:
        times, left = gold_num // bad_num, gold_num % bad_num
        bad_premises = bad_cases * times + sample(bad_cases, left)
    else:
        gold_premises = bad_premises = []

    assert len(gold_premises) == len(bad_premises)
    margin_pairs = (gold_premises, bad_premises)

    # find the best item
    chosen_node = state if next_state is None else [item for item in state if item not in next_state]
    if last_chosen is None:
        best_item = None
    else:
        # pattern decide
        child_p = last_chosen[0].split("->")
        best_item = list()
        if last_cns in child_p[1]:
            # pattern 1, retrieve siblings
            for proof in state:
                if last_cns in proof.split("->")[0]:
                    best_item += pattern_sent.findall(proof.split("->")[0])
        elif last_cns in child_p[0]:
            # pattern 2, retrieve children
            for proof in state:
                if last_cns in proof.split("->")[1]:
                    best_item += pattern_sent.findall(proof.split("->")[0])
        else:
            input("error_B")
        # Best items for max margin learning
    return margin_pairs, best_item, chosen_node, state_ref, good_cases


def retrieved_leaves(sentences_all, node_dict, hypothesis, proofs, gold_paths):
    """ Given a state, get the gold data, the target.
        Init state, given the hypothesis, take the premises with word intersection as the gold to retrieve.
        Pattern_1: ...
        Pattern_2: ...
    """
    sentences = pattern_sent.findall(" ".join(proofs))
    other_sentences = [sent for sent in sentences_all if sent not in sentences]
    goals = list()
    for path_one in gold_paths:
        goal = list()
        init_state = path_one[0]
        cns = list()
        next_state = path_one[1] if len(path_one) > 1 else None
        margin_pairs, best_leaf, cn, state_ref, good_cases = \
            path_parse(other_sentences, sentences_all, node_dict, hypothesis, init_state, next_state, None, cns)
        # step build, best steps and all possible steps
        # print(cn)
        int_new_ = pattern_int.findall(cn[0])
        int_new = [n for n in int_new_ if n not in " ".join(cns)]
        int_new = int_new[0] if len(int_new) > 0 else ("hypothesis" if "hypothesis" in cn[0] else int_new_[0])
        # print(int_new)
        step_ = cn[0].split(": ")[0].split(" -> ")
        # print(step)
        step = step_[0].split(" & ") + [step_[1]]
        step.remove(int_new)
        step = " & ".join(step) + " -> " + int_new
        step = "abd: " + step if int_new in step_[0] else "ded: " + step
        # print(step)
        goal.append((state_ref, margin_pairs, best_leaf, step))
        # pattern = re.compile(r'int\d+')
        # cns = pattern.findall(cn[0])
        path_one = path_one[1:]
        for idx, state in enumerate(path_one):
            if idx + 1 < len(path_one):
                next_state = path_one[idx + 1]
            else:
                next_state = None
            margin_pairs, best_leaf, cn, state_ref, good_cases = path_parse(other_sentences, sentences_all, node_dict,
                                                                            hypothesis, state, next_state, cn, cns, good_cases)
            # step build, best steps and all possible steps
            int_new_ = pattern_int.findall(cn[0])
            int_new = [n for n in int_new_ if n not in " ".join(cns)]
            int_new = int_new[0] if len(int_new) > 0 else ("hypothesis" if "hypothesis" in cn[0] else int_new_[0])
            # print(int_new)
            step_ = cn[0].split(": ")[0].split(" -> ")
            step = step_[0].split(" & ") + [step_[1]]
            step.remove(int_new)
            step = " & ".join(step) + " -> " + int_new
            step = "abd: " + step if int_new in step_[0] else "ded: " + step
            # print(step)
            goal.append((state_ref, margin_pairs, best_leaf, step))
        goals.append(goal)
    return goals


def build_training_data(task_name):
    train_path = "data/entailment_trees_binary/" + task_name + "/train.jsonl"
    test_path = "data/entailment_trees_binary/" + task_name + "/test.jsonl"
    dev_path = "data/entailment_trees_binary/" + task_name + "/dev.jsonl"
    train_set = build_one(train_path)
    test_set = build_one(test_path)
    dev_set = build_one(dev_path)
    save_data((train_set, dev_set, test_set), "data/entailment_trees_binary/" + task_name + "/control_state_seq.pkl")


def path_judge(paths, goals):
    """ Good paths and bad paths
        good paths should cover more and more good elements.
        The paths that cannot cover all the elements are bad ones?
        The paths that cover less elements the worse?
    """
    good_path_scores = dict()
    bad_path_scores = dict()
    path_label = list()
    for path_one, goal_one in zip(paths, goals):
        cover_leaves = set()
        gold_leaves = set(pattern_sent.findall(" ".join(path_one[0])))
        score = 0
        good_path_flag = 0  # for the case all the leaves are gold ones
        for state, goal in zip(path_one, goal_one):
            state_ref, margin_pairs, best_leaf, step = goal
            good_path_flag += (len(margin_pairs[0]) + len(margin_pairs[1]))
            cover_leaves = cover_leaves | set(margin_pairs[0])
            score += len(margin_pairs[0])
        if (len(gold_leaves) - len(cover_leaves)) == 0 or good_path_flag == 0:
            # cover all good or all of the leaves are good
            if score in good_path_scores.keys():
                good_path_scores[score].append(path_one)
            else:
                good_path_scores[score] = [path_one]
            path_label.append(True)
        else:
            path_label.append(False)
            if score in bad_path_scores.keys():
                bad_path_scores[score].append(path_one)
            else:
                bad_path_scores[score] = [path_one]
    # sorted
    good_keys = sorted(good_path_scores.keys(), reverse=True)
    bad_keys = sorted(bad_path_scores.keys(), reverse=True)
    good_ones, bad_ones = list(), list()
    for key in good_keys:
        good_ones += good_path_scores[key]
    for key in bad_keys:
        bad_ones += bad_path_scores[key]
    sorted_paths = good_ones + bad_ones
    return sorted_paths, path_label


def build_one(original_text):
    global global_control
    global global_paths_, llm_data
    tree_info = list()
    state_path_all = list()
    with open(original_text, "r") as f:
        json_lines = f.readlines()
        for line in tqdm(json_lines):
            tree_dict = json.loads(line)
            context = tree_dict["context"]
            sentence_split = re.split('sent\d+: ', context)[1:]
            node_dict = dict()
            sentences_all = list()
            for sent_id, sent in enumerate(sentence_split):
                node_dict["sent" + str(sent_id + 1)] = sent
                sentences_all.append("sent" + str(sent_id + 1))
            hypothesis = tree_dict["hypothesis"]
            proofs = tree_dict["proof"].split("; ")[:-1]
            candidate_leaves = list(node_dict.keys())

            # considering the int nodes
            for proof_item in proofs:
                facts_llm = [node_dict[item_one] for item_one in proof_item.split("->")[0].strip().split(" & ")]
                int_ = proof_item.split("->")[1]
                if int_.strip() != "hypothesis":
                    key_value = int_.split(": ")
                    target_value = key_value[1].strip()
                    node_dict[key_value[0].strip()] = target_value
                else:
                    target_value = hypothesis

                llm_data.append((facts_llm, target_value))

            tree_info.append((sentences_all, node_dict, hypothesis, proofs, candidate_leaves))
    for info in tqdm(tree_info):
        get_paths(state_info=(info[1], info[2], info[3], info[4]))
        paths_info = dict()
        paths_info["node_dict"] = info[1]
        paths_info["hypothesis"] = info[2]
        paths_info["paths"] = global_paths_[:]
        paths_info["goals"] = retrieved_leaves(info[0], info[1], info[2], info[3], global_paths_)
        sorted_paths, path_label = path_judge(paths_info["paths"], paths_info["goals"])
        paths_info["sorted_paths"] = (sorted_paths, path_label)

        path_idx = 1
        for path, goal in zip(paths_info["paths"], paths_info["goals"]):
            path_idx += 1
        state_path_all.append(paths_info)
        global_control = 0
        global_paths_ = list()
    return state_path_all


def get_paths(state_info, state_seq=None, depth=0):
    if state_seq is None:
        state_seq = list()
    global global_control
    global global_paths_
    node_dict, hypothesis, proofs, candidate_leaves = state_info
    if len(proofs) == 0:
        global_paths_.append(state_seq[:])
        global_control += 1

    for proof_item in proofs[:]:
        int_ = proof_item.split("->")[1].strip()
        if int_ != "hypothesis":
            int_ = int_.split(": ")[0].strip()
        exp = int_

        basis = proof_item.split("->")[0].split(" & ")
        basis_true = 0
        for item in basis:
            item = item.strip()
            if item not in candidate_leaves:
                basis_true += 1
                exp = item
        if basis_true == 0 or (basis_true == 1 and (int_ == "hypothesis" or int_ in candidate_leaves)):
            state_seq.append(proofs)
            # update state
            new_proofs = proofs[:]
            new_proofs.remove(proof_item)
            new_candidate_leaves = candidate_leaves[:]
            if exp != "hypothesis":
                new_candidate_leaves.append(exp)
            new_state = node_dict, hypothesis, new_proofs, new_candidate_leaves
            get_paths(new_state, state_seq, depth + 1)
            if global_control >= 500:
                break
            state_seq.pop(-1)
        else:
            continue

def upd_angles_all():
    ref_path = "data/processed_data/slots/task_1-slots/bin_test.jsonl" 
    angle_path = "data/processed_data/angles/task_1/test.jsonl" 
    new_angle_path = "data/processed_data/angles/task_1/bin_test.jsonl" 
    upd_angles(ref_path, angle_path, new_angle_path)
    ref_path = "data/processed_data/slots/task_2-slots/bin_test.jsonl" 
    angle_path = "data/processed_data/angles/task_2/test.jsonl" 
    new_angle_path = "data/processed_data/angles/task_2/bin_test.jsonl" 
    upd_angles(ref_path, angle_path, new_angle_path)
    ref_path = "data/processed_data/slots/task_3-slots/bin_test.jsonl" 
    angle_path = "data/processed_data/angles/task_3/test.jsonl" 
    new_angle_path = "data/processed_data/angles/task_3/bin_test.jsonl" 
    upd_angles(ref_path, angle_path, new_angle_path)


def upd_angles(ref_path, angle_path, new_angle_path):
    ref_json_list = []
    with open(ref_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            tree_json_obj = json.loads(line)
            ref_json_list.append(tree_json_obj)

    ang_json_list = []
    with open(angle_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            tree_json_obj = json.loads(line)
            ang_json_list.append(tree_json_obj)

    new_ang_list = list()
    for ref_one, ang_one in zip(ref_json_list, ang_json_list):
        ang_one["output"] = "$proof$ = " + ref_one["proof"]
        ang_one_new = json.dumps(ang_one)
        new_ang_list.append(ang_one_new)
    
    # write
    write_iterate(new_ang_list, new_angle_path)


if __name__ == "__main__":
    # build_training_data("task_1")
    # build_training_data("task_2")
    # build_training_data("task_3")

    upd_angles_all()
