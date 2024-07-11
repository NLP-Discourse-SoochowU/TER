#!/bin/sh
cd ../code
bleurt_path=../bleurt/bleurt-large-512

cuda_id=3

exp_dir_sub=None
exp_dir_conj=None
exp_dir_if=None
exp_dir_sub_abd=None
exp_dir_conj_abd=None
exp_dir_if_abd=None
exp_dir_single=None
exp_dir_ded=None
exp_dir_abd=None

# MetGen-prefixed
module_types=single
exp_dir_single=../exp/Module_all/para_etree_all/Acdpaxg6
buffer_file=../exp/Module_all/buffer_dict_para_etree_all.json


# ## Step1:  select the checkpoint & the hyper-parameters using the dev split
# data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_1/dev.jsonl
# for i in {1..5}
# do
#     exp_dir_controller=../exp/Controller_task1/retrieve_learning_v5/round_$i
#     for iter in 438 876 1314 1752 2190;
#     do
#     CUDA_VISIBLE_DEVICES=$cuda_id python reasoning_task1.py \
#     --data_file $data_file --bleurt_path $bleurt_path --buffer_file $buffer_file \
#     --module_types $module_types --exp_dir_single $exp_dir_single \
#     --exp_dir_sub $exp_dir_sub --exp_dir_conj $exp_dir_conj --exp_dir_if $exp_dir_if \
#     --exp_dir_sub_abd $exp_dir_sub_abd --exp_dir_conj_abd $exp_dir_conj_abd --exp_dir_if_abd $exp_dir_if_abd \
#     --exp_dir_ded $exp_dir_ded --exp_dir_abd $exp_dir_abd \
#     --exp_dir_controller $exp_dir_controller \
#     --model_name_controller model_$iter.pth \
#     --beam_num 1 --step_top_p 0.1 --step_top_p_abd 0.1  \
#     --save_dir_name select_on_dev --save_details &&
#     ls
#     done
# done

## Step2 run the selected checkpoint and hyperparameters on the test split
data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_1/test.jsonl
round=1
for iter in 876 2190 1314 2190 1314;
do
model_name_controller=model_$iter.pth
exp_dir_controller=../exp/Controller_task1/retrieve_learning_v5/round_$round
beam_num=10
CUDA_VISIBLE_DEVICES=$cuda_id python reasoning_task1.py \
--data_file $data_file --bleurt_path $bleurt_path --buffer_file $buffer_file \
--module_types $module_types --exp_dir_single $exp_dir_single \
--exp_dir_sub $exp_dir_sub --exp_dir_conj $exp_dir_conj --exp_dir_if $exp_dir_if \
--exp_dir_sub_abd $exp_dir_sub_abd --exp_dir_conj_abd $exp_dir_conj_abd --exp_dir_if_abd $exp_dir_if_abd \
--exp_dir_ded $exp_dir_ded --exp_dir_abd $exp_dir_abd \
--exp_dir_controller $exp_dir_controller \
--model_name_controller $model_name_controller \
--beam_num $beam_num --step_top_p 0.1 --step_top_p_abd 0.1  \
--save_dir_name reproduce_task1 --save_details
round=$((round+1))
done
# --retrieve_learning   use this after retrieving only, for most cases, I will not try testing retrieve module any more