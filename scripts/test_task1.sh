cd ../code
bleurt_path=../bleurt/bleurt-large-512

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
module_types=single  # separate_all
exp_dir_single=../exp/Module_all/para_etree_all/Acdpaxg6
buffer_file=../exp/Module_all/buffer_dict_para_etree_all.json

cuda_id=0

## Step1:  select the checkpoint & the hyper-parameters using the dev split
# for beam_num in 1 2 5 10; # select the hyperparameters
# sed -i 's/\r$//' test_task1.sh

# exp_dir_controller=../exp/Controller_task1/rhetorical
# for iter in 438 876 1314 1752 2190 2628 3066 3504 3942 4380 4818 5256 5694 6132 6570 7008 7446 7884 8322 8760 9198 9636 10074 10512 10950;
# do
# echo $iter
# data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_1/dev.jsonl
# CUDA_VISIBLE_DEVICES=$cuda_id python reasoning_task1.py \
# --data_file $data_file --bleurt_path $bleurt_path --buffer_file $buffer_file \
# --module_types $module_types --exp_dir_single $exp_dir_single \
# --exp_dir_sub $exp_dir_sub --exp_dir_conj $exp_dir_conj --exp_dir_if $exp_dir_if \
# --exp_dir_sub_abd $exp_dir_sub_abd --exp_dir_conj_abd $exp_dir_conj_abd --exp_dir_if_abd $exp_dir_if_abd \
# --exp_dir_ded $exp_dir_ded --exp_dir_abd $exp_dir_abd \
# --exp_dir_controller $exp_dir_controller \
# --model_name_controller model_$iter.pth \
# --beam_num 1 --step_top_p 0.1 --step_top_p_abd 0.1  \
# --save_dir_name select_on_dev --save_details &&
# ls
# done

## Step2 run the selected checkpoint and hyperparameters on the test split
model_name_controller=model_438.pth  # model_task1.pth
exp_dir_controller=../exp/Controller_task1/rhetorical  # ../exp/Controller_task1/z7SOg44r
data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_1/test.jsonl
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
