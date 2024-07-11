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
module_types=single
exp_dir_single=../exp/Module_all/para_etree_all/Acdpaxg6
buffer_file=../exp/Module_all/buffer_dict_llm.json

cuda_id=1

# ## Step1:  select the checkpoint using the dev split 
# ##         & select the hyperparameters of reasoning algorithm using the dev split 
# exp_dir_controller=../exp/Controller_task2/rhetorical
# data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_2/dev.jsonl
# # select the checkpoint
# # sed -i 's/\r$//' test_task2.sh
# # for beam_num in 1 2 5 10; # select the hyperparameters
# # for fact_score_thre in 0.001 0.01 0.1 0.2 0.5 0.7; # select the hyperparameters
# for iter in 438 876 1314 1752 2190 2628 3066 3504 3942 4380 4818 5256 5694 6132 6570 7008 7446 7884 8322 8760 9198 9636 10074 10512 10950;
# do
# CUDA_VISIBLE_DEVICES=$cuda_id python reasoning_task2_llm.py \
# --data_file $data_file --bleurt_path $bleurt_path --buffer_file $buffer_file \
# --module_types $module_types --exp_dir_single $exp_dir_single \
# --exp_dir_sub $exp_dir_sub --exp_dir_conj $exp_dir_conj --exp_dir_if $exp_dir_if \
# --exp_dir_sub_abd $exp_dir_sub_abd --exp_dir_conj_abd $exp_dir_conj_abd --exp_dir_if_abd $exp_dir_if_abd \
# --exp_dir_ded $exp_dir_ded --exp_dir_abd $exp_dir_abd \
# --exp_dir_controller $exp_dir_controller \
# --model_name_controller model_$iter.pth \
# --beam_num 1 --step_top_p 0.1 --step_top_p_abd 0.1 --fact_score_thre 0.01 \
# --save_dir_name select_on_dev_llm --save_details &&
# ls
# done

## Step2 run the selected checkpoint and hyperparameters on the test split
model_name_controller=model_task2.pth  # model_9636.pth
exp_dir_controller=../exp/Controller_task2/NuCkQlfx  # ../exp/Controller_task2/rhetorical
data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_2/test.jsonl
beam_num=10
fact_score_thre=0.01
CUDA_VISIBLE_DEVICES=$cuda_id python reasoning_task2_llm.py \
--data_file $data_file --bleurt_path $bleurt_path --buffer_file $buffer_file \
--module_types $module_types --exp_dir_single $exp_dir_single \
--exp_dir_sub $exp_dir_sub --exp_dir_conj $exp_dir_conj --exp_dir_if $exp_dir_if \
--exp_dir_sub_abd $exp_dir_sub_abd --exp_dir_conj_abd $exp_dir_conj_abd --exp_dir_if_abd $exp_dir_if_abd \
--exp_dir_ded $exp_dir_ded --exp_dir_abd $exp_dir_abd \
--exp_dir_controller $exp_dir_controller \
--model_name_controller $model_name_controller \
--beam_num $beam_num --step_top_p 0.1 --step_top_p_abd 0.1 --fact_score_thre $fact_score_thre \
--save_dir_name reproduce_task2_llm --save_details
