cd ../code
data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_3/test.jsonl
gold_data_file=../data/entailment_trees_emnlp2021_data_v2/dataset/task_1/test.jsonl
bleurt_path=../../bleurt/bleurt-large-512

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

cuda_id=0

### Step1 run the selected checkpoint and hyperparameters on the test split
model_name_controller=model_438.pth  # model_task2.pth
exp_dir_controller=../exp/Controller_task2/rhetorical  # ../exp/Controller_task2/NuCkQlfx
beam_num=10
fact_score_thre=0.7
CUDA_VISIBLE_DEVICES=$cuda_id python reasoning_task3.py \
--task3_data_file $data_file --task1_gold_data_file $gold_data_file \
--bleurt_path $bleurt_path --buffer_file $buffer_file \
--module_types $module_types --exp_dir_single $exp_dir_single \
--exp_dir_sub $exp_dir_sub --exp_dir_conj $exp_dir_conj --exp_dir_if $exp_dir_if \
--exp_dir_sub_abd $exp_dir_sub_abd --exp_dir_conj_abd $exp_dir_conj_abd --exp_dir_if_abd $exp_dir_if_abd \
--exp_dir_ded $exp_dir_ded --exp_dir_abd $exp_dir_abd \
--exp_dir_controller $exp_dir_controller \
--model_name_controller $model_name_controller \
--beam_num $beam_num --step_top_p 0.1 --step_top_p_abd 0.1 \
--fact_score_thre $fact_score_thre --max_infer_depth 5 \
--save_dir_name reproduce_task3 --save_details